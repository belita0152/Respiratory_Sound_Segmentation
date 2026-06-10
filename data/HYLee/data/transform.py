from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

"""
Preprocessed signal x --> 3 types of input (Signal / Wavelet scalogram / Mel spectrogram)


* Structure
Preprocessed signal window [T]
        |
        v
transform_window()
        |
        +-- signal            -> [1, T]
        +-- wavelet           -> [1, n_scales, T]
        +-- mel_spectrogram   -> [1, n_mels, n_frames]

"""

TransformKind = Literal["signal", "wavelet", "mel"]


@dataclass
class WindowTransform:
    kind: TransformKind = "signal"
    sample_rate: int = 16000
    normalize: bool = True
    output_channels: int = 1

    # STFT / mel parameters.
    n_fft: int = 512
    hop_length: int = 128
    win_length: int = 512
    n_mels: int = 128
    f_min: float = 50
    f_max: float = 2000

    # CWT parameters.
    cwt_cycles: float = 6.0
    voices_per_octave: int = 10

    def __call__(self, window, mask):
        window = self._as_1d_float_tensor(window)
        mask = torch.as_tensor(mask, dtype=torch.long)

        if self.normalize:
            window = self._peak_normalize(window)

        if self.kind == "signal":
            return window.unsqueeze(0), mask
        if self.kind == "wavelet":
            return self._scalogram(window), mask
        if self.kind == "mel":
            mel = self._mel_spectrogram(window)
            return mel, self._sample_mask_at_mel_frames(mask)
        raise ValueError(f"Unknown transform kind: {self.kind}")

    def _scalogram(self, window: torch.Tensor) -> torch.Tensor:
        max_hz = min(float(self.f_max), self.sample_rate / 2.0)
        min_hz = float(self.f_min)
        octaves = torch.log2(torch.tensor(max_hz / min_hz)).item()
        num_scales = max(8, int(round(octaves * self.voices_per_octave)))

        freqs = torch.logspace(
            torch.log10(torch.tensor(max_hz)),
            torch.log10(torch.tensor(min_hz)),
            num_scales,
            device=window.device,
            dtype=window.dtype,
        )  # 실제 사용할 주파수 list

        power = self._morlet_cwt_power(window, freqs)
        image = self._log_normalize(power)
        return self._add_channels(image)

    def _mel_spectrogram(self, window: torch.Tensor) -> torch.Tensor:
        stft = torch.stft(
            window,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=torch.hann_window(self.win_length, device=window.device),
            center=True,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        mel_filter = self._mel_filterbank(device=power.device, dtype=power.dtype)
        mel = mel_filter @ power
        image = self._log_normalize(mel)
        return self._add_channels(image)

    @staticmethod
    def _as_1d_float_tensor(window: torch.Tensor) -> torch.Tensor:
        window = torch.as_tensor(window, dtype=torch.float32)
        if window.ndim == 2 and window.shape[0] == 1:
            window = window.squeeze(0)
        if window.ndim != 1:
            raise ValueError(
                f"Expected one window with shape [T] or [1, T], got {tuple(window.shape)}"
            )
        return window

    @staticmethod
    def _peak_normalize(window: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return window / window.abs().amax().clamp_min(eps)

    @staticmethod
    def _log_normalize(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        image = torch.log1p(image)
        image = image - image.amin()
        return image / image.amax().clamp_min(eps)

    def _add_channels(self, image: torch.Tensor) -> torch.Tensor:
        image = image.unsqueeze(0)
        if self.output_channels == 1:
            return image
        return image.repeat(self.output_channels, 1, 1)

    def _sample_mask_at_mel_frames(self, mask: torch.Tensor) -> torch.Tensor:
        frame_count = int(mask.numel() // self.hop_length) + 1
        frame_centers = torch.arange(frame_count, device=mask.device) * self.hop_length
        frame_centers = frame_centers.clamp(max=mask.numel() - 1)
        return mask[frame_centers]

    def _mel_filterbank(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        f_max = min(float(self.f_max), self.sample_rate / 2.0)
        fft_freqs = torch.linspace(
            0,
            self.sample_rate / 2.0,
            self.n_fft // 2 + 1,
            device=device,
            dtype=dtype,
        )
        mel_points = torch.linspace(
            self._hz_to_mel(torch.tensor(self.f_min, device=device, dtype=dtype)),
            self._hz_to_mel(torch.tensor(f_max, device=device, dtype=dtype)),
            self.n_mels + 2,
            device=device,
            dtype=dtype,
        )
        hz_points = self._mel_to_hz(mel_points)

        lower = hz_points[:-2].unsqueeze(1)
        center = hz_points[1:-1].unsqueeze(1)
        upper = hz_points[2:].unsqueeze(1)
        freqs = fft_freqs.unsqueeze(0)

        left = (freqs - lower) / (center - lower).clamp_min(1e-8)
        right = (upper - freqs) / (upper - center).clamp_min(1e-8)
        filters = torch.minimum(left, right).clamp_min(0.0)
        return filters / filters.sum(dim=1, keepdim=True).clamp_min(1e-8)

    @staticmethod
    def _hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
        return 2595.0 * torch.log10(1.0 + freq / 700.0)

    @staticmethod
    def _mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
        return 700.0 * (torch.pow(10.0, mel / 2595.0) - 1.0)

    def _morlet_cwt_power(self, window: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
        min_freq = freqs.min().clamp_min(1e-8)
        max_sigma = self.cwt_cycles / (2.0 * torch.pi * min_freq)
        half_width = int(torch.ceil(3.0 * max_sigma * self.sample_rate).item())
        half_width = max(half_width, 8)

        t = (
                torch.arange(
                    -half_width,
                    half_width + 1,
                    device=window.device,
                    dtype=window.dtype,
                )
                / self.sample_rate
        )
        sigma = self.cwt_cycles / (2.0 * torch.pi * freqs).unsqueeze(1)
        phase = 2.0 * torch.pi * freqs.unsqueeze(1) * t.unsqueeze(0)
        envelope = torch.exp(-(t.unsqueeze(0).pow(2)) / (2.0 * sigma.pow(2)))
        real = torch.cos(phase) * envelope
        imag = torch.sin(phase) * envelope

        real = real - real.mean(dim=1, keepdim=True)
        imag = imag - imag.mean(dim=1, keepdim=True)
        real = real / real.norm(dim=1, keepdim=True).clamp_min(1e-8)
        imag = imag / imag.norm(dim=1, keepdim=True).clamp_min(1e-8)

        x = window.view(1, 1, -1)
        real_out = F.conv1d(x, real.unsqueeze(1), padding=half_width).squeeze(0)
        imag_out = F.conv1d(x, imag.unsqueeze(1), padding=half_width).squeeze(0)
        return real_out.pow(2.0) + imag_out.pow(2.0)


class TransformedWindowDataset(Dataset):
    """Apply a window transform to an existing `(signal, mask)` dataset."""

    def __init__(self, base_dataset: Dataset, transform: WindowTransform | TransformKind) -> None:
        self.base_dataset = base_dataset
        self.transform = (
            WindowTransform(kind=transform)
            if isinstance(transform, str)
            else transform
        )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        signal, mask = self.base_dataset[idx]
        return self.transform(signal, mask)


def transform_window(
        window: torch.Tensor,
        mask: torch.Tensor,
        config: WindowTransform | TransformKind,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = WindowTransform(kind=config) if isinstance(config, str) else config
    return transform(window, mask)


if __name__ == "__main__":
    sample_rate = 16000
    window_sec = 8
    window = torch.randn(sample_rate * window_sec)
    mask = torch.zeros(sample_rate * window_sec, dtype=torch.long)

    for kind in ("signal", "wavelet", "mel"):
        transform = WindowTransform(kind=kind, sample_rate=sample_rate)
        transformed, transformed_mask = transform(window, mask)
        print(
            f"{kind:15s} "
            f"data.shape={tuple(transformed.shape)} "
            f"mask.shape={tuple(transformed_mask.shape)} "
            f"dtype={transformed.dtype}"
        )
