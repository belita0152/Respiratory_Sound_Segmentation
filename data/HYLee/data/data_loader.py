from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence, Tuple
import librosa
from scipy.signal import butter, filtfilt

import numpy as np
import torch
from torch.utils.data import Dataset

from parser import LabelParser


def preprocessing(signal: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    # 1. Downsampling
    signal = librosa.resample(signal, orig_sr=sr, target_sr=target_sr)

    low_cut = 50
    high_cut = 2000
    order = 4
    nyquist = 0.5 * target_sr
    low = low_cut / nyquist
    high = high_cut / nyquist

    # 2. Bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # 3. Rectification (absolute values)
    filtered = filtfilt(b, a, signal)  # zero-phase filtering to avoid the phase shift issue
    rectified = np.abs(filtered)  # transform to absolute values

    return rectified, target_sr


def sliding_window_1d(
    arr: np.ndarray,
    fs: int,
    window_sec: float,
    step_sec: float,
) -> np.ndarray:

    window_size = int(round(window_sec * fs))
    step_size = int(round(step_sec * fs))

    if len(arr) < window_size:
        return np.empty((0, window_size), dtype=arr.dtype)

    starts = range(0, len(arr) - window_size + 1, step_size)
    return np.stack([arr[start:start + window_size] for start in starts], axis=0)  # (n_windows, window_size)


class SlidingWindowDataset(Dataset):
    def __init__(
        self,
        wav_base_path: str,
        label_base_path: str,
        *,
        down_sampling: bool = True,
        train: bool = True,
        train_ratio: float = 0.8,
        target_sr: int = 16000,
        window_sec: float = 8.0,
        step_sec: float = 4.0,  # 50% overlap
        sheet_name: str | int = 1,
        start_col: str = "cycle_start_time",
        end_col: str = "cycle_end_time",
        label_col: str = "labels",
    ):
        super().__init__()
        self.wav_base_path = wav_base_path
        self.label_base_path = label_base_path
        self.target_sr = target_sr
        self.window_sec = window_sec
        self.step_sec = step_sec
        self.down_sampling = down_sampling

        self.target_sr = target_sr
        self.sheet_name = sheet_name
        self.start_col = start_col
        self.end_col = end_col
        self.label_col = label_col
        self.parser = LabelParser(
            wav_base_path,
            label_base_path,
            sheet_name=sheet_name,
            start_col=start_col,
            end_col=end_col,
            label_col=label_col,
        )

        matched = self.parser.match_files()
        self.items = sorted(matched.values(), key=lambda item: str(item[0]))
        if len(self.items) == 0:
            raise FileNotFoundError(
                f"No matched wav/xlsx pairs under: {wav_base_path} and {label_base_path}"
            )

        split = int(len(self.items) * float(train_ratio))
        target_items = self.items[:split] if train else self.items[split:]
        self.data_arr, self.mask_arr = self._load_and_window_all(target_items)

    def _load_one(self, data_path, label_path):
        data_, sr = librosa.load(data_path, sr=None)
        data, down_sr = preprocessing(data_, int(sr), self.target_sr)

        label = self.parser.make_label_array(
            label_path=label_path,
            sr=down_sr,
            n_samples=len(data),
        )

        return data, label, int(down_sr)

    def _window_one(self, sig_1d: np.ndarray, label_1d: np.ndarray, sr: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        sig_w = sliding_window_1d(sig_1d, sr, self.window_sec, self.step_sec)
        label_w = sliding_window_1d(label_1d, sr, self.window_sec, self.step_sec)

        n = min(sig_w.shape[0], label_w.shape[0])  # select the minimum number of windows
        if n == 0:
            w = int(round(self.window_sec * sr))
            return (
                np.empty((0, w), dtype=np.float32),
                np.empty((0, w), dtype=np.int64),
            )

        return sig_w[:n], label_w[:n]

    def _load_and_window_all(self, items: Sequence[Tuple[Path, Path]]) -> Tuple[np.ndarray, np.ndarray]:
        data_chunks = []
        mask_chunks = []

        for wav_path, label_path in items:
            signal_1d, mask_1d, sr = self._load_one(wav_path, label_path)
            signal_w, mask_w = self._window_one(signal_1d, mask_1d, sr)

            if self.down_sampling:
                keep_idx = np.any(mask_w >= 0, axis=-1)
                signal_w = signal_w[keep_idx]
                mask_w = mask_w[keep_idx]

            data_chunks.append(signal_w)
            mask_chunks.append(mask_w)

        data_arr = (
            np.concatenate(data_chunks, axis=0)
            if len(data_chunks)
            else np.empty((0, 0), dtype=np.float32)
        )

        mask_arr = (
            np.concatenate(mask_chunks, axis=0)
            if len(mask_chunks)
            else np.empty((0, 0), dtype=np.int64)
        )

        return data_arr, mask_arr

    def __len__(self) -> int:
        return self.mask_arr.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        signal = torch.tensor(self.data_arr[idx], dtype=torch.float32)
        mask = torch.tensor(self.mask_arr[idx], dtype=torch.long)
        return signal, mask



# -----------------------------------------------------------------------------
# Concrete Datasets
# -----------------------------------------------------------------------------

class LungSoundDataset(SlidingWindowDataset):
    """SNUCH Child Lung Sound Dataset"""

    def __init__(
        self,
        wav_base_path: str,
        label_base_path: str,
        *,
        down_sampling: bool = True,
        train: bool = True,
        train_ratio: float = 0.8,
        target_sr: int = 16000,
        window_sec: float = 8.0,
        step_sec: float = 4.0,
    ):
        super().__init__(
            wav_base_path,
            label_base_path,
            down_sampling=down_sampling,
            train=train,
            train_ratio=train_ratio,
            target_sr=target_sr,
            window_sec=window_sec,
            step_sec=step_sec,
            sheet_name=1,
            start_col="cycle_start_time",
            end_col="cycle_end_time",
            label_col="labels",
        )


if __name__ == "__main__":
    root = os.path.join(os.getcwd(), "..", "..")
    label_folder = os.path.join(root, "raw", "label_250520")
    data_folder = os.path.join(root, "raw", "241205")

    train_dataset = LungSoundDataset(
        data_folder,
        label_folder,
        train=True,
        train_ratio=0.8,
        down_sampling=True,
        target_sr=16000,
        window_sec=8.0,
        step_sec=4.0,
    )

    print(f"train windows: {len(train_dataset)}")
    print(f"train data shape: {train_dataset.data_arr.shape}")
    print(f"train mask shape: {train_dataset.mask_arr.shape}")

    if len(train_dataset) > 0:
        signal, mask = train_dataset[0]
        values, counts = torch.unique(mask, return_counts=True)
        label_counts = {
            int(value): int(count)
            for value, count in zip(values, counts)
        }

        print("\nfirst train sample")
        print(f"signal.shape: {tuple(signal.shape)}")
        print(f"mask.shape: {tuple(mask.shape)}")
        print(f"mask label counts: {label_counts}")