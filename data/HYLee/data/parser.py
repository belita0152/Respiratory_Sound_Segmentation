from __future__ import annotations

import os
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils import matched_ids as cohort_matched_ids


class LabelParser:
    def __init__(
        self,
        wav_base_path: str | Path,
        label_base_path: str | Path,
        *,
        sheet_name: str | int = 1,
        start_col: str = "cycle_start_time",
        end_col: str = "cycle_end_time",
        label_col: str = "labels",
        label_map: Optional[Dict[str, int]] = None,
        background_label: int = -1,
    ):
        self.wav_base_path = Path(wav_base_path)
        self.label_base_path = Path(label_base_path)
        self.sheet_name = sheet_name
        self.start_col = start_col
        self.end_col = end_col
        self.label_col = label_col
        self.label_map = label_map or {
            "Normal": 0,
            "Stridor": 1,
            "Rhonchi": 2,
            "Wheezing": 3,
            "Crackle": 4,
        }
        self.background_label = background_label
        self.matched_ids = sorted(str(sample_id) for sample_id in cohort_matched_ids)

    @staticmethod
    def extract_match_key(path: str | Path) -> str:
        path = Path(path)
        if path.suffix.lower() == ".xlsx":
            return path.stem.split(" - ")[0].strip()
        return path.stem.strip()

    def build_wav_index(self) -> Dict[str, Path]:
        wav_files = sorted(self.wav_base_path.rglob("*.wav"))
        wav_groups: Dict[str, List[Path]] = {}
        wav_ids: Dict[str, Path] = {}

        for path in wav_files:
            sample_id = self.extract_match_key(path)
            wav_groups.setdefault(sample_id, []).append(path)

        for sample_id, paths in wav_groups.items():
            if len(paths) == 1:
                wav_ids[sample_id] = paths[0]
                continue

            valid_paths = [
                path for path in paths
                if path.parent.name == sample_id.split("_")[0]
            ]

            # if len(valid_paths) == 1:
            #     wav_ids[sample_id] = valid_paths[0]
            #     skipped_paths = [path for path in paths if path != valid_paths[0]]
            #     print(f"[DUPLICATE wav] use: {valid_paths[0]}")
            #     for path in skipped_paths:
            #         print(f"[DUPLICATE wav] pass: {path}")
            # else:
            #     print(f"[DUPLICATE wav] pass all id: {sample_id}")
            #     for path in paths:
            #         print(f"  wav: {path}")

        return wav_ids

    def build_label_index(self) -> Dict[str, Path]:
        label_files = sorted(
            path
            for path in self.label_base_path.glob("*.xlsx")
            if not path.name.startswith("~$")
        )
        label_ids: Dict[str, Path] = {}

        for path in label_files:
            sample_id = self.extract_match_key(path)
            label_ids[sample_id] = path

        return label_ids

    def match_files(self) -> Dict[str, Tuple[Path, Path]]:
        wav_ids = self.build_wav_index()
        label_ids = self.build_label_index()
        matched_ids = [
            sample_id for sample_id in self.matched_ids
            if sample_id in wav_ids and sample_id in label_ids
        ]

        return {
            sample_id: (wav_ids[sample_id], label_ids[sample_id])
            for sample_id in matched_ids
        }

    def summarize_matching(self) -> Dict[str, object]:
        wav_ids = self.build_wav_index()
        label_ids = self.build_label_index()
        matched_ids = [
            sample_id for sample_id in self.matched_ids
            if sample_id in wav_ids and sample_id in label_ids
        ]
        cohort_ids = set(self.matched_ids)
        cohort_missing_wav = sorted(cohort_ids - set(wav_ids.keys()))
        cohort_missing_label = sorted(cohort_ids - set(label_ids.keys()))
        missing_wav = sorted(set(label_ids.keys()) - set(wav_ids.keys()))
        missing_label = sorted(set(wav_ids.keys()) - set(label_ids.keys()))

        return {
            "num_wav": len(wav_ids),
            "num_label": len(label_ids),
            "num_matched": len(matched_ids),
            "matched_ids": matched_ids,
            "missing_wav": missing_wav,
            "missing_label": missing_label,
            "cohort_missing_wav": cohort_missing_wav,
            "cohort_missing_label": cohort_missing_label,
        }

    @staticmethod
    def get_wav_info(wav_path: str | Path) -> Tuple[int, int]:
        with wave.open(str(wav_path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            num_samples = wav_file.getnframes()
        return sample_rate, num_samples

    def parse_label_value(self, raw_label) -> int:
        if pd.isna(raw_label):
            return self.background_label

        label_text = str(raw_label).strip()
        if label_text == "":
            return self.background_label

        if label_text.lower() in {"none", "unknown", "nan"}:
            return self.background_label

        parts = (
            label_text.replace("\n", ",")
            .replace("/", ",")
            .replace(";", ",")
            .replace("+", ",")
            .split(",")
        )
        labels = [part.strip() for part in parts if part.strip()]

        if len(labels) == 1:
            return self.label_map.get(labels[0], self.background_label)

        return self.background_label

    def make_label_array(self, label_path: str | Path, sr: int, n_samples: int,
        ) -> np.ndarray:

        mask = np.full(n_samples, self.background_label, dtype=np.int64)
        label_df = pd.read_excel(label_path, sheet_name=self.sheet_name)

        for _, row in label_df.iterrows():
            if pd.isna(row[self.start_col]) or pd.isna(row[self.end_col]):
                continue

            label_id = self.parse_label_value(row[self.label_col])

            start_idx = max(0, int(round(float(row[self.start_col]) * sr)))
            end_idx = min(n_samples, int(round(float(row[self.end_col]) * sr)))

            if end_idx <= start_idx:
                continue

            mask[start_idx:end_idx] = label_id

        return mask


if __name__ == "__main__":
    root = os.path.join(os.getcwd(), "..")
    label_folder = os.path.join(root, "raw", "label_250520")
    data_folder = os.path.join(root, "raw", "wav")

    parser = LabelParser(data_folder, label_folder)
    summary = parser.summarize_matching()
    matched = parser.match_files()

    print(f"wav files: {summary['num_wav']}")
    print(f"label files: {summary['num_label']}")
    print(f"matched: {summary['num_matched']}")
    print()

    # for sample_id, (wav_path, label_path) in matched.items():
    #     sample_rate, num_samples = parser.get_wav_info(wav_path)
    #     label_arr = parser.make_label_array(label_path, sample_rate, num_samples)
    #     values, counts = np.unique(label_arr, return_counts=True)
    #     label_counts = {
    #         int(value): int(count)
    #         for value, count in zip(values, counts)
    #     }
    #
    #     print(f"[{sample_id}]")
    #     print(f"  wav: {wav_path.name}")
    #     print(f"  label: {label_path.name}")
    #     print(f"  sample_rate: {sample_rate}")
    #     print(f"  duration_sec: {num_samples / sample_rate:.3f}")
    #     print(f"  data.shape: {tuple([num_samples])}")
    #     print(f"  label.shape: {label_arr.shape}")
    #     print(f"  label_counts: {label_counts}")
