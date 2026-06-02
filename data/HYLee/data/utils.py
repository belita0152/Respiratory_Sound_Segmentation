import os
from pathlib import Path
import pandas as pd

ROOT = os.path.abspath(os.path.join(Path(__file__).resolve().parent, '..', '..'))
label_folder = os.path.join(ROOT, 'raw', 'label_250520')  # 744 label files
data_folder = os.path.join(ROOT, 'raw', 'wav')  # 2191 .wav files
save_path = os.path.join(ROOT, 'new_gt', 'raw')  # recent: raw segments

# Cohort exclusion criteria:
# 1. Exclude manually listed None-only or Unknown-only label files.
# 2. Exclude samples collected on or before 2024-08-27.
# 3. Exclude samples marked Yes in HFNC, T-can, or Home vent.
screening_file = os.path.join(os.getcwd(), '..', 'test', '전자청진음_기기착용_2024.9-2026.4.txt')
screening_columns = ["HFNC", "T-can", "Home vent"]
performed_at_column = "\uac80\uc0ac\uc2dc\ud589\uc77c"
cutoff_date = pd.Timestamp("2024-08-27")

# Exclusion 1: None-only or Unknown-only label files
none_file_names = {
    "240807_0924 - 박지수.xlsx",  # None-only or included
    "240807_0935 - 박지수.xlsx",
    "240807_0942 - 박지수.xlsx",
    "240807_1002 - 박지수.xlsx",
    "240807_1019 - 박지수.xlsx",
    "240807_1026 - 박지수.xlsx",
    "240814_1647 - 유희석.xlsx",
    "240930_1653 - 유희석.xlsx",
    "240807_0914 - 박지수.xlsx",
    "240807_0956 - 박지수.xlsx",
    "240807_1037 - 박지수.xlsx",
    "240807_1041 - 박지수.xlsx",
    "240808_1244 - 박지수.xlsx",
    "240819_1632 - 박지수.xlsx",
    "240819_1650 - 박지수.xlsx",
    "240906_1605 - 문재원.xlsx",
    "241108_1450 - 유희석.xlsx",  # Unknown-only
    "240808_1252 - 문재원.xlsx",
}

none_ids = {Path(file_name).stem.split(" - ")[0].strip() for file_name in none_file_names}


def sample_id_to_datetime(sample_id):
    return pd.to_datetime(sample_id, format="%y%m%d_%H%M", errors="coerce")


def get_ids_until_cutoff(sample_ids, cutoff):
    excluded_ids = set()
    for sample_id in sample_ids:
        sample_datetime = sample_id_to_datetime(sample_id)
        if pd.notna(sample_datetime) and sample_datetime.normalize() <= cutoff:
            excluded_ids.add(sample_id)

    return excluded_ids


def get_screening_excluded_ids(screening_path):
    screening_df = pd.read_csv(screening_path, encoding="utf-8-sig")
    screening_df.columns = screening_df.columns.str.strip()

    yes_mask = (
        screening_df[screening_columns]
        .astype(str)
        .apply(lambda col: col.str.strip().str.casefold().eq("yes"))
        .any(axis=1)
    )
    performed_at = pd.to_datetime(screening_df.loc[yes_mask, performed_at_column], errors="coerce")
    return set(performed_at.dropna().dt.strftime("%y%m%d_%H%M"))


wav_files = list(Path(data_folder).rglob("*.wav"))
wav_ids = {p.stem: p for p in wav_files}

label_files = [
    p for p in Path(label_folder).glob("*.xlsx")
    if not p.name.startswith("~$")
    and p.name.strip() not in none_file_names
]


label_ids = {}
for p in label_files:
    sample_id = p.stem.split(" - ")[0].strip()
    label_ids[sample_id] = p

candidate_ids = set(wav_ids.keys()) & set(label_ids.keys())
old_matched_ids = sorted(candidate_ids - none_ids)

# Exclusion 2: sample_id date is on or before 2024-08-27
cutoff_excluded_ids = get_ids_until_cutoff(candidate_ids, cutoff_date)

# Exclusion 3: any device column is Yes in the screening file
screening_excluded_ids = get_screening_excluded_ids(screening_file)

cutoff_excluded_ids = sorted(set(old_matched_ids) & cutoff_excluded_ids)
device_excluded_ids = sorted((set(old_matched_ids) - set(cutoff_excluded_ids)) & screening_excluded_ids)
screening_overlap_with_cutoff_ids = sorted(set(cutoff_excluded_ids) & screening_excluded_ids)

cutoff_excluded_files = [label_ids[sample_id].name for sample_id in cutoff_excluded_ids]
device_excluded_files = [label_ids[sample_id].name for sample_id in device_excluded_ids]

excluded_ids = none_ids | set(cutoff_excluded_ids) | screening_excluded_ids

matched_ids = sorted(candidate_ids - excluded_ids)  # exclude files which include 'none' and 'unknown' label
missing_wav = sorted(set(label_ids.keys()) - set(wav_ids.keys()))
missing_label = sorted(set(wav_ids.keys()) - set(label_ids.keys()))


if __name__ == "__main__":
    print(len(list(Path(data_folder).rglob("*.wav"))))
    print(len(list(Path(label_folder).glob("*.xlsx"))))
    print(len(matched_ids))

    print(f"old matched files: {len(old_matched_ids)}")
    print(f"cutoff excluded files: {len(cutoff_excluded_files)}")
    print(f"device excluded files: {len(device_excluded_files)}")
    print(f"device ids already excluded by cutoff: {len(screening_overlap_with_cutoff_ids)}")

    print("\n[cutoff excluded files]")
    for file_name in cutoff_excluded_files:
        print(file_name)

    print("\n[device excluded files]")
    for file_name in device_excluded_files:
        print(file_name)
