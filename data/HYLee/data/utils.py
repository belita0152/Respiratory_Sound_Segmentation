import os
from pathlib import Path

ROOT = os.path.join(os.getcwd(), "../..")
label_folder = os.path.join(ROOT,'raw', 'label_250520')  # 262 label files (server) / 744 (local)
data_folder = os.path.join(ROOT, 'raw', '241205')  # 2191 .wav files
save_path = os.path.join(ROOT, 'new_gt', 'raw')  # recent: raw segments

wav_files = list(Path(data_folder).rglob("*.wav"))
label_files = list(Path(label_folder).glob("*.xlsx"))

wav_ids = {p.stem: p for p in wav_files}

label_ids = {}
for p in label_files:
    sample_id = p.stem.split(" - ")[0].strip()
    label_ids[sample_id] = p
matched_ids = sorted(set(wav_ids.keys()) & set(label_ids.keys()))
missing_wav = sorted(set(label_ids.keys()) - set(wav_ids.keys()))
missing_label = sorted(set(wav_ids.keys()) - set(label_ids.keys()))

# print(wav_files)
# print(f"wav files: {len(wav_files)}")
# print(f"label files: {len(label_files)}")
# print(f"matched: {len(matched_ids)}")
# print(f"label exists but wav missing: {len(missing_wav)}")
# print(f"wav exists but label missing: {len(missing_label)}")
#
# print("\nExamples - label exists but wav missing:")
# print(missing_wav[:10])
#
# print("\nExamples - wav exists but label missing:")
# print(missing_label[:10])