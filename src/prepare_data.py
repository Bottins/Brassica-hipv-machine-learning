"""Build the analysis dataset from the three original measurement tables."""

from pathlib import Path
import re

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_DIR / "data" / "raw"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "receiver_plant_dataset.xlsx"

GAS_FILE = RAW_DIR / "gas_exchange_fluorescence.xlsx"
VOC_FILE = RAW_DIR / "voc_gcms.xlsx"
LEAF_FILE = RAW_DIR / "leaf_area_damage.xlsx"

ID_COLS = ["merge_key", "run", "id sample", "labels"]

# Corrections established during reconciliation of the experimental records.
GAS_KEY_CORRECTIONS = {
    "3_6r1h0h1": "3_6r1h0h0",
    "4_1r2h1hh1": "4_1r2h1h1",
    "5_1r1h0h0": "5_1r1h0h0",
}

VOC_KEY_CORRECTIONS = {
    "2_7r2h0h1": "2_7r2h1h1",
    "2_9r2h1h1": "2_9r2h0h1",
}


def clean_sample_token(value):
    token = str(value).strip().replace("_", "").replace(" ", "").lower()
    return token.replace("ho", "h0").replace("hh", "h")


def make_merge_key(run, sample):
    return f"{int(run)}_{clean_sample_token(sample)}"


def label_from_key(key):
    match = re.search(r"(h[01]h[01])$", str(key).lower())
    return match.group(1) if match else None


def normalize_leaf_key(run, label):
    token = clean_sample_token(label)
    match = re.match(r"(?P<number>\d+)(?:r(?P<replicate>[12]))?(?P<label>h[01]h[01])$", token)
    if not match:
        return None

    number = match.group("number")
    replicate = match.group("replicate")
    label_code = match.group("label")
    if replicate is None:
        replicate = "1" if label_code in {"h0h0", "h1h0"} else "2"
    return f"{int(run)}_{number}r{replicate}{label_code}"


def prepare_gas_data():
    df = pd.read_excel(GAS_FILE)
    df["merge_key_raw"] = [make_merge_key(run, sample) for run, sample in zip(df["run"], df["id sample"])]
    df["merge_key"] = df["merge_key_raw"].replace(GAS_KEY_CORRECTIONS)
    df["labels"] = df["labels"].astype(str).str.lower()

    key_labels = df["merge_key"].map(label_from_key)
    mismatches = df.loc[key_labels != df["labels"], ["run", "id sample", "labels", "merge_key"]]
    if not mismatches.empty:
        raise ValueError(f"Gas labels do not match corrected keys:\n{mismatches.to_string(index=False)}")

    excluded = {"run", "id sample", "labels", "merge_key_raw", "merge_key"}
    features = [column for column in df.columns if column not in excluded]
    columns = ["merge_key", "merge_key_raw", "run", "id sample", "labels"] + features
    return df[columns].rename(columns={column: f"gas_{column}" for column in features})


def prepare_voc_data():
    df = pd.read_excel(VOC_FILE, sheet_name="definitivo")
    df["merge_key_raw"] = [make_merge_key(run, sample) for run, sample in zip(df["run"], df["sample"])]
    df["merge_key"] = df["merge_key_raw"].replace(VOC_KEY_CORRECTIONS)
    df["labels_voc"] = df["merge_key"].map(label_from_key)

    excluded = {"run", "sample", "group", "merge_key_raw", "merge_key", "labels_voc"}
    features = [column for column in df.columns if column not in excluded]
    columns = ["merge_key", "merge_key_raw", "run", "sample", "group", "labels_voc"] + features
    return df[columns].rename(columns={column: f"voc_{column}" for column in features})


def prepare_leaf_damage():
    df = pd.read_excel(LEAF_FILE, sheet_name="raw")
    df["merge_key"] = [normalize_leaf_key(run, label) for run, label in zip(df["run"], df["label"])]
    return (
        df.dropna(subset=["merge_key"])
        .groupby("merge_key", as_index=False)["cm2"]
        .mean()
        .rename(columns={"cm2": "leaf_damage_cm2"})
    )


def main():
    gas = prepare_gas_data()
    voc = prepare_voc_data()
    leaf = prepare_leaf_damage()

    merged = gas.merge(voc.drop(columns=["run"]), on="merge_key", how="inner", suffixes=("", "_voc"))
    mismatches = merged.loc[merged["labels"] != merged["labels_voc"], ["merge_key", "labels", "labels_voc"]]
    if not mismatches.empty:
        raise ValueError(f"Gas/VOC labels do not match after correction:\n{mismatches.to_string(index=False)}")

    merged = merged.merge(leaf, on="merge_key", how="left")
    merged["leaf_damage_cm2"] = merged["leaf_damage_cm2"].fillna(0.0)
    drop_columns = ["merge_key_raw", "merge_key_raw_voc", "sample", "group", "labels_voc"]
    merged = merged.drop(columns=[column for column in drop_columns if column in merged.columns])
    features = [column for column in merged.columns if column not in ID_COLS]
    merged = merged[ID_COLS + features].sort_values(["run", "merge_key"]).reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved {len(merged)} samples and {len(features)} features to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
