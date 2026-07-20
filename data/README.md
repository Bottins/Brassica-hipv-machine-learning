# Data description

## Directory contents

### Raw data

`raw/gas_exchange_fluorescence.xlsx` contains sample identifiers, treatment codes and 20 gas-exchange or chlorophyll-fluorescence measurements. The source workbook contains one sheet, `Foglio1`.

`raw/voc_gcms.xlsx` contains GC-MS measurements for 20 volatile organic compounds. The analysis reads the sheet named `definitivo`.

`raw/leaf_area_damage.xlsx` contains consumed leaf area in square centimetres. The analysis uses the `raw` sheet; the other sheets in the source workbook contain intermediate summaries and are retained for provenance.

The raw workbooks are copies of the source data. Their internal sheet and column names have deliberately been preserved so that the preparation steps remain traceable.

### Processed data

`processed/receiver_plant_dataset.xlsx` is the table used by `src/run_analysis.py`. It contains four identifier columns and 41 numeric variables:

- 20 columns prefixed with `gas_`: gas exchange and chlorophyll fluorescence;
- 20 columns prefixed with `voc_`: GC-MS volatile profiles;
- `leaf_damage_cm2`: consumed leaf area.

Identifier columns:

| Column | Description |
|---|---|
| `merge_key` | Normalised key used to match records across workbooks |
| `run` | Experimental run |
| `id sample` | Original sample identifier from the physiological dataset |
| `labels` | Treatment code (`h0h0`, `h0h1`, `h1h0` or `h1h1`) |

The dataset contains 30 samples per treatment and has no missing values.

## Data preparation

`src/prepare_data.py` performs the following operations:

1. normalises sample identifiers and treatment codes;
2. applies the documented corrections required to reconcile a small number of source identifiers;
3. matches gas-exchange and GC-MS records by the corrected key;
4. averages leaf-damage records that map to the same sample;
5. assigns zero leaf damage when a matched receiver was not subjected to herbivory;
6. prefixes measurement columns by domain and writes the analysis-ready workbook.

The correction mappings are declared explicitly near the beginning of `prepare_data.py`; they are not applied silently to the raw files.
