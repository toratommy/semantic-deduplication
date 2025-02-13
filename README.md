# Fuzzy and Semantic Deduplication Pipelines

This repository contains a **modular** pipeline for **fuzzy deduplication** of two datasets (Dataset 1 and Dataset 2) that must be merged into a unified format (Dataset 1’s schema) and de-duplicated. The approach uses:

- [**recordlinkage**](https://recordlinkage.readthedocs.io/) for **blocking** and fuzzy matching  
- [**rapidfuzz**](https://github.com/maxbachmann/rapidfuzz) for string similarity  
- **YAML** config for easy customization of paths, thresholds, and column mappings  

## Repository Structure

```
.
├── config.yml
├── src
│   └── fuzzy_helpers.py
├── fuzzy_deduplication.ipynb
├── data
│   ├── dataset1.csv
│   └── dataset2.csv
├── requirements.txt
└── README.md
```

1. **`config.yml`**  
   Stores user-editable configuration: input paths, output path, column mappings, blocking keys, fuzzy keys, similarity threshold, etc.

2. **`fuzzy_helpers.py`**  
   Contains the **helper functions** for loading data, mapping columns, combining DataFrames, blocking, fuzzy comparison, and merging duplicates.

3. **`fuzzy_deduplication.ipynb`**  
   Jupyter Notebook demonstrating how to:
   1. Load the YAML config  
   2. Load and map Dataset 1 & Dataset 2  
   3. Perform blocking and fuzzy matching  
   4. Merge duplicates  
   5. Save the final deduplicated data  

4. **`dataset1.csv`**  
   Example dummy dataset in the target schema (Dataset 1’s format).

5. **`dataset2.csv`**  
   Example dummy dataset in a different schema that must be mapped into Dataset 1’s format.

6. **`requirements.txt`**  
   Lists the Python packages needed to run this pipeline.

7. **`README.md`**  
   The file you are reading now.

---

## Getting Started

### 1. **Clone** or download this repository:
```bash
git clone https://github.com/toratommy/semantic-deduplication
cd semantic-deduplication
```

### 2. Set Up a Conda Environment (Recommended)

1. **Create** a new conda environment (e.g. “fuzzy-dedup”) with Python 3.12:
   ```bash
   conda create -n fuzzy-dedup python=3.12
   ```
2. **Activate** the environment:
   ```bash
   conda activate fuzzy-dedup
   ```
3. **Install** dependencies using `pip` within the environment:
   ```bash
   pip install -r requirements.txt
   ```

> *Note*: If you prefer not to use Conda, you can simply install Python 3.12+ and run `pip install -r requirements.txt` in a regular virtual environment.

### 3. Adjust `config.yml`

```yaml
dataset1_path: "data/dataset1.csv"
dataset2_path: "data/dataset2.csv"
output_path: "combined_deduplicated.csv"

dataset1_columns:
  - "..."

dataset2_to_dataset1:
  "...": "..."

blocking_keys:
  - "..."

fuzzy_keys:
  - "..."

similarity_threshold: 0.85
```

- **`dataset1_path`**: Path to Dataset 1 CSV (in `data/`)  
- **`dataset2_path`**: Path to Dataset 2 CSV (in `data/`)  
- **`output_path`**: Where the merged, deduplicated CSV will be written  
- **`dataset1_columns`**: Full list of columns in Dataset 1’s schema  
- **`dataset2_to_dataset1`**: Mapping from Dataset 2 fields to Dataset 1 columns  
- **`blocking_keys`**: Fields requiring an exact match before fuzzy comparisons  
- **`fuzzy_keys`**: Columns on which to perform string similarity  
- **`similarity_threshold`**: Average similarity above which two rows are considered duplicates  

### 4. Run the Pipeline in Jupyter Notebook

1. Launch Jupyter from your conda environment:
   ```bash
   jupyter notebook fuzzy_deduplication.ipynb
   ```
2. Open **`fuzzy_deduplication.ipynb`** in your browser.
3. Run the notebook cells from top to bottom.
4. The final deduplicated CSV will be saved at the `output_path` you specified in `config.yml`.

---

## Project Flow

1. **Load Config**: We parse `config.yml` to get file paths, column mappings, thresholds, etc.  
2. **Load Datasets**:
   - **Dataset 1** is already in the target schema.  
   - **Dataset 2** is mapped to that schema using `dataset2_to_dataset1`.  
3. **Combine**: Merge both data sources into one DataFrame.  
4. **Blocking**: With `recordlinkage.Index()`, only rows that share the same `blocking_keys` values become “candidate pairs.”  
5. **Fuzzy Matching**: For each candidate pair, we compute textual similarity on `fuzzy_keys` via Jaro-Winkler (default in recordlinkage’s `Compare.string()`).  
6. **Classification**: We average the similarity across the `fuzzy_keys` for each pair. If >= `similarity_threshold`, the pair is considered duplicates.  
7. **Merge**: We union-find all duplicates into groups, keeping one record (earliest) per group.  
8. **Save**: The final DataFrame is written to `output_path`.

---

## Customization

- **Column Mappings**: Adjust `dataset2_to_dataset1` in `config.yml` if your Dataset 2 column names differ.  
- **Blocking**: If you need more or fewer columns for blocking, update `blocking_keys`.  
- **Fuzzy Method**:  
  - By default, `Compare.string(..., method='jaro_winkler')` returns a score in `[0,1]`.  
  - You can switch to `'levenshtein'`, `'jaro'`, or pass a custom function.  
- **Merging Logic**: Currently we drop all but the earliest row in each duplicate group. Modify `merge_duplicate_pairs()` in **`src/fuzzy_helpers.py`** if you’d prefer to sum volumes or combine text fields.  
- **Semantic Dedup**: If your data requires concept-level matching, you can replace or augment fuzzy matching with embeddings or advanced text similarity.