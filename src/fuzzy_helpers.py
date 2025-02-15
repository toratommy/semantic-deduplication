# fuzzy_helpers.py

import pandas as pd
import numpy as np
import recordlinkage
from recordlinkage import Index, Compare
from recordlinkage.base import BaseCompareFeature
from rapidfuzz.distance import JaroWinkler
from typing import List, Dict, Any
import yaml
from collections import defaultdict

##############################################################################
# Custom Comparator using Rapidfuzz
##############################################################################

class RapidfuzzComparator(BaseCompareFeature):
    """
    A custom comparator for recordlinkage that uses Rapidfuzz's JaroWinkler
    similarity in [0,1].
    """

    def __init__(self, left_on, right_on, label=None):
        """
        For older recordlinkage versions, we must call super().__init__ with:
            super().__init__(left_on, right_on, label)
        in that order.
        """
        super().__init__(left_on, right_on, label)

    def _compute_vectorized(self, s_left: pd.Series, s_right: pd.Series) -> np.ndarray:
        """
        s_left, s_right are Series from the left/right columns. We compute 
        JaroWinkler.similarity(...) for each pair, returning a NumPy array
        in [0..1].
        """
        scores = [
            JaroWinkler.similarity(str(x), str(y))
            for x, y in zip(s_left, s_right)
        ]
        return np.array(scores)


##############################################################################
# Load / Mapping Functions
##############################################################################

def load_config(config_file: str) -> dict:
    """
    Load YAML configuration from a file path and return as a dictionary.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_dataset1(path: str, dataset1_columns: List[str]) -> pd.DataFrame:
    """
    Reads Dataset1 CSV, ensures it has all columns in dataset1_columns,
    and reorders them. Missing columns are filled with NaN.
    """
    df = pd.read_csv(path)
    
    # Ensure all required columns exist
    for col in dataset1_columns:
        if col not in df.columns:
            df[col] = np.nan
    
    # Reorder columns
    df = df[dataset1_columns]
    return df


def load_dataset2_and_map(path: str, dataset1_columns: List[str], mapping: Dict[str, str]) -> pd.DataFrame:
    """
    Loads Dataset2 from CSV and maps columns to Dataset1 format using a dict.
    Unmapped columns are dropped. Any columns in dataset1_columns not found
    in the mapping will be filled with NaN.
    """
    df2 = pd.read_csv(path)
    
    # Create a new DataFrame with Dataset1 columns
    mapped_df = pd.DataFrame(columns=dataset1_columns)
    
    # Map columns from Dataset2 -> Dataset1
    for col2, col1 in mapping.items():
        if col2 in df2.columns and col1 in mapped_df.columns:
            mapped_df[col1] = df2[col2]
    
    # Fill any remaining columns with NaN
    for col1 in dataset1_columns:
        if col1 not in mapped_df.columns:
            mapped_df[col1] = np.nan
    
    # Reorder columns
    mapped_df = mapped_df[dataset1_columns]
    return mapped_df


def combine_dataframes(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Combine both dataframes (in Dataset1 format) into a single dataframe.
    """
    combined = pd.concat([df1, df2], ignore_index=True).reset_index(drop=True)
    return combined


##############################################################################
# Blocking & Fuzzy Comparison
##############################################################################

def block_data(df: pd.DataFrame, blocking_keys: List[str]) -> pd.MultiIndex:
    """
    Use recordlinkage to identify candidate pairs by blocking on the given columns.
    We do a "self-join" for deduplication within a single dataframe.
    
    Returns:
        A MultiIndex of candidate (row_i, row_j) pairs, excluding self-matches (i, i).
    """
    indexer = Index()
    for key in blocking_keys:
        indexer.block(left_on=key, right_on=key)
    
    candidate_pairs = indexer.index(df, df)
    
    # Remove self-matches (where row i == row j)
    candidate_pairs = candidate_pairs[
        candidate_pairs.get_level_values(0) != candidate_pairs.get_level_values(1)
    ]
    
    return candidate_pairs


def build_comparison_features(
    df: pd.DataFrame,
    candidate_pairs: pd.MultiIndex,
    fuzzy_keys: List[str]
) -> pd.DataFrame:
    """
    For each pair in candidate_pairs, compute fuzzy similarity on the specified columns
    using RapidfuzzComparator, which calls rapidfuzz JaroWinkler internally.
    
    Returns:
        A DataFrame (same index as candidate_pairs) of similarity scores.
    """
    compare = Compare()
    # For older recordlinkage versions, pass the column names positionally:
    # e.g., compare.add(RapidfuzzComparator("Opportunity Name", "Opportunity Name", label="Opportunity Name"))
    for col in fuzzy_keys:
        compare.add(RapidfuzzComparator(col, col, label=col))
    
    features = compare.compute(candidate_pairs, df, df)
    return features


##############################################################################
# Classification & Merging
##############################################################################

def classify_duplicates(features_df: pd.DataFrame, threshold: float) -> pd.MultiIndex:
    """
    Decide which pairs are duplicates by averaging fuzzy similarity across columns.
    If avg similarity >= threshold, classify as duplicates.
    
    Returns:
        A MultiIndex of (i, j) rows considered duplicates.
    """
    features_df["mean_similarity"] = features_df.mean(axis=1)
    duplicates_idx = features_df[features_df["mean_similarity"] >= threshold].index
    return duplicates_idx


def merge_duplicate_pairs(df: pd.DataFrame, duplicates_idx: pd.MultiIndex) -> pd.DataFrame:
    """
    Given a MultiIndex of (i, j) duplicates, group them (union-find style)
    and keep the earliest index in each group, dropping the rest.
    
    Returns:
        A new DataFrame with duplicates dropped.
    """
    adjacency = defaultdict(set)
    
    for (i, j) in duplicates_idx:
        adjacency[i].add(j)
        adjacency[j].add(i)
    
    visited = set()
    groups = []
    
    def dfs(node, group):
        stack = [node]
        while stack:
            n = stack.pop()
            for neigh in adjacency[n]:
                if neigh not in visited:
                    visited.add(neigh)
                    group.add(neigh)
                    stack.append(neigh)
    
    # Build connected components of duplicates
    for i in adjacency.keys():
        if i not in visited:
            visited.add(i)
            g = {i}
            dfs(i, g)
            groups.append(g)
    
    # For each group, keep the earliest row, drop the rest
    to_drop = []
    for g in groups:
        sorted_list = sorted(g)
        keep = sorted_list[0]
        drop = sorted_list[1:]
        to_drop.extend(drop)
    
    df_merged = df.drop(labels=to_drop, axis=0).reset_index(drop=True)
    return df_merged
