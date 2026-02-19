
import pandas as pd
import numpy as np
from typing import List, Optional

# Global configuration
DEFAULT_SCALING_FACTOR = 1.5

class DataPipeline:
    def __init__(self, scaling_factor: float = DEFAULT_SCALING_FACTOR):
        self.scaling_factor = scaling_factor

    def normalize(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Normalize specific columns in the dataframe.
        """
        df_copy = df.copy()
        for col in columns:
            mean = df_copy[col].mean()
            std = df_copy[col].std()
            df_copy[col] = (df_copy[col] - mean) / std
        return df_copy

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract features as a numpy array.
        """
        return df.select_dtypes(include=[np.number]).to_numpy()

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove null values from the dataset.
    """
    return data.dropna()
