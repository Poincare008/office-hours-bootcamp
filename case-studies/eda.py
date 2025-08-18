import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
    HAS_SEABORN = True
except Exception:
    HAS_SEABORN = False


class EDA:
    """
    Lightweight EDA helper for tabular datasets.
    Provides quick data overview and commonly used visualizations.
    Designed to be used from notebooks or scripts.
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None, title_prefix: str = ""):
        self.df = df
        self.target_col = target_col
        self.title_prefix = (title_prefix + " - ") if title_prefix else ""

    # --------- Textual overview ---------
    def overview(self, head: int = 5) -> dict:
        info = {
            "shape": self.df.shape,
            "dtypes": self.df.dtypes.astype(str).to_dict(),
            "missing_counts": self.df.isna().sum().to_dict(),
            "describe_numeric": self.df.select_dtypes(include=[np.number]).describe().to_dict(),
        }
        sample = self.df.head(head)
        print(f"{self.title_prefix}Data shape: {info['shape']}")
        print("\nDtypes:")
        print(pd.Series(info["dtypes"]))
        print("\nMissing values per column:")
        print(pd.Series(info["missing_counts"]))
        print("\nNumeric summary (describe):")
        print(self.df.select_dtypes(include=[np.number]).describe())
        print("\nHead:")
        print(sample)
        return info

    # --------- Visualizations ---------
    def plot_missingness(self):
        miss = self.df.isna().mean().sort_values(ascending=False)
        plt.figure(figsize=(max(6, len(miss) * 0.6), 4))
        if HAS_SEABORN:
            sns.barplot(x=miss.index, y=miss.values, color="#4C72B0")
        else:
            plt.bar(miss.index, miss.values, color="#4C72B0")
        plt.ylabel("Fraction Missing")
        plt.title(f"{self.title_prefix}Missingness by Column")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    def plot_distributions(self, cols: Optional[Sequence[str]] = None, bins: int = 30):
        num_df = self.df.select_dtypes(include=[np.number])
        if cols is None:
            cols = list(num_df.columns)
        if not cols:
            print("No numeric columns to plot.")
            return
        n = len(cols)
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        plt.figure(figsize=(ncols * 4, nrows * 3.2))
        for i, c in enumerate(cols, 1):
            plt.subplot(nrows, ncols, i)
            if HAS_SEABORN:
                sns.histplot(num_df[c].dropna(), kde=True, bins=bins, color="#4C72B0")
            else:
                plt.hist(num_df[c].dropna(), bins=bins, color="#4C72B0", alpha=0.85)
            plt.title(c)
            plt.grid(True, alpha=0.2)
        plt.suptitle(f"{self.title_prefix}Numeric Distributions", y=1.02)
        plt.tight_layout()
        plt.show()

    def plot_correlation(self, method: str = "pearson"):
        num_df = self.df.select_dtypes(include=[np.number])
        if num_df.shape[1] < 2:
            print("Not enough numeric columns to compute correlation.")
            return
        corr = num_df.corr(method=method)
        plt.figure(figsize=(max(6, corr.shape[1]), max(5, corr.shape[0])))
        if HAS_SEABORN:
            sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
        else:
            plt.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            plt.colorbar()
            plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
            plt.yticks(range(len(corr.index)), corr.index)
        plt.title(f"{self.title_prefix}Correlation Heatmap ({method})")
        plt.tight_layout()
        plt.show()

    def plot_target_balance(self):
        if self.target_col is None or self.target_col not in self.df.columns:
            print("Target column not set or not found; skipping target balance plot.")
            return
        vc = self.df[self.target_col].value_counts(dropna=False).sort_index()
        plt.figure(figsize=(5, 3.5))
        if HAS_SEABORN:
            sns.barplot(x=vc.index.astype(str), y=vc.values, palette="pastel")
        else:
            plt.bar(vc.index.astype(str), vc.values, color="#55A868")
        plt.title(f"{self.title_prefix}Target Balance: {self.target_col}")
        plt.ylabel("Count")
        plt.xlabel(self.target_col)
        plt.tight_layout()
        plt.show()

    # --------- Convenience runner ---------
    def quicklook(self, distributions: bool = True, correlation: bool = True, missingness: bool = True, target_balance: bool = True):
        self.overview()
        if missingness:
            self.plot_missingness()
        if distributions:
            self.plot_distributions()
        if correlation:
            self.plot_correlation()
        if target_balance:
            self.plot_target_balance()
