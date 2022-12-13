# Imports
from submodules.BaseClass import _BaseClass
import seaborn as sns
import pandas as pd
from pprint import pprint
import matplotlib.pyplot as plt

class Eda(_BaseClass):
    """
    Collection of exploratory data analysis methods. Inherits from _BaseClass.
    """

    def eda_clean_check(
        data: pd.DataFrame,
        fig_size: tuple[int, int] = (10, 6),
    ):
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                print("Couldn't transform data file to DataFrame.")
                raise ValueError()
        # Check Dtypes
        print("Inspect dtypes:")
        pprint(data.info())
        print("\n")
        # Check descriptive statistics
        print("Inspect descriptive statistics:")
        pprint(data.describe())
        print("\n")
        # Check for NaNs visually and written
        _fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(
            data.isna(), ax=ax, xticklabels=True
        )  # using the ticklabels command we can plot ALL labels. Omit for y labels because it may take forever to plot when there is alot of rows
        ax.set_title("Overview over all NaNs in dataset")
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
        # Check balance
        for feature in data.columns:
            print(
                f"Value counts of {feature}:"
            )  #! improve output readability (each value_count is a pd.Series of varying length)
            pprint(data[feature].value_counts())
            print("\n")
        print("Amount of NaNs:")
        pprint(data.isna().sum())
        print("\n")
        # Check histograms
        data.hist(figsize=fig_size)
        plt.show()