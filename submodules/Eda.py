# Imports
from IPython.display import display
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from submodules.BaseClass import _BaseClass


class Eda(_BaseClass):
    """
    Collection of exploratory data analysis methods. Inherits from _BaseClass.
    """

    @staticmethod
    def eda_clean_check(
        data: pd.DataFrame,
        head: int = 10,
        fig_size: tuple[int, int] = (10, 8),
    ) -> None:
        """
        Some docstring
        """
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as _e:
                print("Couldn't transform data file to DataFrame.")
                raise ValueError() from _e
        # Info and describe
        Eda.eda_check_descriptives(data=data)
        # Check feature counts
        Eda.eda_check_feature_counts(data=data, head=head)
        # Check NaNs
        Eda.eda_check_nans(data=data, fig_size=fig_size)
        # Check histograms
        data.hist(figsize=fig_size)
        plt.show()

    @staticmethod
    def eda_check_descriptives(
        data: pd.DataFrame,
    ) -> None:
        """
        Checks df descriptives
        """
        # Check Dtypes
        print("\nInspect dtypes:")
        display(data.info())
        print("\n")
        # Check descriptive statistics
        print("Inspect descriptive statistics:")
        display(data.describe())
        print("\n")

    @staticmethod
    def eda_check_nans(
        data: pd.DataFrame, fig_size: tuple[int, int] = (10, 6), xticklabels: bool = True
    ) -> None:
        """
        Checks for NaNs
        With ticklabels we can plot ALL labels. Omit for y labels b/c it may take forever to plot when there is alot of rows
        """

        print("Amount of NaNs per feature:")
        display(data.isna().sum())
        print("\n")
        # Check for NaNs visually and written
        _fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(data.isna(), ax=ax, xticklabels=xticklabels)
        ax.set_title("Overview over all NaNs in dataset")
        ax.set_xlabel("Feature labels")
        ax.set_ylabel("Data rows")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    @staticmethod
    def eda_check_feature_counts(
        data: pd.DataFrame,
        head: int = 10,
    ) -> None:
        """
        Checks feature counts
        """
        for feature in data.columns:
            print(f"Top {head} value counts of {feature=}:")
            display(data[feature].value_counts().head(head))
            print("\n")
