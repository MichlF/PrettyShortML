# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

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
        Perform exploratory data analysis to check the quality and structure
        of the data.

        Parameters:
        data (pd.DataFrame): The input data to be analyzed.
        head (int, optional): The number of rows to show when checking feature
        counts. Defaults to 10.
        fig_size (tuple of ints, optional): The size of the histogram figures.
        Defaults to (10, 8).

        Returns:
        None
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
        Display data types and descriptive statistics for the input DataFrame.

        Parameters:
        data (pd.DataFrame): The input DataFrame to be analyzed.

        Returns:
        None
        """

        # Check Dtypes
        info = data.info()
        print("\nInspect dtypes:")
        print(info)
        print("\n")
        # Check descriptive statistics
        describe = data.describe(include="all")
        print("Inspect descriptive statistics:")
        print(describe)
        print("\n")

    @staticmethod
    def eda_check_nans(
        data: pd.DataFrame,
        fig_size: tuple[int, int] = (10, 6),
        xticklabels: bool = True,
    ) -> plt.Axes:
        """
        Display the number of NaN values in the input DataFrame and a heatmap
        visualizing the NaN values.

        Parameters:
        data (pd.DataFrame): The input DataFrame to be analyzed.
        fig_size (tuple, optional): The size of the heatmap plot. Default is
        (10, 6).
        xticklabels (bool, optional): Whether to display x-axis labels in the
        heatmap plot. Default is True.

        Returns:
        None
        """

        missing_values = pd.DataFrame(
            [
                data.isna().sum(),
                (data.isna().sum() * 100 / len(data)).round(2),
            ],
            index=["Number", "Percentage"],
        ).T
        print("Missing values per feature:")
        print(missing_values)
        print("\n")
        # Check for NaNs visually and written
        _fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(data.isna(), ax=ax, vmin=0, vmax=1)
        ax.set_title("Overview over all NaNs in dataset")
        ax.set_xlabel("Feature labels")
        ax.set_ylabel("Data rows")
        if xticklabels:
            ax.set_xticklabels(
                ax.get_xticklabels(),
                rotation=45,
                ha="right",
                rotation_mode="anchor",
            )

        return ax

    @staticmethod
    def eda_check_feature_counts(
        data: pd.DataFrame,
        head: int = 10,
    ) -> None:
        """
        Displays the top `head` value counts for each feature in the input
        dataframe.

        Parameters:
        data (pd.DataFrame): Input dataframe.
        head (int): Number of top value counts to display for each feature
        (default is 10).

        Returns:
        None
        """

        for feature in data.columns:
            feature_counts = data[feature].value_counts()
            print("\n")
            print(f"Top {head} value counts of {feature=}:")
            display(feature_counts.head(head))
            print("\n")

    @staticmethod
    def eda_count_nonpositive_values(data: pd.DataFrame) -> pd.DataFrame:
        """Count the number of non-positive values in each column of a
        dataframe"""
        nonpositive_counts = {}

        for column in data.select_dtypes(include=[np.number]).columns:
            nonpositive_counts[column] = (data[column] <= 0).sum()

        return pd.DataFrame(
            {
                "Non-positive value count of numerical columns": nonpositive_counts
            }
        )
