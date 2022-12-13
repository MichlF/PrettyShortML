# Imports
import pandas as pd
import functools
from time import perf_counter
from dataclasses import dataclass

@dataclass
class _BaseClass:
    """
    Baseclass contains class data shared among all other classes as well as decorator-types.
    """

    # For future versions, we might want to consider setting these to private variables and work with
    # @property or at least getters and setters to prevent freak overwritting glitches of these vars.
    df_data: pd.DataFrame = None
    X_train: pd.DataFrame = None
    X_test: pd.DataFrame = None
    y_train: pd.DataFrame = None
    y_test: pd.DataFrame = None
    modelled_data: pd.DataFrame = None
    unmodelled_data: pd.DataFrame = None

    def __post_init__(self):
        """
        Does data type checks.
        """
        # Data type checks
        if self.df_data is not None:
            assert isinstance(
                self.df_data, pd.DataFrame
            ), "df_data must be a pandas DataFrame !"
        if self.X_train is not None:
            assert isinstance(
                self.X_train, pd.DataFrame
            ), "X_train must be a pandas DataFrame !"
        if self.y_train is not None:
            assert isinstance(
                self.y_train, (pd.DataFrame, pd.Series)
            ), "y_train must be a pandas DataFrame or Series !"
        if self.X_test is not None:
            assert isinstance(
                self.X_test, pd.DataFrame
            ), "X_test must be a pandas DataFrame!"
        if self.y_test is not None:
            assert isinstance(
                self.y_test, (pd.DataFrame, pd.Series)
            ), "y_test must be a pandas DataFrame or Series !"
        if self.modelled_data is not None:
            assert isinstance(
                self.modelled_data, pd.DataFrame
            ), "modelled_data must be a pandas DataFrame !"
        if self.unmodelled_data is not None:
            assert isinstance(
                self.unmodelled_data, pd.DataFrame
            ), "unmodelled_data must be a pandas DataFrame !"

    def _timer(function, decimal_places=3, print_result=True):
        """
        Decorater/wrapper function to measure elapsed time of input function

        Parameters
        ----------
        function : Function
            To be wrapped function.
        """

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            before = perf_counter()
            value = function(*args, **kwargs)
            after = perf_counter()
            fname = function.__name__
            time_diff = after - before
            formatted_time = f"{time_diff:.{decimal_places}f}"
            if print_result:
                print(f"{fname} took {formatted_time} secs to run.\n")

            return value

        return wrapper