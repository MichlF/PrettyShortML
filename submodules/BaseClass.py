# Imports
import functools
import logging
import logging.config as logging_conf
import os
import typing
from dataclasses import dataclass
from time import perf_counter

import pandas as pd
import yaml


@dataclass(slots=True)
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

    def init_logger(
        self,
        default_path: str = "logging.yaml",
        default_level: typing.Any = logging.INFO,
        env_key: str = "LOG_CFG",
    ):
        """
        Initializes a logger with settings from separate YAML file.

        Parameters
        ----------
        default_path : str, optional
            Path to YAML file containing the logger settings, by default "logging.yaml"
        default_level : typing.Any, optional
            Python logging object determining the logging level, by default logging.INFO
        env_key : str, optional
            _description_, by default "LOG_CFG"
        """
        path = default_path
        value = os.getenv(env_key, None)
        if value:
            path = value
        if os.path.exists(path):
            with open(path, "rt") as f:
                config = yaml.safe_load(f.read())
            logging_conf.dictConfig(config)
        else:
            logging.basicConfig(level=default_level)
        logging.info("")
        logging.info("Starting a new logging session...")
        logging.info("")

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
