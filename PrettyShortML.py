# Requires Python 3.9+
import requests
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from textwrap import wrap
from time import time
from dataclasses import dataclass
from pathlib import Path
from langdetect import detect_langs
from imblearn.under_sampling import (
    RandomUnderSampler,
    NearMiss,
)  # python 3.9+ install with: conda install -c conda-forge imbalanced-learn
from imblearn.over_sampling import (
    RandomOverSampler,
    SMOTE,
)  # python 3.9+ install with: conda install -c conda-forge imbalanced-learn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_curve,
)


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
        Post_init method that does data type checks.
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
            assert (isinstance(self.y_train, pd.DataFrame)) or (
                isinstance(self.y_train, pd.Series)
            ), "y_train must be a pandas DataFrame or Series !"
        if self.X_test is not None:
            assert isinstance(
                self.X_test, pd.DataFrame
            ), "X_test must be a pandas DataFrame!"
        if self.y_test is not None:
            assert (isinstance(self.y_test, pd.DataFrame)) or (
                isinstance(self.y_test, pd.Series)
            ), "y_test must be a pandas DataFrame or Series !"
        if self.modelled_data is not None:
            assert isinstance(
                self.modelled_data, pd.DataFrame
            ), "modelled_data must be a pandas DataFrame !"
        if self.unmodelled_data is not None:
            assert isinstance(
                self.unmodelled_data, pd.DataFrame
            ), "unmodelled_data must be a pandas DataFrame !"

    def _timer(function):
        """
        Decorater/wrapper function to measure elapsed time of input function

        Parameters
        ----------
        function : Function
            To be wrapped function.
        """

        def wrapper(*args, **kwargs):
            before = time()
            value = function(*args, **kwargs)
            after = time()
            fname = function.__name__
            print(f"{fname} took {after - before} secs to run.\n")
            return value

        return wrapper


@dataclass
class PrettyShortModelling(_BaseClass):
    """
    Class containing blue print methods for quickly run standard model training and evaluation pipelines.
    Inherits from _BaseClass.
    """

    @staticmethod
    def data_undersampler(
        max_sample_size: int,
        df: pd.DataFrame,
        y_label: str = None,
        method: str = "pandas",
        method_imblearn: str = "rus",
        *args,
        **kwargs,
    ):
        """
        Undersamples features of a given dataset (a pandas Dataframe) to a given maximum of observations
        either using pd.sample or Undersample from the imblearn module.

        Parameters
        ----------
        max_sample_size : int
            Maximum sample size for each target class. In case a given class is < max_sample_size, the data
            for that class is unchanged.
        df : pd.DataFrame
            Data (df) to undersample.
        y_label : str, optional
            Label of the target used as column name in data, defaults to None but needs to be define for
            when method == "pandas".
        method : str, optional
            Method for undersampling. "pandas" uses the pd.DataFrame.sample method. "imblearn" uses imblearn
            module to achieve undersampling, by default "pandas".
        method_imblearn : str, optional
            Method for imblearn undersampling. "rus" corresponds to Random undersampling which selects a
            subsample from the majority class randomly. "nm" refers to Near Miss, which selects the
            subsample based on the distances between the majority class datapoints and the minority
            class datapoints. In NearMiss3 the nearest n neighbors to each minority sample is selected,
            and then the furthest point among these is eliminated. In versions 1 and 2, the average
            and minimum distances between points are used as selection criteria rather than the largest.
        *args / *kwargs
            Will be passed to the pd.DataFrame.sample(), RandomUnderSampler() or NearMiss() function
            depending on the chosen method parameters.

        Returns
        -------
        if method == "pandas":
        tuple[pd.DataFrame, pd.DataFrame]
            Returns to pd.DataFrames. df_undersamp is the undersampled dataframe ready for modelling.
            df_undersamp_reject contains all rejected feature values.
        if method == "imblearn":
            Returns imblearn, either a random undersampling or near miss, object which needs to be used
            by calling its fit_resample method to obtain a resampled X_train, y_train dataset
        """
        print(f"Restricting max n for each feature to {max_sample_size}...")
        if method == "pandas":
            df_select, df_reject = (
                [],
                [],
            )  # df with max_sample_size n for each feature class and df with all rejected n for each feature class (we can use the latter for testing our model independently)
            for x_class in df[y_label].unique():
                if df.loc[df[y_label] == x_class].shape[0] > max_sample_size:
                    select = df.loc[df[y_label] == x_class].sample(
                        n=max_sample_size, *args, **kwargs
                    )
                    df_select.append(select)
                    reject = (
                        pd.merge(
                            df.loc[df[y_label] == x_class],
                            select,
                            indicator=True,
                            how="outer",
                        )
                        .query('_merge=="left_only"')
                        .drop("_merge", axis=1)
                    )
                    df_reject.append(reject)
                else:
                    df_select.append(df.loc[df[y_label] == x_class])
            # Rebuilt dfs for modelling (selected) and testing (rejected) - TBC: we could also rebuilt the original lyrics_data from these
            df_undersamp = pd.concat(df_select).dropna().reset_index(drop=True)
            df_undersamp_reject = pd.concat(df_reject).dropna().reset_index(drop=True)

            return df_undersamp, df_undersamp_reject

        elif method == "imblearn":
            if method_imblearn == "rus":
                imblearn_obj = RandomUnderSampler(
                    sampling_strategy={0: max_sample_size}, *args, **kwargs
                )
            elif method_imblearn == "nm":
                imblearn_obj = NearMiss(
                    sampling_strategy={0: max_sample_size}, *args, **kwargs
                )

            return imblearn_obj

    @staticmethod
    def data_oversampler(
        min_sample_size: int,
        df: pd.DataFrame,
        y_label: str,
        method: str = "pandas",
        method_imblearn: str = "ros",
        *args,
        **kwargs,
    ):
        """
        Undersamples features of a given dataset (a pandas Dataframe) to a given maximum of observations
        either using pd.sample or Undersample from the imblearn module.

        Parameters
        ----------
        min_sample_size : int
            Minimum sample size for each target class. In case a given class is > min_sample_size, the data
            for that class is unchanged.
        df : pd.DataFrame
            Data (df) to oversample.
        y_label : str, optional
            Label of the target used as column name in data, defaults to None but needs to be define for
            when method == "pandas".
        method : str, optional
            Method for oversampling. "pandas" uses the pd.DataFrame.sample method. "imblearn" uses imblearn
            module to achieve oversampling, by default "pandas".
        method_imblearn : str, optional
            Method for imblearn oversampling. "ros" corresponds to Random undersampling which selects a
            subsample from the majority class randomly. "smote" refers to SMOTE.
        *args / *kwargs
            Will be passed to the pd.DataFrame.sample(), RandomOverSampler() or SMOTE() function
            depending on the chosen method parameters.

        Returns
        -------
        if method == "pandas":
        tuple[pd.DataFrame, pd.DataFrame]
            Returns to pd.DataFrames. df_undersamp is the undersampled dataframe ready for modelling.
            df_undersamp_reject contains all rejected feature values.
        if method == "imblearn":
            Returns imblearn, either a random oversampling or SMOTE, object which needs to be used
            by calling its fit_resample method to obtain a resampled X_train, y_train dataset
        """
        if method == "pandas":
            raise NotimplementedError("Not yet here. Come back later, please.")

        elif method == "imblearn":
            if method_imblearn == "ros":
                imblearn_obj = RandomOverSampler(
                    sampling_strategy={1: min_sample_size}, *args, **kwargs
                )
            elif method_imblearn == "smote":
                imblearn_obj = SMOTE(
                    sampling_strategy={1: min_sample_size}, *args, **kwargs
                )
            return imblearn_obj

    @_BaseClass._timer
    def model_train(
        self,
        estimator_object,
        param_grid: dict = None,
        numeric_features: list[str] = None,
        categorical_features: list[str] = None,
        ordinal_features: list[str] = None,
        ordinal_categories: list[list[str]] = None,
        add_transformers: list = None,
        add_pipesteps: list = None,
        *args,
        **kwargs,
    ) -> tuple[Pipeline, float]:
        """
        Builds a generic ML pipeline and fits any given estimator. Also does gridsearch, if a
        parameter grid is provided. Standard procedure for numerical features is simple imputing
        (strategy="mean") and standard scaling. For categorical features, a one-hot encoder
        (handle_unknown="ignore", drop="first") is used. For ordinal features, an ordinal encoder
        (handle_unknown="ignore") is used.

        Parameters
        ----------
        estimator_object : Scikit-learn estimator object
            Scikit-learn estimator object that should be fitted.
        param_grid : dict, optional
            Parameter grid for GridSearchCV. If not provided, no grid search is performed, default None.
        numeric_features : list of str, optional
            Numeric features to be added to the preprocessor, default None.
        categorical_features : list of str, optional
            Categorical features to be added to the preprocessor, default None.
        ordinal_features : list of str, optional
            Ordinal features to be added to the preprocessor, default None.
        ordinal_categories : list of list of str, optional
            When ordinal features are provided, the order of categories has to be provided for the
            OrdinalEncoder. Each feature should be encoded as list of the unique feature values in
            ascending order, default None.
        add_transformers : tuple or list of tuples, optional
            Additional transformers for the preprocessor. Formatted as tuple (label str, sklearn
            transformer object, list of features to which transformation should be applied to),
            defaults None.
        add_pipesteps : list, optional
            Additional steps for the main pipeline. Formatted as tuple (label str, sklearn
            object), defaults None.
        *args / *kwargs
            Will be passed to the sklearn.GridsearchCV() function.

        Returns
        -------
        Pipeline : Scitkit-learn Pipeline object
            Pipeline object with fitted estimator as class attribute.
        accuracy : float
            Classification accuracy obtained with sklearn score method.
        """
        print(
            f"Building and fitting {estimator_object.__class__.__name__} estimator..."
        )
        # Build transformer for preprocessor
        transformer = []
        if numeric_features:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler()),
                ]
            )
            transformer.append(("numeric", numeric_transformer, numeric_features))
        if categorical_features:
            categorical_transformer = OneHotEncoder(
                handle_unknown="ignore", drop="first"
            )
            transformer.append(
                ("categoric", categorical_transformer, categorical_features)
            )
        if ordinal_features:
            assert (
                ordinal_categories
            ), "When using ordinal_features, ordinal_categories has to be provided !"
            ordinal_transformer = OrdinalEncoder(
                categories=ordinal_categories, handle_unknown="ignore"
            )
            transformer.append(("ordinal", ordinal_transformer, ordinal_features))
        if add_transformers:
            transformer.append(add_transformers)
        self.preprocessor = ColumnTransformer(transformers=transformer)
        # Build pipeline
        pipeline_steps = [
            ("preprocessor", self.preprocessor),
            ("estimator", estimator_object),
        ]
        if add_pipesteps:
            pipeline_steps.append(add_pipesteps)
        self.pipe = Pipeline(steps=pipeline_steps)
        # Optional: perform gridsearch
        if param_grid:
            self.pipe = GridSearchCV(
                estimator=self.pipe,
                param_grid=param_grid,
                *args,
                **kwargs,
            )
        # Fit pipeline
        self.pipe.fit(self.X_train, self.y_train)
        # Get basic metrics
        train_accuracy = self.pipe.score(self.X_train, self.y_train)

        return self.pipe, train_accuracy

    @_BaseClass._timer
    def model_NLP_clf_train(
        self, estimator_object, param_grid: dict = None, *args, **kwargs
    ) -> tuple[Pipeline, float]:
        """
        Builds a pipeline with column vectorizer and tfidf transformer that fits a given estimator.
        Also does gridsearch, if a parameter grid is provided.

        Parameters
        ----------
        estimator_object : Scikit-learn estimator object
            Scikit-learn estimator object that should be fitted.
        param_grid : dict, optional
            Parameter grid for GridSearchCV. If not provided, no grid search is performed, default None.
        *args / *kwargs
            Will be passed to the sklearn.GridsearchCV() function.

        Returns
        -------
        Pipeline : Scitkit-learn Pipeline object
            Pipeline object with fitted estimator as class attribute.
        accuracy : float
            Classification accuracy obtained with sklearn score method.
        """
        print(
            f"Building and fitting {estimator_object.__class__.__name__} estimator..."
        )
        self.pipe = Pipeline(
            [
                (
                    "vector",
                    CountVectorizer(lowercase=True, ngram_range=(1, 2)),
                ),  # Note: ngram_range massively increases runtime
                ("tfidf", TfidfTransformer()),
                ("estimator", estimator_object),
            ]
        )  # Note: Combine vector and tfidf steps by using TfidfVectorizer()
        if param_grid:
            self.pipe = GridSearchCV(
                estimator=self.pipe,
                param_grid=param_grid,
                *args,
                **kwargs,
            )
        self.pipe.fit(self.X_train, self.y_train)
        train_accuracy = self.pipe.score(self.X_train, self.y_train)

        return self.pipe, train_accuracy

    def model_clf_evaluate(
        self,
        model: Pipeline = None,
        average: str = "micro",
        plot_confusion: bool = False,
        normalize_conmat: bool = True,
        *args,
        **kwargs,
    ) -> tuple[float, float, float, float]:
        """
        Evaluates a given model (the fitted Pipeline object) and returns basic metrics (accuracy,
        precision, recall, and f1-score). If requested plots multiclass confusion matrices.

        Parameters
        ----------
        model : Scitkit-learn Pipeline object, optional
            Pipeline object with fitted estimator. If not provided, it will be attempted to use the class
            instance-specific object which only exists if the instance has run model_train before,
            by default None.
        average : str, optional
            Averaging method for calculating sklearn's precision, recall and f1 score, by default "micro".
        plot_confusion : bool, optional
            Whether or not a confusion matrix should be plotted, by default False. Works for
            multiclass problems, too.
        normalize_conmat : bool, optional
            Whether or not the values of the confusion matrix are normalized, by default True.
            Is ignored, if plot_confusion is False.
        *args / *kwargs
            Will be passed to the seaborn.heatmap() function.

        Returns
        -------
        accuracy : float
            Accuracy score on test data predictions.
        precision : float
            Precision score on test data predictions.
        recall : float
            Recall score on test data predictions.
        f1score : float
            F1-score on test data predictions.
        """
        if not model:
            try:
                model = self.pipe
                print("No model provided, used instance-specific model instead...")
            except Exception as e:
                raise Exception(
                    e,
                    "No model found. Did you forgot to provide a model or did you not run a model training function ?",
                )
        try:
            model_name = model.estimator["estimator"].__class__.__name__
        except:  # if no gridsearch
            model_name = model["estimator"].__class__.__name__
        y_pred = model.predict(self.X_test)
        print(
            "\n",
            model_name,
            "Model\n",
            "Predicted labels\n",
            pd.Series(y_pred).value_counts(),
            "\n\nActual labels\n",
            pd.Series(self.y_test).value_counts(),
        )
        accuracy = accuracy_score(
            self.y_test,
            y_pred,
        )
        precision = precision_score(self.y_test, y_pred, average=average)
        recall = recall_score(self.y_test, y_pred, average=average)
        f1score = f1_score(self.y_test, y_pred, average=average)
        if plot_confusion:
            palette = sns.color_palette("YlOrBr", as_cmap=True)
            con_mat = confusion_matrix(self.y_test, y_pred)
            if normalize_conmat:
                con_mat = con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis]
            _, ax = plt.subplots()
            sns.heatmap(
                con_mat,
                ax=ax,
                annot=True,
                fmt=".4f",
                linewidths=1,
                linecolor="white",
                cmap=palette,
                cbar=False,
                *args,
                **kwargs,
            )
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
            )
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            ax.set_title(f"Confusion matrix for {model_name} model")
            plt.show()

        return accuracy, precision, recall, f1score


@dataclass
class PrettyShortPlotting(_BaseClass):
    """
    Collection of blue print methods to plot metrics of ML pipelines. Inherits from _BaseClass.
    """

    @classmethod
    def plot_eda_pairplot(
        cls,
        data: pd.DataFrame = None,
        corner: bool = True,
        dropna: bool = True,
        plot_hist: bool = True,
        fig_size: tuple[int] = (8, 6),
        *args,
        **kwargs,
    ) -> plt.Axes:
        if data is None:
            try:
                data = cls.df_data
                print("No data provided, used instance-specific df_data instead...")
            except Exception as e:
                raise Exception(
                    e,
                    "No data found. Did you forgot to provide data ?",
                )
        if plot_hist:
            data.hist(figsize=fig_size)
        _fig, ax = plt.subplots(fig_size=fig_size)
        sns.pairplot(
            data=data,
            kind="reg",
            diag_kind="kde",
            corner=corner,
            dropna=dropna,
            plot_kws={"scatter_kws": {"alpha": 0.2}},
            ax=ax * args,
            **kwargs,
        )
        ax.fig.suptitle("Pairwise relationships")
        sns.despine()

        return ax

    @classmethod
    def plot_eda_corr_mat(
        cls,
        data: pd.DataFrame = None,
        metric: str = "spearman",
        cmap="vlag",
        mask: bool = True,
        annot: bool = True,
        linewidths=0.5,
        fig_size: tuple[int] = (12, 8),
        wrap_length: int = 60,
        *args,
        **kwargs,
    ) -> plt.Axes:
        if data is None:
            try:
                data = cls.df_data
                print("No data provided, used instance-specific df_data instead...")
            except Exception as e:
                raise Exception(
                    e,
                    "No data found. Did you forgot to provide data ?",
                )
        if mask:
            mask = np.triu(np.ones_like(data.corr()))
        _fig, ax = plt.subplots(figsize=fig_size)
        kwargs.setdefault("cbar", False)
        sns.heatmap(
            data.corr(metric).round(2),
            annot=annot,
            cmap=cmap,
            linewidths=linewidths,
            vmin=-1,
            vmax=1,
            mask=mask,
            ax=ax,
            *args,
            **kwargs,
        )

        ax.set_title("Spearman correlation matrix")
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
        s = "\n".join(
            wrap(
                "Note: One of two highly correlating features should be excluded because...",
                wrap_length,
            )
            + wrap(
                "a. ...it won't contribute any more variance to the model", wrap_length
            )
            + wrap("b. ...both features are likely to be highly colinear.", wrap_length)
        )
        plt.figtext(
            x=0.6,
            y=0.65,
            s=s,
            horizontalalignment="left",
            fontsize=9.0,
            family="serif",
            style="normal",
            fontweight="normal",
            rotation=0,
            bbox={"facecolor": "w", "alpha": 1},
        )  # figtext uses relaive coordinates, if more space is needed uses plt.subplots_adjust(top=.1)
        sns.despine(trim=True)

        return ax

    @staticmethod
    def plot_cv_indices(
        cv,
        X,
        y,
        group=None,
        n_splits: int = 5,
        lw: int = 10,
        fig_size: tuple[int] = (8, 6),
        cmap_data=plt.cm.Paired,
        cmap_cv=plt.cm.coolwarm,
    ) -> plt.Axes:
        """
        Create a sample plot for indices of a cross-validation object. For template, see:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
        """
        _fig, ax = plt.subplots(fig_size=fig_size)
        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0
            # Visualize the results
            ax.scatter(
                range(len(indices)),
                [ii + 0.5] * len(indices),
                c=indices,
                marker="_",
                lw=lw,
                cmap=cmap_cv,
                vmin=-0.2,
                vmax=1.2,
            )
        # Plot the data classes and groups at the end
        ax.scatter(
            range(len(X)), [ii + 1.5] * len(X), c=y, marker="_", lw=lw, cmap=cmap_data
        )
        ax.scatter(
            range(len(X)),
            [ii + 2.5] * len(X),
            c=group,
            marker="_",
            lw=lw,
            cmap=cmap_data,
        )
        # Formatting
        yticklabels = list(range(n_splits)) + ["class", "group"]
        ax.set(
            yticks=np.arange(n_splits + 2) + 0.5,
            yticklabels=yticklabels,
            xlabel="Sample index",
            ylabel="CV iteration",
            ylim=[n_splits + 2.2, -0.2],
            xlim=[0, 100],
        )
        ax.set_title("{}".format(type(cv).__name__), fontsize=15)

        return ax

    @classmethod
    def plot_reg_coefficients(
        cls,
        coefs=None,
        feature_names: bool = None,
        fig_size: tuple[int] = (8, 6),
        wrap_length: int = 60,
        *args,
        **kwargs,
    ) -> plt.Axes:
        if coefs is None:
            try:
                # gridsearch
                coefs = cls.pipe.best_estimator_["estimator"].coefs_
            except Exception as e:
                try:  # NO gridsearch
                    coefs = cls.pipe.estimator["estimator"].coefs_
                except:
                    raise Exception(
                        e,
                        "No coefs found. Did you forgot to provide coefs ?",
                    )
            print("No coefs provided, used instance-specific coefs instead...")
        if not isinstance(coefs, pd.DataFrame):
            coefs = pd.DataFrame(
                coefs,
                columns=["Coefficients"],
                index=feature_names,
            )
        _fig, ax = plt.subplots(fig_size=fig_size)
        sns.barplot(data=coefs, y="Coefficients", orient="h", ax=ax, *args, **kwargs)
        ax.axvline(x=0, color=".75", lw=2)
        ax.set_title("Coefficients")
        ax.set_xlabel("Raw coefficient values")
        s = "\n".join(
            wrap(
                "Note: Values demonstrate conditional dependencies, meaning dependencies between a specific feature and the target, when all other feature remain constant.",
                wrap_length,
            )
        )
        plt.figtext(
            x=0.9,
            y=0.1,
            s=s,
            horizontalalignment="center",
            fontsize=9.5,
            family="serif",
            style="normal",
            fontweight="normal",
            rotation=0,
            bbox={"facecolor": "w", "alpha": 1},
        )  # figtext uses relaive coordinates, if more space is needed uses plt.subplots_adjust(top=.1)
        sns.despine(trim=True)

        return ax

    @staticmethod
    def plot_reg_predictions(
        y_train: list = None,
        y_train_pred: list = None,
        y_test: list = None,
        y_test_pred: list = None,
        datetime_var: datetime = None,
        fig_size: tuple[int] = (8, 6),
        *args,
        **kwargs,
    ) -> list:
        return_val = []
        # If provided, plot actual and predicted TRAINING values
        if (y_train) or (y_train_pred):
            assert all(
                var in locals() for var in ("y_train", "y_train_pred")
            ), "If either is defined, both y_train and y_train_pred have to be defined."
            assert (
                len(y_train) == y_train_pred
            ), "y_train and y_train_pred must have same length."
            _fig1, ax1 = plt.subplots(figsize=fig_size)
            # When a datetime object is provided, we plot actual and predicted against time
            if datetime_var is not None:
                if not isinstance(datetime_var, datetime.datetime):
                    raise ValueError(e, "\nDatetime_var must be a datetime object.")
                try:
                    ax1.plot(
                        x=datetime_var,
                        y=y_train,
                        fmt="r--",
                        label="Actual (y_train)",
                        *args,
                        *kwargs,
                    )
                    ax1.plot(
                        x=datetime_var,
                        y=y_train_pred,
                        fmt="b-",
                        label="Predicted (y_train_pred)",
                        *args,
                        *kwargs,
                    )
                    ax1.set_ylabel("y label")
                except Exception as e:
                    raise ValueError(
                        e, "\nDatetime_var must be a valid datetime object."
                    )
            # When no datetime object is provided, we plot actual against predicted
            else:
                ax1.scatter(
                    y_train,
                    y_train_pred,
                    edgecolors=(0, 0, 0),
                    *args,
                    **kwargs,
                )
                ax1.plot(
                    x=[y_train.min(), y_train.max()],
                    y=[y_train.min(), y_train.max()],
                    lw=4,
                    fmt="r--",
                    label="Perfect performance",
                )
                ax1.set_xlabel("Actual label (y_train)")
                ax1.set_ylabel("Predicted")
            ax1.set_title("Predictions on training data")
            sns.despine(trim=True)
            return_val.append(ax1)
        # If provided, plot actual and predicted TEST values
        if (y_test) or (y_test_pred):
            assert all(
                var in locals() for var in ("y_test", "y_test_pred")
            ), "If either is defined, both y_test and y_test_pred have to be defined."
            assert (
                len(y_test) == y_test_pred
            ), "y_test and y_test_pred must have same length."
            _fig2, ax2 = plt.subplots(figsize=fig_size)
            # If a datetime object is provided, we plot actual and predicted against time
            if datetime_var is not None:
                if not isinstance(datetime_var, datetime.datetime):
                    raise ValueError(e, "\nDatetime_var must be a datetime object.")
                try:
                    ax2.plot(
                        x=datetime_var,
                        y=y_test,
                        fmt="r--",
                        label="Actual label (y_test)",
                        *args,
                        *kwargs,
                    )
                    ax2.plot(
                        x=datetime_var,
                        y=y_test_pred,
                        fmt="b-",
                        label="Predicted (y_test_pred)",
                        *args,
                        *kwargs,
                    )
                    ax2.set_ylabel("y label")
                except Exception as e:
                    raise ValueError(
                        e, "\nDatetime_var must be a valid datetime object."
                    )
            # When no datetime object is provided, we plot actual against predicted
            else:
                ax2.scatter(
                    y_test,
                    y_test_pred,
                    edgecolors=(0, 0, 0),
                    *args,
                    **kwargs,
                )
                ax2.plot(
                    x=[y_test.min(), y_test.max()],
                    y=[y_test.min(), y_test.max()],
                    fmt="r--",
                    label="Perfect Performance",
                    lw=4,
                )
                ax2.set_xlabel("Actual label (y_test)")
                ax2.set_ylabel("Predicted")
            ax2.set_title("Predictions on test data")
            sns.despine(trim=True)
            return_val.append(ax2)

        return tuple(return_val)

    @staticmethod
    def plot_confusion_mat(
        y,
        y_pred,
        normalize: bool = True,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        title: str = "Confusion matrix",
        fig_size: tuple[int] = (8, 6),
        *args,
        **kwargs,
    ) -> plt.Axes:
        con_mat = confusion_matrix(y, y_pred)
        if normalize:
            con_mat = con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis]
        _fig, ax = plt.subplots(fig_size=fig_size)
        sns.heatmap(
            con_mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            linewidths=1,
            linecolor="white",
            cmap=cmap,
            *args,
            **kwargs,
        )
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        ax.set_title(title)
        sns.despine(trim=True)

        return ax

    @staticmethod
    def plot_ROC_binary(
        y_test,
        y_pred_proba,
        palette="icefire",
        title: str = "ROC Curve",
        fig_size: tuple[int] = (8, 6),
        *args,
        **kwargs,
    ) -> tuple[plt.Axes, float]:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        youden_J = tpr - fpr
        ix = np.argmax(youden_J)
        threshold_best = thresholds[ix]

        # Plot
        _fig, ax = plt.subplots(fig_size=fig_size)
        kwargs.setdefault("markers", True)
        sns.lineplot(
            x=fpr,
            y=tpr,
            hue=thresholds,
            palette=sns.color_palette(palette, as_cmap=True),
            ax=ax,
            *args,
            **kwargs,
        )
        ax.plot(
            [0, 1], [0, 1], linestyle="--", label="No Skill classifier", color="gray"
        )
        ax.legend("bottom right")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        sns.despine(trim=True)

        return ax, threshold_best

    @staticmethod
    def plot_PRC_binary(
        y_test,
        y_pred_proba,
        palette="icefire",
        title: str = "ROC Curve",
        fig_size: tuple[int] = (8, 6),
        *args,
        **kwargs,
    ) -> tuple[plt.Axes, float]:
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        threshold_best = thresholds[ix]
        _fig, ax = plt.subplots(fig_size=fig_size)
        sns.lineplot(
            x=precision,
            y=recall,
            hue=thresholds,
            palette=sns.color_palette(palette, as_cmap=True),
            ax=ax,
            markers=True,
            *args,
            **kwargs,
        )
        ax.plot(
            [0, 1], [0, 1], linestyle="--", label="No Skill classifier", color="gray"
        )
        ax.legend("bottom left")
        ax.set_xlabel(r"Recall \frac{TP}{TP+FP}$")
        ax.set_ylabel(r"Precision  $\frac{TP}{TP+FN}$")
        ax.set_title(title)
        sns.despine(trim=True)

        return ax, threshold_best


class PrettyShortEda(_BaseClass):
    """
    Collection of exploratory data analysis methods. Inherits from _BaseClass.
    """

    @classmethod
    def eda_clean_check(
        cls,
        data: pd.DataFrame = None,
        fig_size: tuple[int] = (12, 10),
    ):
        if data is None:
            try:
                data = cls.df_data
                print("No data provided, used instance-specific df_data instead...")
            except Exception as e:
                raise Exception(
                    e,
                    "No data found. Did you forgot to provide data ?",
                )
        # Check Dtypes
        print(data.info(), "\n")
        # Check overall layout
        print(data.describe(), "\n")
        # Check for NaNs visually and written
        _fig, ax = plt.subplots(figsize=fig_size)
        sns.heatmap(
            data.isna(), ax=ax, xticklabels=True
        )  # using the ticklabels command we can plot ALL labels. Omit for y labels because it may take forever to plot when there is alot of rows
        ax.set_title("Overview over all NaNs in dataset")
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
        )
        print("Amount of NaNs:")
        print(data.isna().sum(), "\n")
        # Check balance
        for feature in data.columns:
            print(
                f"Value counts of {feature}:"
            )  #! improve output readability (each value_count is a pd.Series of varying length)
            print(data[feature].value_counts(), "\n")
        # Check histograms
        data.hist(figsize=fig_size)
        plt.show()


class PrettyShortStatics:
    """
    Collection of staticmethods that do not fit semantically to any of the other classes (yet).
    """

    @staticmethod
    def crawl_get_website_content(
        url: str,
        path_save: str = None,
        verbose: bool = False,
        t_wait: float = np.random.uniform(1, 3),
        *args,
        **kwargs,
    ) -> str:
        """
        Simple webcrawler method that requests and, if desired, stores the html
        content of a given website.

        Parameters
        ----------
        url : str
            URL to the website.
        path_save : str, optional
            Path to save the content of the website. If None provided,
            content is not saved, by default None.
        verbose : bool, optional
            Whether or not status code should always be printed or just
            for errors, by default False.
        t_wait : float, optional
            Time, in seconds, to wait before sending the request, by default.
            Random float between 1 and 3. Meant to reduce the likelihood
            of getting banned or timed out by requested server.
        *args / *kwargs
            Will be passed to the pathlib.write_text() function.

        Returns
        -------
        request.text : str
            Website content.
        """
        if t_wait:
            time.sleep(t_wait)
        try:
            req = requests.get(url)
            if verbose:
                print(f"Website status code: {req.status_code}")
            if (req.status_code >= 400) and (req.status_code < 500):
                print(
                    f"\nERROR: User authorization or input error. Server response: {req.status_code}"
                )
            elif (req.status_code >= 500) and (req.status_code < 600):
                print(
                    f"\nERROR: Server-sided error. Server response: {req.status_code}"
                )
            if path_save:
                try:
                    path_save = Path(
                        str(Path(".").resolve()) + "/" + path_save
                    )  # this is nasty, but VSC seems to mess up pathlib functions
                    path_save.parent.mkdir(exist_ok=True, parents=True)
                    path_save.write_text(req.text, *args, **kwargs)
                except Exception as e:
                    print(e, "\nERROR: Website could not be written to file.")

            return req.text
        except Exception as e:
            print(e, "\nERROR: Could not request url !")

    @staticmethod
    def NLP_get_lang_proba(text: str, language: str = "en") -> float:
        """
        Detects the likelihood a given string is in the specified language.

        Parameters
        ----------
        text : str
            The text string that should be analyzed.
        language : str, optional
            The language code for the analyzer, by default "en" for English.

        Returns
        -------
        float
            Returns the probability that the text is in the language specified.
        """
        detections = detect_langs(text)
        for detection in detections:
            if detection.lang == language:

                return detection.prob

        return 0.0


@dataclass
class PrettyShortML(
    PrettyShortModelling, PrettyShortPlotting, PrettyShortEda, PrettyShortStatics
):
    """
    PrettyShortML is a set of classes that contain blue-print-like methods for crucial steps
    in a typical sklearn Machine Learning work-flow, including but not limited to exploratory
    data analysis (EDA), modelling, model evaluation and many different visualizations.
    Functionally, PrettyShortML is an empty main class that inherits methods from all other
    classes.
    """
