# Imports
from dataclasses import dataclass
from collections import defaultdict
from IPython.display import display
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
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
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn import set_config
from submodules.BaseClass import _BaseClass
from submodules.Plotting import Plotting


@dataclass
class Modelling(_BaseClass):
    """
    Class containing blue print methods for quickly run standard model training and evaluation pipelines.
    Inherits from _BaseClass.
    """

    @staticmethod
    def data_undersampler(
        df: pd.DataFrame,
        max_sample_size: int,
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
                imblearn_obj = NearMiss(sampling_strategy={0: max_sample_size}, *args, **kwargs)
            else:
                raise ValueError(f"Invalid value for method_imblearn: {method_imblearn}")

            return imblearn_obj

        else:
            raise ValueError(f"Invalid value for method: {method}")

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
            raise NotImplementedError("Not yet here. Come back later, please.")

        elif method == "imblearn":
            if method_imblearn == "ros":
                imblearn_obj = RandomOverSampler(
                    sampling_strategy={1: min_sample_size}, *args, **kwargs
                )
            elif method_imblearn == "smote":
                imblearn_obj = SMOTE(sampling_strategy={1: min_sample_size}, *args, **kwargs)
            else:
                raise ValueError(f"Invalid value for method_imblearn: {method_imblearn}")

            return imblearn_obj
        else:
            raise ValueError(f"Invalid value for method: {method}")

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
        transform_output: str = "pandas",
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
        transform_output : list, optional
            ! EMPTY, defaults None.
        *args / *kwargs
            Will be passed to the sklearn.GridsearchCV() function.

        Returns
        -------
        Pipeline : Scitkit-learn Pipeline object
            Pipeline object with fitted estimator as class attribute.
        accuracy : float
            Classification accuracy obtained with sklearn score method.
        """
        print(f"Building and fitting {estimator_object.__class__.__name__} estimator...")
        try:
            set_config(transform_output=transform_output)
        except Exception as e:
            print(e, "Couldn't set sklearn's set_config. Is your sklearn version 1.2+ ?")
        # Build transformer for preprocessor
        transformer = []
        if numeric_features is not None:
            numeric_transformer = Pipeline(
                steps=[
                    ("imputer", SimpleImputer()),
                    ("scaler", StandardScaler()),
                ]
            )
            transformer.append(("numeric", numeric_transformer, numeric_features))
        if categorical_features is not None:
            categorical_transformer = OneHotEncoder(handle_unknown="ignore", drop="first")
            transformer.append(("categoric", categorical_transformer, categorical_features))
        if ordinal_features is not None:
            assert (
                ordinal_categories
            ), "When using ordinal_features, ordinal_categories has to be provided !"
            ordinal_transformer = OrdinalEncoder(
                categories=ordinal_categories, handle_unknown="ignore"
            )
            transformer.append(("ordinal", ordinal_transformer, ordinal_features))
        if add_transformers is not None:
            transformer.append(add_transformers)
        self.preprocessor = ColumnTransformer(transformers=transformer)
        # Build pipeline
        pipeline_steps = [
            ("preprocessor", self.preprocessor),
            ("estimator", estimator_object),
        ]
        if add_pipesteps is not None:
            pipeline_steps.append(add_pipesteps)
        self.pipe = Pipeline(steps=pipeline_steps)
        # Optional: perform gridsearch
        if param_grid is not None:
            self.pipe = GridSearchCV(
                estimator=self.pipe,
                param_grid=param_grid,
                *args,
                **kwargs,
            )
        # Fit pipeline
        if isinstance(self.y_train, pd.DataFrame):
            self.pipe.fit(self.X_train, self.y_train.values.ravel())
        else:
            self.pipe.fit(self.X_train, self.y_train)
        # Get basic metrics
        if param_grid is not None: # GridSearch applies CV, so return best_score
            train_accuracy = self.pipe.best_score_
        else:
            train_accuracy = self.pipe.score(self.X_train, self.y_train)
        # Display pipeline object
        display(self.pipe)

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
        print(f"Building and fitting {estimator_object.__class__.__name__} estimator...")
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
        print_class_report: bool = True,
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
            except Exception as _e:
                raise ValueError(
                    _e,
                    "No model found. Did you forgot to provide a model or did you not run a model"
                    " training function ?",
                ) from _e
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
            self.y_test.value_counts()
            if isinstance(self.y_test, pd.DataFrame)
            else pd.Series(self.y_test).value_counts(),
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
                fmt=".4f" if normalize_conmat else "d",
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
        if print_class_report:
            print("The model's performance on test data:\n")
            print(
                classification_report(
                    y_true=self.y_test,
                    y_pred=y_pred,
                    labels=self.y_test.iloc[:, 0].unique(),
                    digits=4,
                )
            )

        return accuracy, precision, recall, f1score

    def model_reg_evaluate(
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
                raise ValueError(
                    e,
                    "No model found. Did you forgot to provide a model or did you not run a model"
                    " training function ?",
                ) from e
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
            self.y_test.value_counts()
            if isinstance(self.y_test, pd.DataFrame)
            else pd.Series(self.y_test).value_counts(),
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
                fmt=".4f" if normalize_conmat else "d",
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

    @staticmethod
    def cluster_informed_feature_selection(
        data: pd.DataFrame, cluster_threshold: int, dist_linkage=None
    ):
        if dist_linkage is None:
            _, dist_linkage = Plotting.plot_hierarchical_clustering(
                data=data,
                metric="spearman",
            )
        cluster_ids = hierarchy.fcluster(dist_linkage, cluster_threshold, criterion="distance")
        cluster_id_to_feature_ids = defaultdict(list)
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_id_to_feature_ids[cluster_id].append(idx)
        selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]

        return selected_features
