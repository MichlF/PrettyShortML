# Imports
import datetime
from textwrap import wrap
from dataclasses import dataclass
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from submodules.BaseClass import _BaseClass


@dataclass
class Plotting(_BaseClass):
    """
    Collection of blue print methods to plot metrics of ML pipelines. Inherits from _BaseClass.
    """

    def plot_eda_pairplot(
        self,
        data: pd.DataFrame = None,
        kind="reg",
        corner: bool = True,
        dropna: bool = True,
        plot_hist: bool = True,
        fig_size: tuple[int, int] = (12, 8),
        *args,
        **kwargs,
    ) -> plt.Axes:
        print(
            "Starting to plot pairplots. Depending on the dateset size, this may take a while...",
            "\nShould it take too long, consider changing the default 'kind' parameter to something else than 'reg'.",
        )
        if plot_hist:
            _, ax = plt.subplots(figsize=fig_size)
            data.hist(ax=ax)
            plt.show()
        ax = sns.pairplot(
            data=data,
            kind=kind,
            diag_kind="kde",
            corner=corner,
            dropna=dropna,
            plot_kws={"scatter_kws": {"alpha": 0.2}},
            *args,
            **kwargs,
        )
        ax.fig.set_size_inches(fig_size)
        ax.fig.suptitle("Pairwise relationships")
        sns.despine()
        plt.show()

        return ax

    @staticmethod
    def plot_hierarchical_clustering(
        data: pd.DataFrame,
        metric: str = "spearman",
        fig_size: tuple[int, int] = (10, 6),
        *args,
        **kwargs,
    ) -> tuple[plt.Axes, np.ndarray]:
        # https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
        corr = data.corr(metric)
        _, ax = plt.subplots(figsize=fig_size)
        # Convert to distance matrix
        distance_matrix = 1 - np.abs(corr)
        dist_linkage = hierarchy.ward(squareform(distance_matrix))
        # Plot hierarchical clustering
        hierarchy.dendrogram(
            dist_linkage,
            labels=data.columns.tolist(),
            ax=ax,
            leaf_rotation=90,
            *args,
            **kwargs,
        )
        ax.set_title("Hierarchical clustering as dendrogram (using Ward's linkage")
        ax.set_ylabel("Threshold")
        ax.set_xlabel("Feature labels")
        plt.show()

        return ax, dist_linkage

    @staticmethod
    def plot_eda_corr_mat(
        data: pd.DataFrame = None,
        metric: str = "spearman",
        cmap: str = "vlag",
        mask: bool = True,
        annot: bool = True,
        linewidths: int = 0.5,
        fig_size: tuple[int, int] = (12, 8),
        wrap_length: int = 60,
        *args,
        **kwargs,
    ) -> plt.Axes:
        if mask:
            mask = np.triu(np.ones_like(data.corr()))
        _, ax = plt.subplots(figsize=fig_size)
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
        )  # figtext uses relative coordinates, if more space is needed uses plt.subplots_adjust(top=.1)
        sns.despine(trim=True)

        return ax

    @staticmethod
    def plot_cv_indices(
        cv,
        X,
        y,
        groups=None,
        n_splits: int = 5,
        lw: int = 10,
        fig_size: tuple[int, int] = (8, 5),
        cmap_data=plt.cm.Paired,
        cmap_cv=plt.cm.coolwarm,
    ) -> plt.Axes:
        """
        Create a sample plot for indices of a cross-validation object. For template, see:
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html
        """
        _, ax = plt.subplots(figsize=fig_size)
        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=groups)):
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
            c=groups,
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
        plt.show()

        return ax

    def plot_reg_coefficients(
        self,
        coefs=None,
        feature_names: bool = None,
        fig_size: tuple[int, int] = (10, 6),
        zero_line: bool = False,
        annot: bool = False,
        wrap_length: int = 50,
        *args,
        **kwargs,
    ) -> plt.Axes:
        if not isinstance(coefs, pd.DataFrame):
            coefs = pd.DataFrame(
                coefs,
                columns=["Coefficients"],
                index=feature_names,
            ).T
        _, ax = plt.subplots(figsize=fig_size)
        sns.barplot(data=coefs, orient="h", ax=ax, *args, **kwargs)
        if zero_line:
            ax.axvline(x=0, color=".75", lw=2)
        ax.set_title("Coefficients")
        ax.set_xlabel("Raw coefficient values")
        ax.set_ylabel("Features")
        s = "\n".join(
            wrap(
                "Note: Values demonstrate conditional dependencies, meaning dependencies between a specific feature and the target, when all other feature remain constant.",
                wrap_length,
            )
        )
        if annot:
            plt.figtext(
                x=0.9,
                y=0.2,
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
    def plot_reg_prediction_errors(
        y_train: list = None,
        y_train_pred: list = None,
        y_test: list = None,
        y_test_pred: list = None,
        datetime_var: datetime = None,
        fig_size: tuple[int, int] = (10, 6),
        *args,
        **kwargs,
    ) -> list:
        # Similar to the newly (1.2v) published sklearn function PredictionErrorDisplay
        # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.PredictionErrorDisplay.html#sklearn.metrics.PredictionErrorDisplay.from_estimator
        return_val = []
        # If provided, plot actual and predicted TRAINING values
        if (y_train) is not None or (y_train_pred) is not None:
            assert (isinstance(y_train, (pd.DataFrame, np.ndarray, list))) and (
                isinstance(y_train_pred, (pd.DataFrame, np.ndarray, list))
            ), "If either is defined, both y_train and y_train_pred have to be defined."
            assert len(y_train) == len(
                y_train_pred
            ), "y_train and y_train_pred must have same length."
            _, ax1 = plt.subplots(figsize=fig_size)
            # When a datetime object is provided, we plot actual and predicted against time
            if datetime_var is not None:
                if not isinstance(datetime_var, datetime.datetime):
                    raise ValueError("\nDatetime_var must be a datetime object.")
                try:
                    ax1.plot(
                        datetime_var,  # x
                        y_train,  # y
                        "r--",  # fmt
                        label="Actual (y_train)",
                        *args,
                        *kwargs,
                    )
                    ax1.plot(
                        x=datetime_var,  # x
                        y=y_train_pred,  # y
                        fmt="b-",  # fmt
                        label="Predicted (y_train_pred)",
                        *args,
                        *kwargs,
                    )
                    ax1.set_ylabel("y label")
                except Exception as e:
                    raise ValueError(
                        e, "\nDatetime_var must be a valid datetime object."
                    ) from e
            # When no datetime object is provided, we plot actual against predicted
            else:
                data = pd.concat(
                    [
                        y_train.reset_index(drop=True),
                        pd.DataFrame(y_train_pred).reset_index(drop=True),
                    ],
                    axis=1,
                )
                sns.lineplot(
                    x=[float(y_train.min()), float(y_train.max())],
                    y=[float(y_train.min()), float(y_train.max())],
                    ax=ax1,
                    errorbar=None,
                    linestyle="--",
                    lw=4,
                    color="red",
                    label="Perfect performance",
                )
                sns.scatterplot(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    ax=ax1,
                    edgecolors=(0, 0, 0),
                    *args,
                    **kwargs,
                )
                ax1.set_xlabel("Actual value (y_train)")
                ax1.set_ylabel("Predicted")
            ax1.set_title("Predictions on training data")
            sns.despine(trim=True)
            return_val.append(ax1)
            plt.show()
        # If provided, plot actual and predicted TEST values
        if (y_test) is not None or (y_test_pred) is not None:
            assert (isinstance(y_test, (pd.DataFrame, np.ndarray, list))) and (
                isinstance(y_test_pred, (pd.DataFrame, np.ndarray, list))
            ), "If either is defined, both y_train and y_train_pred have to be defined."
            assert len(y_test) == len(
                y_test_pred
            ), "y_train and y_train_pred must have same length."
            _, ax2 = plt.subplots(figsize=fig_size)
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
                    ) from e
            # When no datetime object is provided, we plot actual against predicted
            else:
                data = pd.concat(
                    [
                        y_test.reset_index(drop=True),
                        pd.DataFrame(y_test_pred).reset_index(drop=True),
                    ],
                    axis=1,
                )
                sns.lineplot(
                    x=[float(y_test.min()), float(y_test.max())],
                    y=[float(y_test.min()), float(y_test.max())],
                    ax=ax2,
                    errorbar=None,
                    linestyle="--",
                    lw=4,
                    color="red",
                    label="Perfect performance",
                )
                sns.scatterplot(
                    x=data.iloc[:, 0],
                    y=data.iloc[:, 1],
                    ax=ax2,
                    edgecolors=(0, 0, 0),
                    *args,
                    **kwargs,
                )
                ax2.set_xlabel("Actual value (y_test)")
                ax2.set_ylabel("Predicted")
            ax2.set_title("Predictions on test data")
            sns.despine(trim=True)
            return_val.append(ax2)
            plt.show()

        return tuple(return_val)

    @staticmethod
    def plot_confusion_mat(
        y,
        y_pred,
        normalize: bool = True,
        cmap=sns.color_palette("YlOrBr", as_cmap=True),
        title: str = "Confusion matrix",
        fig_size: tuple[int, int] = (8, 5),
        vmin=-1,
        vmax=1,
        *args,
        **kwargs,
    ) -> plt.Axes:
        con_mat = confusion_matrix(y, y_pred)
        if normalize:
            con_mat = con_mat.astype("float") / con_mat.sum(axis=1)[:, np.newaxis]
        _, ax = plt.subplots(fig_size=fig_size)
        sns.heatmap(
            con_mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            linewidths=1,
            linecolor="white",
            vmin=vmin,
            vmax=vmax,
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
        average="macro",
        fig_size: tuple[int, int] = (10, 6),
        title: str = "ROC Curve",
        label="ROC Curve",
        *args,
        **kwargs,
    ) -> tuple[plt.Axes, float]:
        roc_auc = roc_auc_score(
            y_test,
            y_pred_proba,
            average=average,
        )
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        youden_j = tpr - fpr
        ix = np.argmax(youden_j)
        threshold_best = thresholds[ix]

        # Plot
        _, ax = plt.subplots(figsize=fig_size)
        sns.lineplot(
            x=fpr,
            y=tpr,
            ax=ax,
            errorbar=None,
            label=label + f" {average} average (AUC = {roc_auc:.2f})",
            marker=".",
            *args,
            **kwargs,
        )
        sns.lineplot(
            x=[0, 1],
            y=[0, 1],
            ax=ax,
            errorbar=None,
            label="Chance level classification (AUC = 0.5)",
            linestyle="--",
            color="grey",
        )
        ax.legend(loc="lower right")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title(title)
        sns.despine(trim=True)
        plt.show()

        return ax, threshold_best

    @staticmethod
    def plot_PRC_binary(
        y_test,
        y_pred_proba,
        average: str = "macro",
        fig_size: tuple[int, int] = (10, 6),
        title: str = "Precision Recall Curve",
        label="PRC Curve",
        *args,
        **kwargs,
    ) -> tuple[plt.Axes, float]:
        average_precision = average_precision_score(y_test, y_pred_proba)
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        fscore = (2 * precision * recall) / (precision + recall)
        ix = np.argmax(fscore)
        threshold_best = thresholds[ix]
        _, ax = plt.subplots(figsize=fig_size)
        sns.lineplot(
            x=precision,
            y=recall,
            ax=ax,
            errorbar=None,
            label=label + f" {average} average (AP = {average_precision:.2f})",
            marker=".",
            *args,
            **kwargs,
        )
        ax.legend(loc="lower left")
        ax.set_xlabel(r"Recall   ( $\frac{TP}{TP+FP}$ )")
        ax.set_ylabel(r"Precision   ( $\frac{TP}{TP+FN}$ )")
        ax.set_title(title)
        sns.despine(trim=True)
        plt.show()

        return ax, threshold_best