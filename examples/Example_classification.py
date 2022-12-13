# Imports
from PrettyShortML import PrettyShortML as psml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

# Load in sample data: breast cancer classification (binary classes)
random_state = 42
X, y = datasets.load_breast_cancer(return_X_y=True)
X = pd.DataFrame(X[:, :10])  # to speed up, we just take the first 10 features
y = pd.DataFrame(y)
# Also, Sklearn does not like int columns
X.columns = X.columns.astype(str)
y.columns = y.columns.astype(str)

# Do basic EDA
psml.eda_clean_check(X)

# This data is workable, so let's load a class instance after doing a train-test split
data_train, data_test, labels_train, labels_test = train_test_split(
    X, y, test_size=0.3, random_state=random_state
)
my_dataset = psml(
    X_train=data_train,
    X_test=data_test,
    y_train=labels_train,
    y_test=labels_test,
)

# Plot feature pair plots to inspect distributions and linear relationships
_plotting_ax = psml.plot_eda_pairplot(
    data=my_dataset.X_train, corner=True, dropna=False, plot_hist=False, fig_size=(9, 6)
)

# Plot feature correlation matrix to spot colinearity
_plotting_ax = psml.plot_eda_corr_mat(
    data=my_dataset.X_train, metric="spearman", cmap="vlag", mask=True, annot=True
)

# Normally, we'd do a decent amount of feature cleaning and engineering here.
# For now, let's just do hierarchical clustering on the features' rank-order correlations
# to select features.
_plotting_ax, dist_linkage = psml.plot_hierarchical_clustering(data=my_dataset.X_train)
selected_features = psml.cluster_informed_feature_selection(
    data=my_dataset.X_train, cluster_threshold=0.5
)
# Select the features
my_dataset.X_train = my_dataset.X_train.iloc[:, selected_features]
my_dataset.X_test = my_dataset.X_test.iloc[:, selected_features]

# let's train the model now using a CV splitter and some
# params to do gridsearch. Model_train runs SimpleImputer and StandardScaler on default
# for numerical features. We can use the instance instead of passing the data explicitly.
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=random_state)
param_grid = {
    "estimator__solver": ("lbfgs", "liblinear", "newton-cg"),
    "estimator__fit_intercept": (True, False),
}
pipeline, training_score = my_dataset.model_train(
    LogisticRegression(),
    numeric_features=my_dataset.X_train.columns,
    param_grid=param_grid,
    cv=cv_splitter,
)
print(f"The model's performance on training data:\n{training_score:.4f}")

# Let's evaluate our model on the test data
accuracy, precision, recall, f1score = my_dataset.model_clf_evaluate(
    average="binary", plot_confusion=True, normalize_conmat=False
)
print(
    "The model's performance on test data:",
    f"\nAccuracy: {accuracy:.4f}",
    f"\nPrecision: {precision:.4f}",
    f"\nRecall: {recall:.4f}",
    f"\nf1-score: {f1score:.4f}",
)

# Let's plot the ROC and precision recall curve
y_test_proba = pipeline.predict_proba(my_dataset.X_test)
# ROC
_plotting_ax, best_threshold = psml.plot_ROC_binary(
    y_test=my_dataset.y_test, y_pred_proba=y_test_proba[:, 1], average="macro"
)

# PRC
_plotting_ax, best_threshold = psml.plot_PRC_binary(
    y_test=my_dataset.y_test, y_pred_proba=y_test_proba[:, 1], average="macro"
)