# Imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from PrettyShortML import PrettyShortML as psml

# Load in sample data: regression problem
RANDOM_STATE = 42
n_samples, n_features, n_informative = (
    3000,
    15,
    7,
)
X, y = make_regression(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=n_informative,
    bias=500,
    noise=150,
    random_state=RANDOM_STATE,
)
X = pd.DataFrame(X)
y = pd.DataFrame(y)
# Also, Sklearn does not like int columns
X.columns = X.columns.astype(str)
y.columns = y.columns.astype(str)

# Do basic EDA
psml.eda_clean_check(X)

# This data is workable, so let's load a class instance after doing a train-test split
data_train, data_test, labels_train, labels_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
my_dataset = psml(
    X_train=data_train,
    X_test=data_test,
    y_train=labels_train,
    y_test=labels_test,
)

# Plot feature pair plots to inspect distributions and linear relationships
_plotting_ax = my_dataset.plot_eda_pairplot(
    my_dataset.X_train, corner=True, dropna=False, plot_hist=False, fig_size=(9, 6)
)

# Plot feature correlation matrix to spot colinearity
_plotting_ax = my_dataset.plot_eda_corr_mat(
    my_dataset.X_train, metric="spearman", cmap="vlag", mask=True, annot=True
)

# Normally, we'd do a decent amount of feature cleaning and engineering here.
# For now, let's just do hierarchical clustering on the features' rank-order correlations
# to select features.
# We could run the following to just inspect clustering:
# _plotting_ax, dist_linkage = my_dataset.plot_hierarchical_clustering(
#     data=my_dataset.X_train
# )
# When we know the threshold we want to choose, we can run:
selected_features = my_dataset.cluster_informed_feature_selection(
    data=my_dataset.X_train, cluster_threshold=0.5
)
# Select the features
# In this example, this will essentially selected all features.
my_dataset.X_train = my_dataset.X_train.iloc[:, selected_features]
my_dataset.X_test = my_dataset.X_test.iloc[:, selected_features]

# let's train the model now using a CV splitter and some
# params to do gridsearch. Model_train runs SimpleImputer and StandardScaler on default
# for numerical features. We can use the instance instead of passing the data explicitly.
cv_splitter = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
param_grid = {
    "estimator__fit_intercept": (True, False),
}
pipeline, training_score = my_dataset.model_train(
    LinearRegression(),
    numeric_features=tuple(my_dataset.X_train.columns),
    param_grid=param_grid,
    cv=cv_splitter,
    n_jobs=-1,
)
print(f"The model's performance on training data:\n{training_score:.4f}")

# Let's evaluate our model on the test data
# First, retrieve our scoring metric
# CV was already applied so just predict b/c the model is refit on the best model performance
print(
    "The score on predicting the test data using the best model is"
    f" {pipeline.score(my_dataset.X_test, my_dataset.y_test):.3f}."
)

# Now, plot our coefficients
_plotting_ax = my_dataset.plot_reg_coefficients(coefs=pipeline.best_estimator_["estimator"].coef_)

# Now our predictions against the real values
y_train_pred = pipeline.predict(my_dataset.X_train)
y_test_pred = pipeline.predict(my_dataset.X_test)
_plotting_axes = my_dataset.plot_reg_prediction_errors(
    y_train=my_dataset.y_train,
    y_train_pred=y_train_pred,
    y_test=my_dataset.y_test,
    y_test_pred=y_test_pred,
    plotting_residuals=False,
)
# Finally, plot our predictions against the residuals
residuals_train = my_dataset.y_train.iloc[:, 0] - y_train_pred
residuals_test = my_dataset.y_test.iloc[:, 0] - y_test_pred
_plotting_axes = my_dataset.plot_reg_prediction_errors(
    y_train=residuals_train,
    y_train_pred=y_train_pred,
    y_test=residuals_test,
    y_test_pred=y_test_pred,
    plotting_residuals=True,
)