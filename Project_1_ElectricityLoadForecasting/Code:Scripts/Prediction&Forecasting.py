# Load required packages/modules/dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import r2_score

# Plotting defaults
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True

# Load the dataset (AEP_hourly.csv)
df = pd.read_csv(
    "xxxx/Test_Lakehouse.Lakehouse/Files/Data/AEP_hourly.csv",
    parse_dates=["Datetime"],   
    index_col="Datetime"
)

# Sort by date and time
df = df.sort_index()
df.head()
df.info()

# Subet the data
series = df["AEP_MW"]
data = series.loc["2015-01-01":"2017-12-31"] # Use 3 years total: 2 years train, 1 year test 

# Resample hourly - daily mean (for trend / pattern / faster training)
daily = data.resample("D").mean()
daily.plot(title="Daily Average Load - AEP (2015–2017)")
plt.ylabel("MW")
plt.show()

# Split train - test
train = daily.loc["2015-01-01":"2016-12-31"]
test  = daily.loc["2017-01-01":"2017-12-31"]
print("Train size:", train.shape[0], "days")
print("Test size:", test.shape[0], "days")

def make_features(series: pd.Series) -> pd.DataFrame:
    df_feat = series.to_frame(name="load")
    
    # lags (previous days)
    df_feat["lag_1"]  = df_feat["load"].shift(1)
    df_feat["lag_7"]  = df_feat["load"].shift(7)
    df_feat["lag_14"] = df_feat["load"].shift(14)

    # calendar features
    df_feat["dayofweek"] = df_feat.index.dayofweek
    df_feat["month"]     = df_feat.index.month
    df_feat["is_weekend"] = (df_feat["dayofweek"] >= 5).astype(int)

    return df_feat

full_feat = make_features(daily)

full_feat = full_feat.dropna()

# Re-split with features
train_feat = full_feat.loc["2015-01-01":"2016-12-31"]
test_feat  = full_feat.loc["2017-01-01":"2017-12-31"]
X_train = train_feat.drop(columns=["load"])
y_train = train_feat["load"]

X_test = test_feat.drop(columns=["load"])
y_test = test_feat["load"]

X_train.head()

# Naive (Baseline) model
y_pred_naive = test_feat["lag_1"]

mae_naive = mean_absolute_error(y_test, y_pred_naive)
rmse_naive = mean_squared_error(y_test, y_pred_naive, squared=False)

print(f"Naive baseline - MAE: {mae_naive:.2f}, RMSE: {rmse_naive:.2f}")

# Cost-aware loss function for Naive model 
def cost_aware_loss(y_true, y_pred, under_cost=2.0, over_cost=1.0):
    diff = y_pred - y_true
    under = np.where(diff < 0, -diff, 0)   # predicted < actual
    over  = np.where(diff > 0,  diff, 0)   # predicted > actual
    return under_cost * under.mean() + over_cost * over.mean()

cost_naive = cost_aware_loss(y_test.values, y_pred_naive.values)
print(f"Naive baseline - cost-aware loss: {cost_naive:.2f}")

# Random Forest (RF) Model
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred_rf = rf.predict(X_test)

mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
cost_rf = cost_aware_loss(y_test.values, y_pred_rf)

print(f"RandomForest - MAE: {mae_rf:.2f}, RMSE: {rmse_rf:.2f}, Cost-aware: {cost_rf:.2f}")

# Plot Actual (Observed) vs Predictions using  (Naive n RF)
plt.figure(figsize=(14,5))
y_test["2017-01-01":"2017-03-31"].plot(label="Actual")
pd.Series(y_pred_naive, index=y_test.index)["2017-01-01":"2017-03-31"].plot(label="Naive", alpha=0.7)
pd.Series(y_pred_rf, index=y_test.index)["2017-01-01":"2017-03-31"].plot(label="RandomForest", alpha=0.7)
plt.title("Daily Load Forecasts (Jan–Mar 2017)")
plt.ylabel("MW")
plt.legend()
plt.show()

# Parameter Optimisation (balancing accuracy vs training time) for Random Forest model 

tscv = TimeSeriesSplit(n_splits=4)

param_dist = {
    "n_estimators": [100, 150, 200, 300],
    "max_depth": [5, 8, 10, 12, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)

random_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=param_dist,
    n_iter=20,             
    cv=tscv,
    scoring="neg_mean_absolute_error",  
    random_state=42,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

best_rf = random_search.best_estimator_
best_rf.fit(X_train, y_train)
y_pred_best = best_rf.predict(X_test)

mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
cost_best = cost_aware_loss(y_test.values, y_pred_best)

print("Best params:", random_search.best_params_)
print("Best CV MAE:", -random_search.best_score_)
print(f"Best RF - MAE: {mae_best:.2f}, RMSE: {rmse_best:.2f}, Cost-aware: {cost_best:.2f}")

# Future 30 days Prediction and Forecast

horizon = 30
last_date = daily.index.max()
future_index = pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

# Start from the full series up to last_date
extended_series = daily.copy()

future_preds = []

for date in future_index:
    # Build feature row for date
    lag_1  = extended_series.loc[date - pd.Timedelta(days=1)]
    lag_7  = extended_series.loc[date - pd.Timedelta(days=7)]
    lag_14 = extended_series.loc[date - pd.Timedelta(days=14)]
    
    dayofweek = date.dayofweek
    month     = date.month
    is_weekend = int(dayofweek >= 5)
    
    x = pd.DataFrame([{
    "lag_1": lag_1,
    "lag_7": lag_7,
    "lag_14": lag_14,
    "dayofweek": dayofweek,
    "month": month,
    "is_weekend": is_weekend
}])

    
    y_hat = best_rf.predict(x)[0]
    future_preds.append(y_hat)
    
    extended_series.loc[date] = y_hat

future_forecast = pd.Series(future_preds, index=future_index, name="forecast")

future_forecast.plot()
plt.title("30-day Ahead Forecast – AEP Daily Load")
plt.ylabel("MW")
plt.show()

# User defined functions for metrics
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def accuracy_from_mape(y_true, y_pred):
    return 100 - mape(y_true, y_pred)

# Calculate Accuracy for RF Model
y_pred_rf = rf.predict(X_test)

# Random Forest Metrics
mae_rf = mean_absolute_error(y_test, y_pred_rf)
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
r2_rf = r2_score(y_test, y_pred_rf)

mape_rf = mape(y_test, y_pred_rf)
accuracy_rf = accuracy_from_mape(y_test, y_pred_rf)

print("Random Forest Performance:")
print(f"MAE:      {mae_rf:.2f}")
print(f"RMSE:     {rmse_rf:.2f}")
print(f"R²:       {r2_rf:.4f}")
print(f"MAPE:     {mape_rf:.2f}%")
print(f"Accuracy: {accuracy_rf:.2f}%")

# Calculate Accuracy for Tuned RF Model
y_pred_best = best_rf.predict(X_test)

# RF (tuned) metrics
mae_best = mean_absolute_error(y_test, y_pred_best)
rmse_best = mean_squared_error(y_test, y_pred_best, squared=False)
r2_best = r2_score(y_test, y_pred_best)

mape_best = mape(y_test, y_pred_best)
accuracy_best = accuracy_from_mape(y_test, y_pred_best)

print("\nTuned Random Forest Performance:")
print(f"MAE:      {mae_best:.2f}")
print(f"RMSE:     {rmse_best:.2f}")
print(f"R²:       {r2_best:.4f}")
print(f"MAPE:     {mape_best:.2f}%")
print(f"Accuracy: {accuracy_best:.2f}%")

# Calculate Accuracy for Navie Model
y_pred_naive = test_feat["lag_1"]

#  Naive Model Metrics
mae_naive = mean_absolute_error(y_test, y_pred_naive)
rmse_naive = mean_squared_error(y_test, y_pred_naive, squared=False)
r2_naive = r2_score(y_test, y_pred_naive)

mape_naive = mape(y_test, y_pred_naive)
accuracy_naive = accuracy_from_mape(y_test, y_pred_naive)
cost_naive = cost_aware_loss(y_test.values, y_pred_naive.values)

print("Naive Model Performance:")
print(f"MAE:      {mae_naive:.2f}")
print(f"RMSE:     {rmse_naive:.2f}")
print(f"R²:       {r2_naive:.4f}")
print(f"MAPE:     {mape_naive:.2f}%")
print(f"Accuracy: {accuracy_naive:.2f}%")
print(f"Cost-aware Loss: {cost_naive:.2f}")


# Compute Metrics for all three models
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def accuracy_from_mape(y_true, y_pred):
    return 100 - mape(y_true, y_pred)

def cost_aware_loss(y_true, y_pred, under_cost=2.0, over_cost=1.0):
    diff = y_pred - y_true
    under = np.where(diff < 0, -diff, 0)
    over  = np.where(diff > 0,  diff, 0)
    return under_cost * under.mean() + over_cost * over.mean()


# Naive
metrics_naive = {
    "MAE": mean_absolute_error(y_test, y_pred_naive),
    "RMSE": mean_squared_error(y_test, y_pred_naive, squared=False),
    "MAPE": mape(y_test, y_pred_naive),
    "Accuracy": accuracy_from_mape(y_test, y_pred_naive),
    "R2": r2_score(y_test, y_pred_naive),
    "Cost": cost_aware_loss(y_test.values, y_pred_naive.values)
}

# Random Forest (RF)
metrics_rf = {
    "MAE": mean_absolute_error(y_test, y_pred_rf),
    "RMSE": mean_squared_error(y_test, y_pred_rf, squared=False),
    "MAPE": mape(y_test, y_pred_rf),
    "Accuracy": accuracy_from_mape(y_test, y_pred_rf),
    "R2": r2_score(y_test, y_pred_rf),
    "Cost": cost_aware_loss(y_test.values, y_pred_rf)
}

# Tuned Random Forest 
metrics_best = {
    "MAE": mean_absolute_error(y_test, y_pred_best),
    "RMSE": mean_squared_error(y_test, y_pred_best, squared=False),
    "MAPE": mape(y_test, y_pred_best),
    "Accuracy": accuracy_from_mape(y_test, y_pred_best),
    "R2": r2_score(y_test, y_pred_best),
    "Cost": cost_aware_loss(y_test.values, y_pred_best)
}

metrics_to_plot = ["MAE", "RMSE", "MAPE", "Accuracy", "R2"]
models = ["Naive", "RandomForest", "TunedRF"]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 12))
axes = axes.flatten()

for i, metric in enumerate(metrics_to_plot):
    ax = axes[i]
    
    vals = results_df_metrics.loc[metric, models].values
    ax.bar(range(len(models)), vals, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=0)
    ax.set_title(metric)
    ax.set_ylabel(metric)
    ax.grid(True, linestyle="--", alpha=0.5)

# Hide the empty grid
axes[-1].axis("off")

plt.suptitle("Model Comparison Across Metrics", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()