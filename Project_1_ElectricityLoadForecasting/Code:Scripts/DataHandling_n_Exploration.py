# Load required packages/modules/dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from statsmodels.tsa.seasonal import STL

# plotting defaults
plt.rcParams["figure.figsize"] = (12, 4)
plt.rcParams["axes.grid"] = True
sns.set(style="whitegrid")


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

# Subet the data; select two years 
series = df["AEP_MW"]    
series_2y = series.loc["2015-01-01":"2016-12-31"]

series_2y.head(), series_2y.tail()

# Basic and standard data curation

series_2y.describe()
series_2y.isna().sum(), series_2y.shape[0] # missing values 
missing = series_2y.isna().astype(int)

missing.resample("D").sum().plot(kind="bar", figsize=(12,3))
plt.title("Number of Missing Hours per Day")
plt.ylabel("Count")
plt.show() # plotting the missing hours per day
series_2y_clean = series_2y.interpolate(limit_direction="both")


# Plot the raw data from the subset
series_2y_clean.plot()
plt.title("Hourly Electricity Load – AEP (2015–2016)")
plt.ylabel("MW")
plt.xlabel("Time")
plt.show()

# Plot the hourly load for the month January 2015 for zoomed in view 
series_2y_clean["2015-01-01":"2015-01-31"].plot()
plt.title("Hourly Load – January 2015 (Zoomed)")
plt.ylabel("MW")
plt.show()

# Plot the daily & weekly patterns in relation to time
daily_profile = series_2y_clean.groupby(series_2y_clean.index.hour).mean()
daily_profile.plot()
plt.title("Average Daily Load Profile (2015–2016)")
plt.xlabel("Hour of Day")
plt.ylabel("Average MW")
plt.show()

# Average weekly pattern (days of week)
weekly_profile = series_2y_clean.groupby(series_2y_clean.index.dayofweek).mean()
weekly_profile.index = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
weekly_profile.plot(kind="bar")
plt.title("Average Load by Day of Week (2015–2016)")
plt.ylabel("Average MW")
plt.show()

# Monthly average load 
monthly_mean = series_2y_clean.resample("M").mean()
monthly_mean.plot(marker="o")
plt.title("Monthly Average Load (2015–2016)")
plt.ylabel("MW")
plt.show()

# Distribution (histogram) & outlier detection
sns.histplot(series_2y_clean, bins=40, kde=True)
plt.title("Distribution of Hourly Load (2015–2016)")
plt.xlabel("MW")
plt.show()

# Boxplot distribution by month 
tmp = series_2y_clean.to_frame(name="load")
tmp["month"] = tmp.index.month
plt.figure(figsize=(12,5))
sns.boxplot(data=tmp, x="month", y="load")
plt.title("Load Distribution by Month")
plt.xlabel("Month")
plt.ylabel("MW")
plt.show()

# Seasonality and Trend – STL decomposition

# Average daily load
daily_load = series_2y_clean.resample("D").mean()
daily_load = daily_load.interpolate()

# Weekly seasonality
stl = STL(daily_load, period=7)
result = stl.fit()
fig = result.plot()
fig.suptitle("STL Decomposition – Daily AEP Load (2015–2016)", y=1.02)
plt.show()

