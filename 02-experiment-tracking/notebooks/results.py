"""
Answers for 01-intro homework
"""
import pandas as pd
import seaborn as sns

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error


import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("luisek")

# Question 1
df = pd.read_parquet('./data/green_tripdata_2023-01.parquet')

old_len = df.shape[0]
print(old_len)
# (68211, 20)

# Question 2
df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
# Answer
df["duration"].mean()

# Question 3
df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

categorical = ['PULocationID', 'DOLocationID']
numerical = ['trip_distance']

#Answer
print("Number of rows dropped", old_len-df.count())
print("'\%' of Nas",df[categorical].isna().sum()/df[categorical].count())

# Question 3.1 - fill in the missing NAS
df[categorical] = df[categorical].fillna(-1).astype(int)
df[categorical] = df[categorical].astype(str)

# Question  4 - Running the Dictionary transformer (1 hot encoding on each of the categorical features)
train_dicts = df[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts) 
print("Features new shape", X_train.shape)

# Question 5 - Running base line linera regression
y_train = df.duration.values
lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)
mean_squared_error(y_train, y_pred, squared=False)

# Question 6 - last one
categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df
df_val = read_data('./data/fhv_tripdata_2023-02.parquet')
val_dicts = df_val[categorical].to_dict(orient='records')
X_val = dv.transform(val_dicts) 
y_pred = lr.predict(X_val)
y_val = df_val.duration.values
mean_squared_error(y_val, y_pred, squared=False)