# This model is created using LinearRegression model
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
import mlflow

df=pd.read_csv("D:\powerbi\medical_insurance.csv")

x=df[['age', 'sex', 'bmi', 'children', 'smoker', 'region']]
y=df["charges"]

x_numeric=x.select_dtypes(include="number")
x_string=x.select_dtypes(exclude="number")

preprocess=ColumnTransformer(
    transformers=[
        ("numeric values",StandardScaler(),x_numeric.columns),
        ("string value",OrdinalEncoder(),x_string.columns)
    ],
    remainder="passthrough"
)

pipeline=Pipeline(
    steps=[
        ("preprocess",preprocess),
        ("model",LinearRegression())
    ]
)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
cross=KFold(n_splits=5,shuffle=True,random_state=42)
print(cross_val_score(pipeline,x_train,y_train,cv=cross,scoring="r2"))

pipeline.fit(x_train,y_train)
y_pred=pipeline.predict(x_test)
r2=(r2_score(y_test,y_pred))
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Medical_Insurance_cost_prediction")
with mlflow.start_run(run_name="LinearRegression"):
    mlflow.log_metric("R2 score",r2)
    mlflow.log_metric("MSE",mse)
    mlflow.log_metric("MAE",mae)
    mlflow.sklearn.log_model(pipeline,"LinearRegression_Model")

