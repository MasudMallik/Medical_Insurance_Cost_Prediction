#RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split,KFold,GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import mlflow

df=pd.read_csv("D:/powerbi/medical_insurance.csv")
print(df.columns)
x=df[['age', 'sex', 'bmi', 'children', 'smoker']]
y=df["charges"]

x_num=x.select_dtypes(include="number")
x_char=x.select_dtypes(exclude="number")

preprocess=ColumnTransformer(
    transformers=[
        ("numeric value",StandardScaler(),x_num.columns),
        ("string value",OrdinalEncoder(),x_char.columns)
    ],
    remainder="passthrough"
)
pipeline=Pipeline(
    steps=[
        ("preprocess",preprocess),
        ("model",RandomForestRegressor(random_state=42))
    ]
)
param={
    "model__max_depth":[10,20,30],
    "model__criterion":["absolute_error","squared_error","poisson"],
    "model__n_estimators":[20,30,50]
}

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
cross=KFold(n_splits=5,shuffle=True,random_state=42)
# print(cross_val_score(pipeline,x_train,y_train,cv=cross,scoring="r2"))

model=GridSearchCV(
    estimator=pipeline,
    param_grid=param,
    cv=cross,
    scoring="r2"
)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)


mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Medical_Insurance_cost_prediction")
with mlflow.start_run(run_name="RandomForestRegressor"):
    mlflow.log_params(model.best_params_)
    mlflow.log_metric("R2-score",r2)
    mlflow.log_metric("MAE",mae)
    mlflow.log_metric("MSE",mse)
    mlflow.sklearn.log_model(model,"RandomForestRegressor_model")