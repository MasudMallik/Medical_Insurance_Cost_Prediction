#Decision tree

import pandas as pd
from sklearn.model_selection import train_test_split,KFold,cross_val_score,GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
import mlflow

df=pd.read_csv("D:/powerbi/medical_insurance.csv")
print(df.columns)
print(len(df))
x=df[['age', 'sex', 'bmi', 'children', 'smoker']]
y=df["charges"]
x_str=x.select_dtypes(exclude="number")
x_int=x.select_dtypes(include="number")

preprocess=ColumnTransformer(
    transformers=[
        ("numeric value",StandardScaler(),x_int.columns),
        ("string value",OrdinalEncoder(),x_str.columns),
    ],
    remainder="passthrough"
)

pipeline=Pipeline(
    steps=[
        ("preprocessing",preprocess),
        ("model",DecisionTreeRegressor())
    ]
)
param={
    "model__max_depth":[10,20,30],
    "model__criterion":["absolute_error","squared_error"]
}

cross=KFold(n_splits=5,shuffle=True,random_state=42)
model=GridSearchCV(
    estimator=pipeline,
    param_grid=param,
    cv=cross,
    scoring="r2"
)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
model.fit(x_train,y_train)
best_param=model.best_params_
y_pred=model.predict(x_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)

mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("Medical_Insurance_cost_prediction")
with mlflow.start_run(run_name="Decision_Tree_Regression"):
    mlflow.log_params(model.best_params_)
    mlflow.log_metric("r2-score",r2)
    mlflow.log_metric("MSE",mse)
    mlflow.log_metric("MAE",mae)
    mlflow.set_tag("Model name","DecisionTreeRegressor")
    mlflow.sklearn.log_model(model,"DecisionTreeRegressor_model")
mlflow.register_model("run:/m-98c6772e6ae245ab90dbc10e8e5aac54/DecisionTreeRegressor_model","Medical_insurance")