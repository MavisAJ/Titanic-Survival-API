###Imports
from fastapi import FastAPI
from sklearn import preprocessing
import pickle, uvicorn,os
from pydantic import BaseModel
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import accuracy_score
import pandas as pd 
import numpy as np
from typing import Union
##
#########################

##Variable of environment

DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH ,'assets')
ml_components = os.path.join(ASSETSDIRPATH ,'api_with_ml_pipline.pkl')
print(f"{'*'*10} Config {'*'*10}\n INFO: DIRPATH = {DIRPATH}\n INFO:ASSETSDIRPATH ={ASSETSDIRPATH}")
## API  Basic Config
app = FastAPI(title = 'Titanic Survival API',
               version = '0.0.1',
               desccription = "Prediction of Surival on the titanic Ship" )
## Loading of assets 
with open(ml_components,'rb')as f:
    loaded_items = pickle.load(f)
print("INFO: Loaded assets :",loaded_items)

ml_processsor = loaded_items['pipeline_for_preprocessing']
ml_model = loaded_items['model']
### Base Model
class ModelInput(BaseModel):

    SibSp: float
    PassengerId :float
    Pclass: float
    Parch :float 
    Fare :  float
    Age: float
    Sex_female: float
    Embarked_C : float   
    Embarked_Q : float   
def processing_FE(dataset,  imputer=None, FE=None ): 
    if imputer is not None:
        output_dataset = imputer.transform(dataset)
    else:
        output_dataset = dataset.copy()
    if FE is not None:
        output_dataset = FE.transform(output_dataset)

    return output_dataset


def make_prediction(
    SibSp,PassengerId,Pclass, Parch,Fare, Age,Sex_female, Embarked_C,Embarked_Q
    ):
    df = pd.DataFrame([[SibSp,PassengerId,Pclass, Parch,Fare, Age,Sex_female, Embarked_C,Embarked_Q]],
    columns = ['SibSp','PassengerId','Pclass','Parch','Fare' ,'Age','Sex_female', ' Embarked_C',Embarked_Q ],
    )
    X = processing_FE(dataset = df, FE = None)
    model_output = ml_model.predict(X).tolist()

    return model_output

### Endpoints
@app.post('/titanic survivor')
async def predict(input: ModelInput):
    """__descr___

    ____details___
    """
    output_pred = make_prediction(SibSp = input.SibSp,
        PassengerId = input.PassengerId,
        Pclass = input.Pclass,
        Parch  = input.Parch,
        Fare = input.Fare,
        Age = input.Age,
        Sex_female = input.Sex_female, 
        Embarked_C = input.Embarked_C,
        Embarked_Q = input.Embarked_Q)
    
# Labelling Model output   
    if output_pred == 0:
        output_pred = "No,the person didn't survive"
    else:
        output_pred = "Yes,the person survived"
    return{
    "prediction": output_pred,
    "input": input}
#######################################
@app.get("/")
async def root():
    return {"message": "Hello Mavis"}

######Execution

if __name__ =="__main__":
    uvicorn('api :app',reload = True, )