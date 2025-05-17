import os
import sys


import numpy as np
import pandas as pd
import dill
import pickle

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    """
    Save an object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object to be saved.
    """
    import pickle
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb")as file_obj:
            dill.dump(obj,file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for name, model in models.items():
            try:
                param_grid = param.get(name, {})  # Safe lookup
                gs = GridSearchCV(model, param_grid, cv=3)
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                model.fit(X_train, y_train)

                y_test_pred = model.predict(X_test)
                score = r2_score(y_test, y_test_pred)

                report[name] = score

            except Exception as inner_e:
                report[name] = 0
                print(f"[WARN] Model {name} failed: {inner_e}")

        return report

    except Exception as e:
        raise CustomException(e, sys)



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
