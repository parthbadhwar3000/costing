import os
import sys
from dataclasses import dataclass
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split data into training and test data")

            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "RandomForest":RandomForestRegressor(n_estimators=200,min_samples_split=2,max_features=8,max_depth=None)
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            model_score=max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model=models[best_model_name]

            if model_score<0.6:
                raise CustomException("No best Model found")
            logging.info(f"Best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)
            score=r2_score(y_test,predicted)
            return score

        except Exception as e:
            raise CustomException(e,sys)