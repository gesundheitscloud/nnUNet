from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.logging.mlflow_nnunet_model import nnUNetModel
import types
import mlflow
import numpy as np


class MLflowLogger(nnUNetLogger):

    def __init__(self, verbose: bool = False):
        super().__init__(verbose=verbose)


    def log(self, key, value, epoch: int):
        super().log(key, value, epoch)
        self.check_mlflow_run()
        try: 
            if isinstance(value, list):
                for i, v in enumerate(value):
                        mlflow.log_metric(f"{key}_{i}", v, step=epoch)
            else:
                return mlflow.log_metric(key, float(value), step=epoch)
        except Exception as e:
            print(f"MLflowLogger: Failed to log metric {key} with value {value} at epoch {epoch}: {e}")        


    def log_artifact(self, filename, artifact_path):
        self.check_mlflow_run()
        try:
            return mlflow.log_artifact(filename, artifact_path=artifact_path)
        except Exception as e:
            print(f"MLflowFogger: Failed to log {filename} to MLflow in artifact path {artifact_path}: {e}")


    def log_params(self, params):
        self.check_mlflow_run()
        try: 
            return mlflow.log_params(params)
        except Exception as e:
            print(f"MLflowLogger: Failed to log parameters to MLflow: {e}")


    def log_model(self, name, checkpoint_file, plans_file, dataset_file, *args, **kwargs):
        self.check_mlflow_run()
        try:
            nnUNetModelDyn = types.new_class("nnUNetModelDyn" (nnUNetModel,)),
            return mlflow.pyfunc.log_model(
                name = name,
                python_model = nnUNetModelDyn(),
                artifacts={
                    "checkpoint": checkpoint_file,
                    "plans": plans_file,
                    "dataset": dataset_file
                },
                signature = nnUNetModel.model_signature(),
                input_example = nnUNetModel.input_example(),                
                *args,
                **kwargs,
            )
        except Exception as e:
            print(f"MLflowLogger: Failed to log model to MLflow: {e}")


    def check_mlflow_run(self):
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Please start a run before logging metrics.")   
