from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
import os
import mlflow

class MLflowLogger(nnUNetLogger):

    def __init__(self, tracking_uri, experiment_name, verbose: bool = False):
        mlflow.set_tracking_uri(uri=tracking_uri)
        mlflow.set_experiment(experiment_name)
        super().__init__(verbose=verbose)


    def log(self, key, value, epoch: int):
        super().log(key, value, epoch)
        self.check_mlflow_run()
        try: 
            if isinstance(value, list):
                for i, v in enumerate(value):
                        mlflow.log_metric(f"{key}_{i}", v, step=epoch)
            else:
                mlflow.log_metric(key, float(value), step=epoch)
        except Exception as e:
            print(f"MLflowLogger: Failed to log metric {key} with value {value} at epoch {epoch}: {e}")        


    def log_artifact(self, filename, artifact_path):
        self.check_mlflow_run()
        try:
            mlflow.log_artifact(filename, artifact_path=artifact_path)
        except Exception as e:
            print(f"MLflowFogger: Failed to log artifact {filename} to MLflow: {e}")


    def check_mlflow_run(self):
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Please start a run before logging metrics.")   
