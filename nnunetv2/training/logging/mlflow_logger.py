from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
import os
import mlflow

class MLflowLogger(object):
    def __init__(self, nnunet_logger, tracking_uri, experiment_name):
        assert isinstance(nnunet_logger, nnUNetLogger), "nnunet_logger must be an instance of nnUNetLogger"
        self.nnunet_logger = nnunet_logger
        mlflow.set_tracking_uri(uri=tracking_uri)
        mlflow.set_experiment(experiment_name)


    def log(self, key, value, epoch: int):
        self.nnunet_logger(key, value, epoch)
        self.check_mlflow_run()
        try: 
            mlflow.log_metric(key, value, step=epoch)
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


    def plot_progress_png(self, output_folder):
        self.nnunet_logger(output_folder)


    def get_checkpoint(self):
        return self.nnunet_logger.get_checkpoint()


    def load_checkpoint(self, checkpoint: dict):
        self.nnunet_logger.load_checkpoint(self, checkpoint)
