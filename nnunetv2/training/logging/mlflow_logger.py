from nnunetv2.training.logging.nnunet_logger import LocalLogger
import nnunetv2.training.logging.mlflow_nnunet_model as mlflow_nnunet_model
import types
import mlflow
import numpy as np
import cloudpickle


class MLflowLogger:
    """MLflow logger that plugs into nnUNet's MetaLogger.
    
    Implements the logger interface: update_config(), log(), log_summary().
    """

    def update_config(self, config: dict):
        """Called by MetaLogger with plans, configuration, fold, dataset, hparas."""
        self.check_mlflow_run()
        try:
            # Flatten and stringify for MLflow params
            flat = {}
            for k, v in config.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        flat[f"{k}.{k2}"] = str(v2)[:250]
                else:
                    flat[k] = str(v)[:250]
            mlflow.log_params(flat)
        except Exception as e:
            print(f"MLflowLogger: Failed to log parameters to MLflow: {e}")

    def log(self, key, value, step: int):
        """Called by MetaLogger. Note: MetaLogger already fans out lists,
        so value here is always a scalar and key is e.g. 'mean_fg_dice/class_1'."""
        self.check_mlflow_run()
        try:
            mlflow.log_metric(key, float(value), step=step)
        except Exception as e:
            print(f"MLflowLogger: Failed to log metric {key} with value {value} at step {step}: {e}")

    def log_summary(self, key, value):
        """Called by MetaLogger for one-off values like final validation dice."""
        self.check_mlflow_run()
        try:
            mlflow.log_metric(key, float(value))
        except Exception as e:
            print(f"MLflowLogger: Failed to log summary {key}: {e}")

    # --- Your custom methods (called from your trainer, not from MetaLogger) ---

    def log_artifact(self, filename, artifact_path):
        self.check_mlflow_run()
        try:
            return mlflow.log_artifact(filename, artifact_path=artifact_path)
        except Exception as e:
            print(f"MLflowLogger: Failed to log {filename} to artifact path {artifact_path}: {e}")

    def log_model(self, name, checkpoint_file, plans_file, dataset_file, *args, **kwargs):
        self.check_mlflow_run()
        try:
            cloudpickle.register_pickle_by_value(mlflow_nnunet_model)
            model_class = mlflow_nnunet_model.nnUNetModelForFileInput
            return mlflow.pyfunc.log_model(
                artifact_path=name,
                python_model=model_class(),
                artifacts={
                    "checkpoint": checkpoint_file,
                    "plans": plans_file,
                    "dataset": dataset_file,
                },
                signature=model_class.model_signature(),
                *args,
                **kwargs,
            )
        except Exception as e:
            print(f"MLflowLogger: Failed to log model to MLflow: {e}")

    def check_mlflow_run(self):
        if mlflow.active_run() is None:
            raise RuntimeError("No active MLflow run. Please start a run before logging metrics.")