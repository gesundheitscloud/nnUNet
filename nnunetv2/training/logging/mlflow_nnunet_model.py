import os
from typing import Any, Dict, List
import mlflow
import numpy as np
from pydantic import BaseModel
import torch

from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p

from nnunetv2 import __path__ as nnunetv2_path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels


class nnUNetModel(mlflow.pyfunc.PythonModel):

    def __getstate__(self):
        # Copy the instance dictionary
        state = self.__dict__.copy()
        # Remove the unpicklable attribute
        if "predictor" in state:
            del state["predictor"]
        return state


    def __setstate__(self, state):
        # Restore instance attributes (without predictor)
        self.__dict__.update(state)
        # Optionally, reinitialize predictor to None or rebuild it
        self.predictor = None


    def load_context(self, context):
        self.configure(context.artifacts["checkpoint"], context.artifacts["plans"], context.artifacts["dataset"])


    def configure(self, checkpoint_file, plans_file, dataset_file):
        if os.getenv('DEVICE') is not None:
            device = torch.device(os.getenv('DEVICE'))
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # mps not supported in nnunetv2

        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        dataset_json = load_json(dataset_file)
        plans_manager = PlansManager(load_json(plans_file))

        checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'), weights_only=False)
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']
        inference_allowed_mirroring_axes = checkpoint['inference_allowed_mirroring_axes'] if \
            'inference_allowed_mirroring_axes' in checkpoint.keys() else None

        configuration_manager = plans_manager.get_configuration(configuration_name)        
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnunetv2_path[0], "training", "nnUNetTrainer"),
                                                    trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer.')
        
        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        )
        parameters = []
        parameters.append(checkpoint['network_weights'])
        self.predictor.manual_initialization(network, plans_manager, configuration_manager, parameters, 
                                             dataset_json, trainer_name, inference_allowed_mirroring_axes)


    def predict(self, model_input):
        imgs = [img for img, _ in model_input]
        props = [prop for _, prop in model_input]
        prediction = self.predictor.predict_from_list_of_npy_arrays(
            image_or_list_of_images = imgs, 
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = None, 
            properties_or_list_of_properties = props, 
            truncated_ofname = None, 
            num_processes=2, 
            save_probabilities=False, 
            num_processes_segmentation_export=2,
            )
        return prediction
    