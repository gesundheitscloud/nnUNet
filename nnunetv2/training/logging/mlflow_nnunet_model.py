import os
import mlflow
from mlflow.models import ModelSignature
from mlflow.types import TensorSpec, ColSpec, Schema
import numpy as np
import pandas as pd
import torch

from batchgenerators.utilities.file_and_folder_operations import join, load_json

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels

import base64
import tempfile


class nnUNetModelAbstract(mlflow.pyfunc.PythonModel):

    def __init__(self):
        super().__init__()

    # getstate and setstate are used to control what is pickled and unpickled.
    # We don't want to pickle the predictor object but rebuild it from the artifacts 
    # by configure() instead.
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
            allow_tqdm=False
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
        # Import nnunetv2 path here get the correct path after un-pickling a nnUNetModel instance in a different environment
        from nnunetv2 import __path__ as nnunetv2_path
        nnunet_trainer_path = join(nnunetv2_path[0], "training", "nnUNetTrainer")
        trainer_class = recursive_find_python_class(nnunet_trainer_path, trainer_name, 'nnunetv2.training.nnUNetTrainer')
        if trainer_class is None:
            raise RuntimeError(f'Unable to locate trainer class {trainer_name} in nnunetv2.training.nnUNetTrainer using path {nnunet_trainer_path}.')
        
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



class nnUNetModeForTensorInput(nnUNetModelAbstract):
    # model_input is a dict with two tensors of form {"image": np.ndarray, "spacing": np.ndarray}
    # where image is a 4D numpy array (C, X, Y, Z) (float32) and spacing is 3D numpy array (X, Y, Z) (float32)
    # predict() returns a 3D numpy array (X, Y, Z) (float32) of the predicted segmentation
    # See nnUNetModelFileInput.predict() for how to get model_input from file (image and props)
    # Returned prediction can be written to file with SimpleITKIO().write_seg(returned_pred, output_file, props)
    def predict(self, model_input, params = None):
        image = model_input["image"]
        properties = {"spacing": model_input["spacing"]}
        prediction = self.predictor.predict_from_list_of_npy_arrays(
            image_or_list_of_images = image, 
            segs_from_prev_stage_or_list_of_segs_from_prev_stage = None, 
            properties_or_list_of_properties = properties, 
            truncated_ofname = None, 
            num_processes = 1, 
            save_probabilities=False, 
            num_processes_segmentation_export = 1,
            )
        return prediction[0]


    @staticmethod
    def model_signature():
        input_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, -1, -1, -1), "image"),
                TensorSpec(np.dtype(np.float32), (3,), "spacing"),
            ])
        output_schema = Schema([
                TensorSpec(np.dtype(np.float32), (-1, -1, -1))
            ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)


    # @staticmethod
    # def input_example():
    #     img = np.random.rand(1, 16, 64, 64).astype(np.float32)
    #     spacing = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    #     return {"image": img, "spacing": spacing}


class nnUNetModelForFileInput(nnUNetModelAbstract):

    supported_file_endings = [
        '.nii.gz',
    ]

    def predict(self, model_input, params=None):

        input_filename = None
        input_base64_data = None

        if isinstance(model_input, pd.DataFrame):
            if len(model_input) != 1:
                raise RuntimeError("Input DataFrame must have exactly one row")
            row = model_input.iloc[0]
            input_filename = row["filename"]
            input_base64_data = row["base64_data"]

        if not input_filename or not input_base64_data:
            raise RuntimeError("Both 'filename' and 'base64_data' must be provided")

        predict_file_index = 0

        # Remove suffix from input file name
        input_filename_stem = ""
        is_supported_filetype = False

        for ending in self.supported_file_endings:
            if input_filename.lower().endswith(ending.lower()):
                is_supported_filetype = True
                input_filename_stem = input_filename[: -len(ending)]
                break

        if len(input_filename_stem) == 0:
            # Set input_filename_stem to image- appending a random string
            input_filename_stem = f"image-{predict_file_index}"

        if not is_supported_filetype:
            raise RuntimeError(
                f"filename must end with one of the supported file endings: "
                f"{self.supported_file_endings}"
            )

        # Decode base64-encoded file contents
        if not isinstance(input_base64_data, str):
            raise RuntimeError("'data' must be a base64-encoded string")

        try:
            input_byte_data = base64.b64decode(input_base64_data)
        except Exception as e:
            raise RuntimeError(f"Failed to base64-decode data: {e}")

        # Save to temporary file and read with SimpleITK
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_filepath = os.path.join(tmpdirname, input_filename)
            with open(input_filepath, "wb") as f:
                f.write(input_byte_data)

            # Check that ouput file exists and is not empty
            if not os.path.isfile(input_filepath) or os.path.getsize(input_filepath) == 0:
                raise RuntimeError("Input image file was not created or is empty")

            # This is how we would convert the image to npy to predict with predict_from_list_of_npy_arrays
            # This way we control reading/writing format of files oursevles. However we will use 
            # predict_from_files_sequential below instead and assume that the registered reader/writer
            # supports the file formats we define above.
            # img, props = SimpleITKIO().read_images([input_file])
            # img = np.array(img, dtype=np.float32)
            # prediction = self.predictor.predict_from_list_of_npy_arrays(
            #     image_or_list_of_images = img, 
            #     segs_from_prev_stage_or_list_of_segs_from_prev_stage = None, 
            #     properties_or_list_of_properties = props, 
            #     truncated_ofname = None, 
            #     num_processes = 1, 
            #     save_probabilities=False, 
            #     num_processes_segmentation_export = 1,
            #     )
            # segmentation = prediction[0]

            output_filename = input_filename_stem + "_seg.nii.gz"
            output_filepath = os.path.join(tmpdirname, output_filename)

            print(f"Predicting segmentation for input file {input_filepath}, writing to {output_filepath}")
            self.predictor.predict_from_files_sequential([[input_filepath]], [output_filepath])

            # Check that ouput file exists and is not empty
            if not os.path.isfile(output_filepath) or os.path.getsize(output_filepath) == 0:
                raise RuntimeError("Output segmentation file was not created or is empty")

            output_base64_data = None
            # Read and base64 encode the content of output file
            with open(output_filepath, "rb") as f:
                seg_data = f.read()
                output_base64_data = base64.b64encode(seg_data).decode('utf-8')

            if output_base64_data is None:
                raise RuntimeError("Failed to read or encode the output segmentation file")

            return {"filename": output_filename, "base64_data": output_base64_data}


    @staticmethod
    def model_signature():
        input_schema = Schema([
            ColSpec("string", "filename"),
            ColSpec("string", "base64_data"),
        ])
        output_schema = Schema([
            ColSpec("string", "filename"),
            ColSpec("string", "base64_data"),
        ])
        return ModelSignature(inputs=input_schema, outputs=output_schema)
