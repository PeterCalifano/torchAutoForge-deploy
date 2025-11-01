from AutoForgeDeployPy.setup.model_formats import DeployModelFormat
import onnx, torch
import logging, os, sys
import numpy as np

class DeploymentModelConverter():
    """
    DeploymentModelConverter Helper class for converting models to various formats for deployment.

    _extended_summary_
    """

    def __init__(self, 
                model_or_path: str | onnx.ModelProto, 
                model_name_noext: str,
                model_export_path: str = "deploy_export",
                input_to_output_formats: tuple[DeployModelFormat, DeployModelFormat] = (DeployModelFormat.ONNX, DeployModelFormat.TFLITE)):
        """
        Initialize the DeploymentModelConverter with a model and target format.
        TODO
        """

        # Store options and inputs
        self.model = model
        self.input_to_output_formats = input_to_output_formats
        self.model_input_filepath = model_input_filepath
        self.model_export_path = model_export_path
        self.model_name_noext = model_name_noext

        self.logger = logging.getLogger(name=__name__)

        # Select implementation of convert method based on input and output formats

        def convert(self, sample_input: torch.Tensor | np.ndarray | None = None):
            """
            Convert the model to the target format.

            Parameters
            ----------
            sample_input : torch.Tensor | np.ndarray | None, optional
                Sample input for the model, used for conversion, by default None.
            """
            raise NotImplementedError("This convert method is only a placeholder to dispatch to the actual implementation in the class.")


        def _convert_onnx_to_tflite(self):
            pass

        def _convert_onnx2tf(self):
            """
            _convert_onnx2tf Implementation of convert function using onnx2tf backend, called from command line.

            _extended_summary_
            """
            self.logger.info(f"Conversion process begins: ONNX --> TFlite using onnx-tf backend.")

            # Call the onnx2tf command line tool
            import subprocess

            subprocess.run(["onnx2tf", 
                            "--input", self.model_input_filepath, 
                            "--output", self.model_export_path])
            pass

        def _convert_onnxtf(self, model: str | onnx.ModelProto):
            """
            _convert_onnxtf Implementation of convert function using onnx-tf backend, passing through tensorflow package.

            _extended_summary_
            """

            self.logger.info(f"Conversion process begins: ONNX --> TFlite using onnx-tf backend.")

            # Load ONNX model
            if isinstance(model, str):
                onnx_model = onnx.load(model)

            elif isinstance(model, onnx.ModelProto):
                onnx_model = model
            
            else:
                raise ValueError("Model must be a file path or an ONNX ModelProto instance.")

            import onnx_tf.backend
            import tensorflow as tf

            tf_export_model_path = f"{self.model_export_path}/{self.model_name_noext}_tf_model"

            # Prepare TensorFlow representation from ONNX model
            tensorflow_representation = onnx_tf.backend.prepare(onnx_model)
            tensorflow_representation.export_graph(tf_export_model_path)

            # Export tensorflow model to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_export_model_path)
            self.logger.info(f"Converted TF model saved to: {tf_export_model_path}")

            # Specify conversion options
            # TODO: extend wrapper to allow more options
            converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # No Flex ops
            converter.experimental_disable_per_channel = True
            converter.optimizations = [tf.lite.Optimize.DEFAULT]

            # Convert the model
            tflite_model = converter.convert()

            # Save the TFLite model
            tflite_export_path = f"{self.model_export_path}/{self.model_name_noext}.tflite"
            with open(tflite_export_path, "wb") as f:
                f.write(tflite_model)

            self.logger.info(f"Converted TFLite model saved to: {tflite_export_path}")

        def _convert_torch_to_tflite_edgetorch(self, 
                                               model: str | torch.nn.Module,
                                               sample_input: torch.Tensor | np.ndarray):
            """
            _convert_torch_to_tflite_edgetorch Implementation of convert function using torch2tflite backend, called from command line.

            _extended_summary_
            """
            self.logger.info(f"Conversion process begins: Torch --> TFlite using torch2tflite backend.")

            import ai_edge_torch
            from pyTorchAutoForge.api.torch import LoadModel
            # TODO 

            if isinstance(model, str):
                # Load the model from file
                model = LoadModel()

            edgetorch_export_model_path = f"{self.model_export_path}/{self.model_name_noext}_edgetorch_model.tflite"

            # Call export method from ai_edge_torch package
            edge_model = ai_edge_torch.convert(model, sample_input)
            edge_model.export(edgetorch_export_model_path)
