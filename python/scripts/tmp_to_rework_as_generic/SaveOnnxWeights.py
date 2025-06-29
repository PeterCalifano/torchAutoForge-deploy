from typing import Literal
import onnx
import numpy as np
import scipy.io as sio
import os
import pathlib
import argparse as argp
from autoforgeDeployPy.utils.matlab_utils import Sanitize_matlab_name

# User-options
onnx_model_path = "onnx_models/ambitious_calf_40_epoch_2213.onnx"
output_file = "test_data/onnx_weights.mat"

# TODO Implement argument parser for usage from cli

# %% Auxiliary functions for export
# TODO Move to utils/matlab_utils.py


# TODO (UM) Move to utils/onnx_utils.py
# TODO extend to same to json as well (can be useful)
def Extract_onnx_weights_to_file(onnx_path : str | pathlib.Path, 
                                output_filename : str | pathlib.Path = "onnx_exported_weights.mat",
                                output_format : Literal['mat', 'json'] = 'json') -> dict[str, np.ndarray | dict]:
    """
    Extract weights from ONNX model and save to .mat file
    """
    print(f"Loading ONNX model: {onnx_path}")
    
    if output_format == 'json':
        raise NotImplementedError('Not implement yet')

    if not (output_format == 'mat' or output_format == 'json'):
        raise ValueError(f'Invalid format: {output_format}. Supported: json, mat.')

    # Load ONNX model
    model = onnx.load(onnx_path)
    print("ONNX model loaded successfully")
    
    # Extract weights (initializers)
    weights_dict = {}
    name_mapping = {}  # Store original -> sanitized name mapping
    
    print(f"Extracting {len(model.graph.initializer)} weight tensors...")
    
    for initializer in model.graph.initializer:

        original_name = initializer.name
        sanitized_name = Sanitize_matlab_name(original_name)
        weight_array = onnx.numpy_helper.to_array(initializer)
        
        weights_dict[sanitized_name] = weight_array
        name_mapping[sanitized_name] = original_name
        
        print(f"  {original_name} -> {sanitized_name}: {weight_array.shape}")
    
    # Add name mapping to the .mat file for reference
    weights_dict['name_mapping'] = name_mapping # type: ignore
    
    # Save to .mat file
    sio.savemat(output_filename, weights_dict)

    print(f"Weights saved to: {output_filename}")
    print(f"Name mapping included for reference")
    
    return weights_dict  # type: ignore

if __name__ == "__main__":
    
    # Extract and save weights
    weights = Extract_onnx_weights_to_file(onnx_model_path, output_file)