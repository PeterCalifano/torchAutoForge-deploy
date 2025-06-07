# Script that converts a PyTorch model to ONNX format and compares outputs.
import os, sys
import numpy as np
import torch
import onnxruntime as ort
import scipy.io as sio  
from pyTorchAutoForge.utils import GetDeviceMulti
from pyTorchAutoForge.api.torch import LoadModel, SaveModel, AutoForgeModuleSaveMode
from pyTorchAutoForge.api.onnx import ModelHandlerONNx

# ANSI color codes
BLUE = '\033[94m'
RESET = '\033[0m'

sys.path.append(os.path.join(os.getenv("HOME"), "devDir//nav-frontend/.experimental/neuralCOB/"))

import NeuralCOB_module

def save_test_data_to_mat(test_input, onnx_output, test_name, output_dir="test_data"):
    """
    Save test input and ONNX output to .mat files for MATLAB comparison
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert PyTorch tensor to numpy
    if torch.is_tensor(test_input):
        input_np = test_input.cpu().detach().numpy()
    else:
        input_np = test_input
    
    # ONNX output is already numpy
    output_np = onnx_output[0] if isinstance(onnx_output, list) else onnx_output
    
    # Create data dictionary for .mat file
    mat_data = {
        'test_input': input_np,
        'onnx_output': output_np,
        'test_name': test_name,
        'input_shape': input_np.shape,
        'output_shape': output_np.shape
    }
    
    # Save to .mat file
    mat_filename = os.path.join(output_dir, f"python_onnx_{test_name.lower().replace(' ', '_')}.mat")
    sio.savemat(mat_filename, mat_data)
    
    print(f"✓ Saved test data to: {mat_filename}")
    return mat_filename

def save_all_results_to_mat(all_results, output_dir="test_data"):
    """
    Save all test results to a single .mat file
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data structure for MATLAB
    matlab_data = {
        'test_count': len(all_results),
        'test_names': [result['name'] for result in all_results],
    }
    
    # Add each test case data
    for i, result in enumerate(all_results):
        prefix = f"test_{i+1}"
        matlab_data[f"{prefix}_name"] = result['name']
        matlab_data[f"{prefix}_input"] = result['input']
        matlab_data[f"{prefix}_onnx_output"] = result['onnx_output']
        matlab_data[f"{prefix}_pytorch_output"] = result['pytorch_output']
        matlab_data[f"{prefix}_max_diff"] = result['max_diff']
        matlab_data[f"{prefix}_mean_diff"] = result['mean_diff']
        matlab_data[f"{prefix}_passed"] = result['passed']
    
    # Save comprehensive results
    comprehensive_filename = os.path.join(output_dir, "python_onnx_all_results.mat")
    sio.savemat(comprehensive_filename, matlab_data)
    
    print(f"✓ Saved comprehensive results to: {comprehensive_filename}")
    return comprehensive_filename

def compare_outputs(pytorch_output, onnx_output, tolerance=1e-5):
    """
    Compare PyTorch and ONNX outputs
    """
    # Convert PyTorch tensor to numpy if needed
    if torch.is_tensor(pytorch_output):
        pytorch_np = pytorch_output.cpu().detach().numpy()
    else:
        pytorch_np = pytorch_output
    
    # ONNX output is already numpy array
    onnx_np = onnx_output[0] if isinstance(onnx_output, list) else onnx_output
    
    print(f"PyTorch output shape: {pytorch_np.shape}")
    print(f"ONNX output shape: {onnx_np.shape}")
    print(f"PyTorch output sample: {pytorch_np.flatten()[:5]}")
    print(f"ONNX output sample: {onnx_np.flatten()[:5]}")
    
    # Calculate differences
    max_diff = np.max(np.abs(pytorch_np - onnx_np))
    mean_diff = np.mean(np.abs(pytorch_np - onnx_np))
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    
    # Check if outputs are close
    is_close = np.allclose(pytorch_np, onnx_np, atol=tolerance, rtol=tolerance)
    
    if is_close:
        print(f"✓ Outputs match within tolerance ({tolerance})")
    else:
        print(f"✗ Outputs differ beyond tolerance ({tolerance})")
    
    return is_close, max_diff, mean_diff, pytorch_np, onnx_np

def test_inference(model, onnx_session, test_input, device, test_name):
    """
    Run inference on both PyTorch and ONNX models and compare outputs
    """
    print(f"\n{BLUE}" + "="*50)
    print("TESTING INFERENCE")
    print("="*50 + f"{RESET}")
    
    # PyTorch inference
    model.eval()
    with torch.no_grad():
        pytorch_output = model(test_input)
    
    print(f"PyTorch inference completed")
    
    # ONNX inference
    # Convert input to numpy and move to CPU
    test_input_np = test_input.cpu().numpy()
    
    # Get input name from ONNX model
    input_name = onnx_session.get_inputs()[0].name
    onnx_output = onnx_session.run(None, {input_name: test_input_np})
    
    print(f"ONNX inference completed")
    
    # Compare outputs
    is_close, max_diff, mean_diff, pytorch_np, onnx_np = compare_outputs(pytorch_output, onnx_output)
    
    # Save test data to .mat file
    save_test_data_to_mat(test_input, onnx_output, test_name)
    
    return is_close, max_diff, mean_diff, pytorch_np, onnx_np

def main():
    try:
        # Path management
        root_path = os.path.join(os.getenv("HOME"), "devDir//torchAutoForge-deploy/python/scripts/")
        os.chdir(root_path)

        checkpoints_path = os.path.join(root_path, "checkpoints/")
        onnx_path = os.path.join(root_path, "onnx_models/")
        
        # Ensure output directory exists
        os.makedirs(onnx_path, exist_ok=True)

        state_name = "ambitious_calf_40_epoch_2213"
        nn_model_path = checkpoints_path + state_name + ".pth"
        nn_onnx_output_path = onnx_path + state_name  # ModelHandlerONNx adds .onnx automatically

        print(f"Loading model from: {nn_model_path}")
        print(f"ONNX output path: {nn_onnx_output_path}")

        # Load model
        device = GetDeviceMulti()
        print(f"Using device: {device}")
        
        model = LoadModel(None, nn_model_path).to(device)
        model.eval()  # Set to evaluation mode
        print(model)

        # Get first layer size
        input_layer_size = 12

        # Check if ONNX file already exists and delete it
        actual_onnx_path = nn_onnx_output_path + ".onnx"
        if os.path.exists(actual_onnx_path):
            print(f"Existing ONNX file found: {actual_onnx_path}")
            os.remove(actual_onnx_path)
            print(f"✓ Deleted existing ONNX file")

        print(f"\n{BLUE}" + "="*50)
        print("CONVERTING TO ONNX")
        print("="*50 + f"{RESET}")
        
        # Create dummy input for export
        dummy_input = torch.randn(1, input_layer_size).to(device)
        
        # Export ONNX
        onnx_handler = ModelHandlerONNx(model=model, 
                                        dummy_input_sample=dummy_input,
                                        opset_version=11,
                                        onnx_export_path=nn_onnx_output_path,
                                        generate_report=False)
        
        onnx_handler.torch_export()
        
        print(f"✓ Model successfully exported to: {actual_onnx_path}")

        # Verify ONNX file exists
        if not os.path.exists(actual_onnx_path):
            raise FileNotFoundError(f"ONNX file was not created: {actual_onnx_path}")

        print(f"\n{BLUE}" + "="*50)
        print("LOADING ONNX MODEL")
        print("="*50 + f"{RESET}")
        
        # Load ONNX model for inference
        onnx_session = ort.InferenceSession(actual_onnx_path)
        print(f"✓ ONNX model loaded successfully")
        
        # Print model info
        print(f"ONNX model inputs: {[input.name for input in onnx_session.get_inputs()]}")
        print(f"ONNX model outputs: {[output.name for output in onnx_session.get_outputs()]}")

        # Create test inputs for comparison
        print(f"\n{BLUE}" + "="*50)
        print("CREATING TEST INPUTS")
        print("="*50 + f"{RESET}")
        
        # Create multiple test cases
        test_cases = [
            torch.randn(1, input_layer_size).to(device),  # Random input
            torch.zeros(1, input_layer_size).to(device),  # Zero input
            torch.ones(1, input_layer_size).to(device),   # Ones input
            torch.randn(1, input_layer_size).to(device) * 10,  # Larger scale input
        ]
        
        test_names = ["Random", "Zeros", "Ones", "Large Scale"]
        
        all_passed = True
        all_results = []  # Store all test results for comprehensive .mat export
        
        for i, (test_input, test_name) in enumerate(zip(test_cases, test_names)):
            print(f"\nTest Case {i+1}: {test_name}")
            print("-" * 30)
            
            is_close, max_diff, mean_diff, pytorch_np, onnx_np = test_inference(
                model, onnx_session, test_input, device, test_name
            )
            
            # Store results
            result_data = {
                'name': test_name,
                'input': test_input.cpu().detach().numpy(),
                'pytorch_output': pytorch_np,
                'onnx_output': onnx_np,
                'max_diff': max_diff,
                'mean_diff': mean_diff,
                'passed': is_close
            }
            all_results.append(result_data)
            
            if not is_close:
                all_passed = False
        
        # Save comprehensive results to .mat file
        print(f"\n{BLUE}" + "="*50)
        print("SAVING RESULTS")
        print("="*50 + f"{RESET}")
        
        save_all_results_to_mat(all_results)
        
        print(f"\n{BLUE}" + "="*50)
        print("FINAL RESULTS")
        print("="*50 + f"{RESET}")
        
        if all_passed:
            print("✓ All test cases passed! PyTorch and ONNX models produce equivalent outputs.")
        else:
            print("✗ Some test cases failed. There may be differences between PyTorch and ONNX outputs.")
        
        print(f"\nMAT files created in 'test_data/' directory:")
        print(f"  - Individual test files: python_onnx_<test_name>.mat")
        print(f"  - Comprehensive results: python_onnx_all_results.mat")
        print(f"\nUse these files to compare with MATLAB ONNX results.")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()