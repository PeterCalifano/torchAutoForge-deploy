#!/usr/bin/env python3
"""
Python script to test both PyTorch and ONNX models on filtered data.
This script loads the same filtered dataset and computes comprehensive comparisons
between PyTorch model, ONNX model, and ground truth.
"""

import os
import sys
import numpy as np
import torch
import onnxruntime as ort
import scipy.io as sio
from scipy import stats
import time
import json

# ANSI color codes for better output formatting
BLUE = '\033[94m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

# Add the path for PyTorch model loading (adjust as needed)
sys.path.append(os.path.join(os.getenv("HOME"), "devDir//nav-frontend/.experimental/neuralCOB/"))

try:
    from pyTorchAutoForge.utils import GetDeviceMulti
    from pyTorchAutoForge.api.torch import LoadModel
    PYTORCH_AVAILABLE = True
except ImportError:
    print(f"{YELLOW}Warning: PyTorchAutoForge not available. PyTorch model testing will be disabled.{RESET}")
    PYTORCH_AVAILABLE = False

class ModelTester:
    def __init__(self, pytorch_model_path, onnx_model_path, filtered_data_path):
        """
        Initialize the model tester for both PyTorch and ONNX models.
        
        Args:
            pytorch_model_path (str): Path to the PyTorch model file (.pth)
            onnx_model_path (str): Path to the ONNX model file (.onnx)
            filtered_data_path (str): Path to the filtered data .mat file
        """
        self.pytorch_model_path = pytorch_model_path
        self.onnx_model_path = onnx_model_path
        self.filtered_data_path = filtered_data_path
        
        # Models
        self.pytorch_model = None
        self.onnx_session = None
        self.device = None
        
        # Data
        self.filtered_data = None
        self.ground_truth = None
        
        # Resolution parameters (matching MATLAB code)
        self.resx = 2048
        self.resy = 1536
        
        # Results storage
        self.results = {}
        
    def load_pytorch_model(self):
        """Load the PyTorch model."""
        if not PYTORCH_AVAILABLE:
            print(f"{YELLOW}⚠ PyTorch model loading skipped - PyTorchAutoForge not available{RESET}")
            return False
            
        print(f"{BLUE}{'='*50}")
        print("LOADING PYTORCH MODEL")
        print(f"{'='*50}{RESET}")
        
        if not os.path.exists(self.pytorch_model_path):
            print(f"{RED}PyTorch model not found: {self.pytorch_model_path}{RESET}")
            return False
            
        try:
            self.device = GetDeviceMulti()
            print(f"Using device: {self.device}")
            
            self.pytorch_model = LoadModel(None, self.pytorch_model_path).to(self.device)
            self.pytorch_model.eval()
            
            print(f"✓ PyTorch model loaded successfully from: {self.pytorch_model_path}")
            print(f"Model device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"{RED}Failed to load PyTorch model: {e}{RESET}")
            return False
    
    def load_onnx_model(self):
        """Load the ONNX model."""
        print(f"\n{BLUE}{'='*50}")
        print("LOADING ONNX MODEL")
        print(f"{'='*50}{RESET}")
        
        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"ONNX model not found: {self.onnx_model_path}")
            
        try:
            self.onnx_session = ort.InferenceSession(self.onnx_model_path)
            print(f"✓ ONNX model loaded successfully from: {self.onnx_model_path}")
            
            # Print model info
            inputs = [input.name for input in self.onnx_session.get_inputs()]
            outputs = [output.name for output in self.onnx_session.get_outputs()]
            print(f"Model inputs: {inputs}")
            print(f"Model outputs: {outputs}")
            
            # Get input shape info
            input_shape = self.onnx_session.get_inputs()[0].shape
            print(f"Expected input shape: {input_shape}")
            
            return True
            
        except Exception as e:
            print(f"{RED}Failed to load ONNX model: {e}{RESET}")
            return False
    
    def load_filtered_data(self):
        """Load the filtered data from .mat file."""
        print(f"\n{BLUE}{'='*50}")
        print("LOADING FILTERED DATA")
        print(f"{'='*50}{RESET}")
        
        if not os.path.exists(self.filtered_data_path):
            raise FileNotFoundError(f"Filtered data file not found: {self.filtered_data_path}")
            
        try:
            # Load .mat file
            mat_data = sio.loadmat(self.filtered_data_path)
            
            # Extract filtered matrix and ground truth
            possible_data_keys = [
                'dMixedFeatMatrixAll_filtered',
                'filtered_matrix', 
                'data_filtered',
                'input_data'
            ]
            
            possible_gt_keys = [
                'groundTruth',
                'ground_truth', 
                'gt',
                'targets'
            ]
            
            # Find the correct key for filtered data
            data_key = None
            for key in possible_data_keys:
                if key in mat_data:
                    data_key = key
                    break
                    
            if data_key is None:
                print("Available keys in .mat file:")
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        print(f"  - {key}: shape {mat_data[key].shape}")
                raise KeyError(f"Could not find filtered data. Expected one of: {possible_data_keys}")
                
            # Find the correct key for ground truth
            gt_key = None
            for key in possible_gt_keys:
                if key in mat_data:
                    gt_key = key
                    break
                    
            if gt_key is None:
                print("Warning: Could not find ground truth data. Available keys:")
                for key in mat_data.keys():
                    if not key.startswith('__'):
                        print(f"  - {key}: shape {mat_data[key].shape}")
                print("Continuing without ground truth...")
                self.ground_truth = None
            else:
                self.ground_truth = mat_data[gt_key]
                
            self.filtered_data = mat_data[data_key]
            
            print(f"✓ Filtered data loaded successfully from: {self.filtered_data_path}")
            print(f"Data key used: '{data_key}'")
            print(f"Filtered data shape: {self.filtered_data.shape}")
            
            if self.ground_truth is not None:
                print(f"Ground truth key used: '{gt_key}'")
                print(f"Ground truth shape: {self.ground_truth.shape}")
                
                # MATLAB stores ground truth as (2, N), we might need to transpose
                if self.ground_truth.shape[0] == 2 and self.ground_truth.shape[1] == self.filtered_data.shape[0]:
                    print("Ground truth appears to be in MATLAB format (2, N) - will transpose for Python")
                elif self.ground_truth.shape[1] == 2 and self.ground_truth.shape[0] == self.filtered_data.shape[0]:
                    print("Ground truth appears to be in Python format (N, 2)")
                else:
                    print(f"Warning: Ground truth shape {self.ground_truth.shape} doesn't match expected dimensions")
            
            return True
            
        except Exception as e:
            print(f"{RED}Failed to load filtered data: {e}{RESET}")
            return False
    
    def run_pytorch_inference(self, sample_input):
        """
        Run PyTorch inference on a single sample.
        
        Args:
            sample_input (np.ndarray): Input sample of shape (1, 12) or (12,)
            
        Returns:
            np.ndarray: Model output, or None if PyTorch model not available
        """
        if self.pytorch_model is None:
            return None
            
        # Ensure input is the right shape and convert to tensor
        if sample_input.ndim == 1:
            sample_input = sample_input.reshape(1, -1)
            
        input_tensor = torch.from_numpy(sample_input.astype(np.float32)).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output = self.pytorch_model(input_tensor)
            
        # Convert back to numpy and apply resolution scaling
        output_np = output.cpu().numpy()
        scaled_output = output_np * np.array([self.resx, self.resy])
        
        return scaled_output.flatten()  # Return as 1D array for consistency
    
    def run_onnx_inference(self, sample_input):
        """
        Run ONNX inference on a single sample.
        
        Args:
            sample_input (np.ndarray): Input sample of shape (1, 12) or (12,)
            
        Returns:
            np.ndarray: Model output, or None if ONNX model not available
        """
        if self.onnx_session is None:
            return None
            
        # Ensure input is the right shape
        if sample_input.ndim == 1:
            sample_input = sample_input.reshape(1, -1)
            
        # Get input name
        input_name = self.onnx_session.get_inputs()[0].name
        
        # Run inference
        onnx_output = self.onnx_session.run(None, {input_name: sample_input.astype(np.float32)})
        
        # Extract output (assuming single output)
        output = onnx_output[0]
        
        # Apply resolution scaling (matching MATLAB code)
        scaled_output = output * np.array([self.resx, self.resy])
        
        return scaled_output.flatten()  # Return as 1D array for consistency
    
    def test_all_samples(self):
        """Test both PyTorch and ONNX models on all filtered data samples."""
        print(f"\n{BLUE}{'='*50}")
        print("TESTING ALL FILTERED DATA SAMPLES")
        print(f"{'='*50}{RESET}")
        
        num_samples = self.filtered_data.shape[0]
        print(f"Testing {num_samples} samples from filtered data...")
        
        # Storage for results
        pytorch_predictions = []
        onnx_predictions = []
        pytorch_errors = []
        onnx_errors = []
        pytorch_onnx_diffs = []
        failed_samples = []
        
        # Track which models are available
        pytorch_available = self.pytorch_model is not None
        onnx_available = self.onnx_session is not None
        
        print(f"PyTorch model: {'✓ Available' if pytorch_available else '✗ Not available'}")
        print(f"ONNX model: {'✓ Available' if onnx_available else '✗ Not available'}")
        
        # Progress tracking
        start_time = time.time()
        
        for sample_idx in range(num_samples):
            # Progress reporting
            if sample_idx % 1000 == 0 or sample_idx < 5:
                print(f"Processing sample {sample_idx + 1}/{num_samples}")
                
            try:
                # Get sample input
                sample_input = self.filtered_data[sample_idx, :]  # Shape: (12,)
                
                # Run PyTorch inference
                pytorch_pred = None
                if pytorch_available:
                    pytorch_pred = self.run_pytorch_inference(sample_input)
                    
                    # Check for invalid outputs
                    if pytorch_pred is not None and (np.any(np.isnan(pytorch_pred)) or np.any(np.isinf(pytorch_pred))):
                        print(f"✗ Sample {sample_idx + 1}: Invalid PyTorch output (NaN/Inf)")
                        failed_samples.append(sample_idx)
                        continue
                
                # Run ONNX inference
                onnx_pred = None
                if onnx_available:
                    onnx_pred = self.run_onnx_inference(sample_input)
                    
                    # Check for invalid outputs
                    if onnx_pred is not None and (np.any(np.isnan(onnx_pred)) or np.any(np.isinf(onnx_pred))):
                        print(f"✗ Sample {sample_idx + 1}: Invalid ONNX output (NaN/Inf)")
                        failed_samples.append(sample_idx)
                        continue
                
                # Store predictions
                if pytorch_pred is not None:
                    pytorch_predictions.append(pytorch_pred)
                if onnx_pred is not None:
                    onnx_predictions.append(onnx_pred)
                
                # Compare PyTorch vs ONNX if both available
                if pytorch_pred is not None and onnx_pred is not None:
                    diff = pytorch_pred - onnx_pred
                    pytorch_onnx_diffs.append(diff)
                
                # Compute errors against ground truth if available
                if self.ground_truth is not None:
                    # Handle ground truth format - MATLAB stores as (2, N)
                    if self.ground_truth.shape[0] == 2:
                        gt_sample = self.ground_truth[:, sample_idx]  # Shape: (2,)
                    else:
                        gt_sample = self.ground_truth[sample_idx, :]  # Shape: (2,)
                    
                    # Compute errors: ground_truth - prediction (matching MATLAB)
                    if pytorch_pred is not None:
                        pytorch_error = gt_sample - pytorch_pred
                        pytorch_errors.append(pytorch_error)
                        
                    if onnx_pred is not None:
                        onnx_error = gt_sample - onnx_pred
                        onnx_errors.append(onnx_error)
                    
            except Exception as e:
                print(f"✗ Sample {sample_idx + 1} failed: {e}")
                failed_samples.append(sample_idx)
                
        # Convert to numpy arrays
        if pytorch_predictions:
            pytorch_predictions = np.array(pytorch_predictions)
        else:
            pytorch_predictions = np.array([]).reshape(0, 2)
            
        if onnx_predictions:
            onnx_predictions = np.array(onnx_predictions)
        else:
            onnx_predictions = np.array([]).reshape(0, 2)
            
        if pytorch_errors:
            pytorch_errors = np.array(pytorch_errors)
        else:
            pytorch_errors = np.array([]).reshape(0, 2)
            
        if onnx_errors:
            onnx_errors = np.array(onnx_errors)
        else:
            onnx_errors = np.array([]).reshape(0, 2)
            
        if pytorch_onnx_diffs:
            pytorch_onnx_diffs = np.array(pytorch_onnx_diffs)
        else:
            pytorch_onnx_diffs = np.array([]).reshape(0, 2)
            
        elapsed_time = time.time() - start_time
        
        print(f"\n{'-'*40}")
        print("TESTING SUMMARY:")
        print(f"  Total samples tested: {num_samples}")
        print(f"  PyTorch valid predictions: {len(pytorch_predictions)}")
        print(f"  ONNX valid predictions: {len(onnx_predictions)}")
        print(f"  Failed predictions: {len(failed_samples)}")
        print(f"  Processing time: {elapsed_time:.2f} seconds")
        
        # Store results
        self.results['num_total_samples'] = num_samples
        self.results['num_failed_samples'] = len(failed_samples)
        self.results['failed_sample_indices'] = failed_samples
        self.results['processing_time'] = elapsed_time
        
        self.results['pytorch_predictions'] = pytorch_predictions
        self.results['onnx_predictions'] = onnx_predictions
        self.results['pytorch_errors'] = pytorch_errors
        self.results['onnx_errors'] = onnx_errors
        self.results['pytorch_onnx_diffs'] = pytorch_onnx_diffs
        
        return pytorch_predictions, onnx_predictions, pytorch_errors, onnx_errors, pytorch_onnx_diffs
    
    def compute_error_statistics(self, errors, model_name):
        """
        Compute comprehensive error statistics.
        
        Args:
            errors (np.ndarray): Array of errors with shape (N, 2)
            model_name (str): Name of the model for display
        """
        if len(errors) == 0:
            print(f"No valid errors to compute statistics for {model_name}.")
            return {}
            
        print(f"\n{BLUE}{model_name.upper()} ERROR STATISTICS{RESET}")
        print(f"{'-'*50}")
        
        print('Error Shape: ', errors.shape)

        # Separate errors for X and Y components
        errors_x = errors[:, 0]
        errors_y = errors[:, 1]
        
        # Compute Euclidean distance errors
        euclidean_errors = np.sqrt(errors_x**2 + errors_y**2)
        
        # Compute statistics for each component
        def compute_component_stats(errors, component_name):
            abs_errors = np.abs(errors)
            stats_dict = {
                'mean': np.mean(abs_errors),
                'median': np.median(abs_errors),
                'std': np.std(abs_errors),
                'min': np.min(abs_errors),
                'max': np.max(abs_errors),
                'quantile_95': np.quantile(abs_errors, 0.95),
                'quantile_99': np.quantile(abs_errors, 0.99)
            }
            
            print(f"    {component_name} Component Errors (Absolute Values):")
            print(f"      Mean: {stats_dict['mean']:.6f}, Median: {stats_dict['median']:.6f}")
            print(f"      Std: {stats_dict['std']:.6f}")
            print(f"      Min: {stats_dict['min']:.6f}, Max: {stats_dict['max']:.6f}")
            print(f"      95th quantile: {stats_dict['quantile_95']:.6f}, 99th quantile: {stats_dict['quantile_99']:.6f}")
            
            return stats_dict
        
        # Compute and display statistics
        x_stats = compute_component_stats(errors_x, "X")
        y_stats = compute_component_stats(errors_y, "Y")
        euclidean_stats = compute_component_stats(euclidean_errors, "Euclidean Distance")
        
        # Additional analysis
        print(f"\n    Additional Analysis:")
        print(f"      RMS Error X: {np.sqrt(np.mean(errors_x**2)):.6f}")
        print(f"      RMS Error Y: {np.sqrt(np.mean(errors_y**2)):.6f}")
        print(f"      RMS Euclidean Error: {np.sqrt(np.mean(euclidean_errors**2)):.6f}")
        print(f"      Mean Absolute Error X: {np.mean(np.abs(errors_x)):.6f}")
        print(f"      Mean Absolute Error Y: {np.mean(np.abs(errors_y)):.6f}")
        
        # Correlation between X and Y errors (keep raw values for correlation)
        correlation = np.corrcoef(errors_x, errors_y)[0, 1]
        print(f"      X-Y Error Correlation: {correlation:.6f}")
        
        return {
            'x_component': x_stats,
            'y_component': y_stats,
            'euclidean_distance': euclidean_stats,
            'raw_errors_x': errors_x,
            'raw_errors_y': errors_y,
            'euclidean_errors': euclidean_errors,
            'rms_x': np.sqrt(np.mean(errors_x**2)),
            'rms_y': np.sqrt(np.mean(errors_y**2)),
            'rms_euclidean': np.sqrt(np.mean(euclidean_errors**2)),
            'mean_abs_x': np.mean(np.abs(errors_x)),
            'mean_abs_y': np.mean(np.abs(errors_y)),
            'correlation_xy': correlation
        }
    
    def compute_prediction_statistics(self, predictions, model_name):
        """
        Compute statistics for the predictions themselves.
        
        Args:
            predictions (np.ndarray): Array of predictions with shape (N, 2)
            model_name (str): Name of the model for display
        """
        if len(predictions) == 0:
            print(f"No valid predictions to analyze for {model_name}.")
            return {}
            
        print(f"\n{BLUE}{model_name.upper()} PREDICTION STATISTICS{RESET}")
        print(f"{'-'*50}")
        
        # Mean prediction
        mean_prediction = np.mean(predictions, axis=0)
        print(f"  Mean prediction: [{mean_prediction[0]:.6f}, {mean_prediction[1]:.6f}]")
        
        # Prediction ranges
        pred_x = predictions[:, 0]
        pred_y = predictions[:, 1]
        
        print(f"  X predictions - Min: {np.min(pred_x):.6f}, Max: {np.max(pred_x):.6f}, Std: {np.std(pred_x):.6f}")
        print(f"  Y predictions - Min: {np.min(pred_y):.6f}, Max: {np.max(pred_y):.6f}, Std: {np.std(pred_y):.6f}")
        
        return {
            'mean_prediction': mean_prediction,
            'x_min': np.min(pred_x),
            'x_max': np.max(pred_x),
            'x_std': np.std(pred_x),
            'y_min': np.min(pred_y),
            'y_max': np.max(pred_y),
            'y_std': np.std(pred_y),
            'count': len(predictions)
        }
    
    def compute_model_comparison_statistics(self, pytorch_onnx_diffs):
        """
        Compute statistics comparing PyTorch vs ONNX outputs.
        
        Args:
            pytorch_onnx_diffs (np.ndarray): Array of differences with shape (N, 2)
        """
        if len(pytorch_onnx_diffs) == 0:
            print("No valid predictions to compare between PyTorch and ONNX.")
            return {}
            
        print(f"\n{BLUE}PYTORCH vs ONNX COMPARISON{RESET}")
        print(f"{'-'*50}")
        
        # Separate differences for X and Y components
        diff_x = pytorch_onnx_diffs[:, 0]
        diff_y = pytorch_onnx_diffs[:, 1]
        
        # Compute Euclidean distance differences
        euclidean_diffs = np.sqrt(diff_x**2 + diff_y**2)
        
        # Statistics using absolute values
        abs_diff_x = np.abs(diff_x)
        abs_diff_y = np.abs(diff_y)
        
        print(f"  X Component Differences (Absolute Values):")
        print(f"    Mean: {np.mean(abs_diff_x):.6f}, Std: {np.std(abs_diff_x):.6f}")
        print(f"    Min: {np.min(abs_diff_x):.6f}, Max: {np.max(abs_diff_x):.6f}")
        print(f"    95th quantile: {np.quantile(abs_diff_x, 0.95):.6f}")
        
        print(f"  Y Component Differences (Absolute Values):")
        print(f"    Mean: {np.mean(abs_diff_y):.6f}, Std: {np.std(abs_diff_y):.6f}")
        print(f"    Min: {np.min(abs_diff_y):.6f}, Max: {np.max(abs_diff_y):.6f}")
        print(f"    95th quantile: {np.quantile(abs_diff_y, 0.95):.6f}")
        
        print(f"  Euclidean Distance Differences:")
        print(f"    Mean: {np.mean(euclidean_diffs):.6f}, Std: {np.std(euclidean_diffs):.6f}")
        print(f"    Min: {np.min(euclidean_diffs):.6f}, Max: {np.max(euclidean_diffs):.6f}")
        print(f"    95th quantile: {np.quantile(euclidean_diffs, 0.95):.6f}")
        
        # Check if models are close
        tolerance = 1e-3
        close_samples = euclidean_diffs < tolerance
        close_percentage = 100 * np.sum(close_samples) / len(euclidean_diffs)
        print(f"  Samples within tolerance ({tolerance}): {np.sum(close_samples)}/{len(euclidean_diffs)} ({close_percentage:.1f}%)")
        
        return {
            'x_mean_abs': np.mean(abs_diff_x),
            'x_std_abs': np.std(abs_diff_x),
            'x_min_abs': np.min(abs_diff_x),
            'x_max_abs': np.max(abs_diff_x),
            'x_quantile_95': np.quantile(abs_diff_x, 0.95),
            'y_mean_abs': np.mean(abs_diff_y),
            'y_std_abs': np.std(abs_diff_y),
            'y_min_abs': np.min(abs_diff_y),
            'y_max_abs': np.max(abs_diff_y),
            'y_quantile_95': np.quantile(abs_diff_y, 0.95),
            'euclidean_mean': np.mean(euclidean_diffs),
            'euclidean_std': np.std(euclidean_diffs),
            'euclidean_min': np.min(euclidean_diffs),
            'euclidean_max': np.max(euclidean_diffs),
            'euclidean_quantile_95': np.quantile(euclidean_diffs, 0.95),
            'close_samples': np.sum(close_samples),
            'total_samples': len(euclidean_diffs),
            'close_percentage': close_percentage,
            'tolerance': tolerance,
            'raw_diff_x': diff_x,  # Keep raw values for further analysis
            'raw_diff_y': diff_y
        }
    
    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to native Python types for JSON serialization.
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_numpy_types(item) for item in obj)
        else:
            return obj

    def save_results(self, output_dir="python_model_comparison_results"):
        """Save results to files."""
        print(f"\n{BLUE}{'='*50}")
        print("SAVING RESULTS")
        print(f"{'='*50}{RESET}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save to .mat file for MATLAB compatibility
        mat_filename = os.path.join(output_dir, "python_model_comparison_results.mat")
        
        # Prepare data for .mat file
        mat_data = {
            'num_total_samples': self.results['num_total_samples'],
            'num_failed_samples': self.results['num_failed_samples'],
            'processing_time': self.results['processing_time'],
            'pytorch_model_path': self.pytorch_model_path,
            'onnx_model_path': self.onnx_model_path,
            'data_path': self.filtered_data_path,
            'resolution': [self.resx, self.resy]
        }
        
        # Add model-specific data
        if len(self.results['pytorch_predictions']) > 0:
            mat_data['pytorch_predictions'] = self.results['pytorch_predictions']
            mat_data['num_pytorch_valid'] = len(self.results['pytorch_predictions'])
            
        if len(self.results['onnx_predictions']) > 0:
            mat_data['onnx_predictions'] = self.results['onnx_predictions']
            mat_data['num_onnx_valid'] = len(self.results['onnx_predictions'])
            
        # Add error data if available
        if len(self.results['pytorch_errors']) > 0:
            mat_data['pytorch_errors'] = self.results['pytorch_errors']
            
        if len(self.results['onnx_errors']) > 0:
            mat_data['onnx_errors'] = self.results['onnx_errors']
            
        # Add comparison data
        if len(self.results['pytorch_onnx_diffs']) > 0:
            mat_data['pytorch_onnx_differences'] = self.results['pytorch_onnx_diffs']
            
        # Add statistics if computed
        for key in ['pytorch_error_stats', 'onnx_error_stats', 'pytorch_pred_stats', 
                   'onnx_pred_stats', 'comparison_stats']:
            if key in self.results:
                mat_data[key] = self.results[key]
                
        sio.savemat(mat_filename, mat_data)
        print(f"✓ Results saved to: {mat_filename}")
        
        # Save detailed JSON report
        json_filename = os.path.join(output_dir, "python_model_detailed_report.json")
        
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for key, value in self.results.items():
            if isinstance(value, np.ndarray):
                if value.size < 1000:  # Only save small arrays to JSON
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = f"<large_array_shape_{value.shape}>"
            elif isinstance(value, dict):
                # Use the helper function to convert numpy types recursively
                json_data[key] = self._convert_numpy_types(value)
            else:
                json_data[key] = self._convert_numpy_types(value)
                
        # Add metadata
        json_data['metadata'] = {
            'pytorch_model_path': self.pytorch_model_path,
            'onnx_model_path': self.onnx_model_path,
            'filtered_data_path': self.filtered_data_path,
            'resolution_x': self.resx,
            'resolution_y': self.resy,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'pytorch_available': self.pytorch_model is not None,
            'onnx_available': self.onnx_session is not None
        }
        
        try:
            with open(json_filename, 'w') as f:
                json.dump(json_data, f, indent=2)
            print(f"✓ Detailed report saved to: {json_filename}")
        except TypeError as e:
            print(f"{YELLOW}Warning: Could not save JSON file due to serialization error: {e}{RESET}")
            print(f"Results are still available in .mat and .txt formats")
        
        # Save summary text file
        self._save_text_summary(output_dir)
        
    def _save_text_summary(self, output_dir):
        """Save a comprehensive text summary."""
        summary_filename = os.path.join(output_dir, "python_model_comparison_summary.txt")
        with open(summary_filename, 'w') as f:
            f.write("Python Model Comparison Testing Summary\n")
            f.write("=======================================\n\n")
            f.write(f"PyTorch Model: {self.pytorch_model_path}\n")
            f.write(f"ONNX Model: {self.onnx_model_path}\n")
            f.write(f"Data: {self.filtered_data_path}\n")
            f.write(f"Resolution: {self.resx} x {self.resy}\n\n")
            
            f.write("Test Results:\n")
            f.write(f"  Total samples: {self.results['num_total_samples']}\n")
            f.write(f"  PyTorch valid predictions: {len(self.results['pytorch_predictions'])}\n")
            f.write(f"  ONNX valid predictions: {len(self.results['onnx_predictions'])}\n")
            f.write(f"  Failed predictions: {self.results['num_failed_samples']}\n")
            f.write(f"  Processing time: {self.results['processing_time']:.2f} seconds\n\n")
            
            # Add statistics summaries
            if 'pytorch_pred_stats' in self.results:
                ps = self.results['pytorch_pred_stats']
                f.write("PyTorch Prediction Statistics:\n")
                f.write(f"  Mean prediction: [{ps['mean_prediction'][0]:.6f}, {ps['mean_prediction'][1]:.6f}]\n")
                f.write(f"  X range: [{ps['x_min']:.6f}, {ps['x_max']:.6f}]\n")
                f.write(f"  Y range: [{ps['y_min']:.6f}, {ps['y_max']:.6f}]\n\n")
                
            if 'onnx_pred_stats' in self.results:
                ps = self.results['onnx_pred_stats']
                f.write("ONNX Prediction Statistics:\n")
                f.write(f"  Mean prediction: [{ps['mean_prediction'][0]:.6f}, {ps['mean_prediction'][1]:.6f}]\n")
                f.write(f"  X range: [{ps['x_min']:.6f}, {ps['x_max']:.6f}]\n")
                f.write(f"  Y range: [{ps['y_min']:.6f}, {ps['y_max']:.6f}]\n\n")
                
            if 'comparison_stats' in self.results:
                cs = self.results['comparison_stats']
                f.write("PyTorch vs ONNX Comparison:\n")
                f.write(f"  Mean Euclidean difference: {cs['euclidean_mean']:.6f}\n")
                f.write(f"  Max Euclidean difference: {cs['euclidean_max']:.6f}\n")
                f.write(f"  Samples within tolerance: {cs['close_samples']}/{cs['total_samples']} ({cs['close_percentage']:.1f}%)\n\n")
                
            if 'pytorch_error_stats' in self.results:
                es = self.results['pytorch_error_stats']
                f.write("PyTorch Error Statistics:\n")
                f.write(f"  Mean Euclidean error: {es['euclidean_distance']['mean']:.6f}\n")
                f.write(f"  95th quantile Euclidean error: {es['euclidean_distance']['quantile_95']:.6f}\n\n")
                
            if 'onnx_error_stats' in self.results:
                es = self.results['onnx_error_stats']
                f.write("ONNX Error Statistics:\n")
                f.write(f"  Mean Euclidean error: {es['euclidean_distance']['mean']:.6f}\n")
                f.write(f"  95th quantile Euclidean error: {es['euclidean_distance']['quantile_95']:.6f}\n\n")
                
        print(f"✓ Summary saved to: {summary_filename}")
        
    def run_complete_test(self):
        """Run the complete testing pipeline for both models."""
        print(f"{GREEN}{'='*60}")
        print("PYTHON MODEL COMPARISON TESTING - PYTORCH vs ONNX")
        print(f"{'='*60}{RESET}")
        
        try:
            # Load models and data
            pytorch_loaded = self.load_pytorch_model()
            onnx_loaded = self.load_onnx_model()
            data_loaded = self.load_filtered_data()
            
            if not data_loaded:
                raise RuntimeError("Failed to load filtered data")
                
            if not pytorch_loaded and not onnx_loaded:
                raise RuntimeError("No models could be loaded")
            
            # Run tests
            pytorch_preds, onnx_preds, pytorch_errs, onnx_errs, model_diffs = self.test_all_samples()
            
            # Compute statistics
            if len(pytorch_preds) > 0:
                self.results['pytorch_pred_stats'] = self.compute_prediction_statistics(pytorch_preds, "PyTorch")
                
            if len(onnx_preds) > 0:
                self.results['onnx_pred_stats'] = self.compute_prediction_statistics(onnx_preds, "ONNX")
                
            if len(pytorch_errs) > 0:
                self.results['pytorch_error_stats'] = self.compute_error_statistics(pytorch_errs, "PyTorch")
                
            if len(onnx_errs) > 0:
                self.results['onnx_error_stats'] = self.compute_error_statistics(onnx_errs, "ONNX")
                
            if len(model_diffs) > 0:
                self.results['comparison_stats'] = self.compute_model_comparison_statistics(model_diffs)
            
            # Save results
            self.save_results()
            
            # Final summary
            print(f"\n{GREEN}{'='*60}")
            print("FINAL SUMMARY")
            print(f"{'='*60}{RESET}")
            
            total_samples = self.results['num_total_samples']
            pytorch_success = len(pytorch_preds)
            onnx_success = len(onnx_preds)
            failed = self.results['num_failed_samples']
            
            print(f"Models tested: {self.pytorch_model_path} (PyTorch), {self.onnx_model_path} (ONNX)")
            print(f"Total samples: {total_samples}")
            print(f"PyTorch successful: {pytorch_success} ({100*pytorch_success/total_samples:.1f}%)")
            print(f"ONNX successful: {onnx_success} ({100*onnx_success/total_samples:.1f}%)")
            print(f"Failed: {failed} ({100*failed/total_samples:.1f}%)")
            
            if len(model_diffs) > 0:
                cs = self.results['comparison_stats']
                print(f"Models agree within tolerance: {cs['close_percentage']:.1f}% of samples")
                
            print(f"\nThe Python model comparison validation is complete.")
            
        except Exception as e:
            print(f"{RED}Error during testing: {e}{RESET}")
            import traceback
            traceback.print_exc()
            raise

def main():
    """Main function to run the model comparison testing."""
    
    # Configuration - update these paths as needed
    pytorch_model_path = "checkpoints/ambitious_calf_40_epoch_2213.pth"
    onnx_model_path = "onnx_models/ambitious_calf_40_epoch_2213.onnx"
    filtered_data_path = "datasets/training_data_WithBlob2_WithSunDirAngle_136856_ID0_filtered.mat"
    
    print("Configuration:")
    print(f"  PyTorch Model: {pytorch_model_path}")
    print(f"  ONNX Model: {onnx_model_path}")
    print(f"  Filtered Data: {filtered_data_path}")
    
    # Check if files exist
    missing_files = []
    if not os.path.exists(pytorch_model_path):
        missing_files.append(f"PyTorch model: {pytorch_model_path}")
        
    if not os.path.exists(onnx_model_path):
        missing_files.append(f"ONNX model: {onnx_model_path}")
        
    if not os.path.exists(filtered_data_path):
        missing_files.append(f"Filtered data: {filtered_data_path}")
        
    if missing_files:
        print(f"{RED}Error: Missing files:{RESET}")
        for file in missing_files:
            print(f"  - {file}")
        print("Please ensure all required files are available.")
        return
        
    # Create tester and run complete test
    tester = ModelTester(pytorch_model_path, onnx_model_path, filtered_data_path)
    tester.run_complete_test()

if __name__ == "__main__":
    main()