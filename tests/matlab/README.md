NOTE: This is required to "fix" the old datasets
dMixedFeatMatrixAll(:, 3) = dMixedFeatMatrixAll(:, 16);
dMixedFeatMatrixAll(:, 2) = dMixedFeatMatrixAll(:, 15);

------------------------------------------------------------

Pipeline: 
1. Use process_mat_file to fix the old datasets (wrong centroiding).
2. Use test_MatlabInference to compute performances on the given dataset &
   compare with ONNX results (computed on Python).
3. Use test_SimulinkInference to compute performances on the given dataset.
   Note: it should be exactly equal to Matlab one.

NOTE: In this repository (torchAutoForge-deploy - dev_UM branch) you can 
also find the python scripts: python/scripts

1. ExportPytorchToONNX.py : Converts checkpoint (.pth) into .onnx 
2. TestOnnxAccuracy.py : Computes the .onnx model accuracy on the given 
   dataset (which is the filtered one)

NOTE: The model class can be found in (Gitlab/COSMICA) => 
nav-frontend/.experimental/neuralCOB/NeuralCOB_module

