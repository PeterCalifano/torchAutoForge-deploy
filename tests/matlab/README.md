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

PyTorch-to-ONNX export and numerical output comparison are provided by PTAF's
`ModelHandlerONNx` exporter and `onnx_validate` method. The former hard-coded
`ExportPytorchToONNX.py` script has been removed; its `.mat` report is not required.

The legacy dataset accuracy script remains at
`python/scripts/tmp_to_rework_as_generic/TestOnnxAccuracy.py`.

NOTE: The model class can be found in (Gitlab/COSMICA) => 
nav-frontend/.experimental/neuralCOB/NeuralCOB_module

