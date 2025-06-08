%% MATLAB Script to Compare Python and MATLAB ONNX Results
% This script loads Python ONNX results and compares them with MATLAB ONNX outputs

close all
clear
clc

% Add path to your MATLAB functions
addpath('../../matlab');
addpath('../../../nav-frontend/src/algorithm/Centroiding_COB')

bInferenceFunction = 1; % 0: RunModelInference, 1: RunNeuralCOB 

%% CONFIGURATION
fprintf('=%.50s\n', repmat('=', 1, 50));
fprintf('CONFIGURATION\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Model identifier - change this to test different models
stateName = "adventurous-colt-164_epoch_1262";

% File paths
charONNxModelfilePath = sprintf("onnxModels/%s.onnx", stateName);
pythonDataPath = sprintf("testData/%s_python_onnx_all_results.mat", stateName); % Path to Python results
filteredDataPath = 'datasets/RTO4t1j13p0_test_data_WithBlob2_WithSunDirAngle_6111_ID0_fixed_filtered.mat';

dModelInputSizes = [1, 12]; % Input size of the model (BC)
tolerance = 1e-2; % Tolerance for numerical comparisons

fprintf('Model State: %s\n', stateName);
fprintf('ONNX Model Path: %s\n', charONNxModelfilePath);
fprintf('Input Sizes: [%s]\n', num2str(dModelInputSizes));
fprintf('Tolerance: %.1e\n', tolerance);
fprintf('Python Data Path: %s\n', pythonDataPath);

% Check if ONNX file exists
if ~isfile(charONNxModelfilePath)
    error('ONNX file not found: %s', charONNxModelfilePath);
end

% Check if Python data file exists
if ~isfile(pythonDataPath)
    error('Python ONNX results file not found: %s\nPlease run the Python script first to generate the data.', pythonDataPath);
end

fprintf('✓ ONNX file found\n');
fprintf('✓ Python data file found\n');

%% LOAD PYTHON ONNX RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('LOADING PYTHON ONNX RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

try
    pythonData = load(pythonDataPath);
    fprintf('✓ Python ONNX results loaded successfully\n');
    
    % Display Python test information
    fprintf('Python test count: %d\n', pythonData.test_count);
    
catch ME
    fprintf('Error loading Python ONNX results: %s\n', ME.message);
    return;
end

%% LOAD MATLAB ONNX MODEL
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('LOADING MATLAB ONNX MODEL\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

try
    objModel = ImportONNxModel(charONNxModelfilePath, dModelInputSizes);
    
    % Verify model class
    if strcmpi(class(objModel), 'dlnetwork')
        fprintf('✓ MATLAB ONNX model loaded successfully as dlnetwork\n');
    else
        fprintf('⚠ Model loaded as %s (expected dlnetwork)\n', class(objModel));
    end
    
    % Display model information
    fprintf('Model class: %s\n', class(objModel));
    
catch ME
    fprintf('Error loading MATLAB ONNX model: %s\n', ME.message);
    fprintf('Occurred in: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    return;
end

%% COMPARE PYTHON AND MATLAB ONNX RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('COMPARING PYTHON AND MATLAB ONNX RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Store comparison results
comparisonResults = struct();
allTestsPassed = true;

for testIdx = 1:pythonData.test_count
    fprintf('\nTest Case %d\n', testIdx);
    fprintf('%.40s\n', repmat('-', 1, 40));
    
    try
        % Get Python test data
        pythonInputFieldName = sprintf('test_%d_input', testIdx);
        pythonOutputFieldName = sprintf('test_%d_onnx_output', testIdx);
        
        pythonInput = pythonData.(pythonInputFieldName);
        pythonOnnxOutput = pythonData.(pythonOutputFieldName);
        
        fprintf('Python input shape: [%s]\n', num2str(size(pythonInput)));
        fprintf('Python ONNX output shape: [%s]\n', num2str(size(pythonOnnxOutput)));
        
        % Run MATLAB ONNX inference with the same input
        if bInferenceFunction==0
            matlabOnnxOutput = RunModelInference(objModel, pythonInput, "charInputShapeFormat", "BC");
        elseif bInferenceFunction==1
            matlabOnnxOutput = RunNeuralCOB(pythonInput, false, "charInputShapeFormat", "BC", "objModel",objModel);
        else
            fprintf('ERROR: Inference function not detected!');
        end
        fprintf('MATLAB ONNX output shape: [%s]\n', num2str(size(matlabOnnxOutput)));
        
        % Compare outputs
        outputDiff = abs(pythonOnnxOutput' - matlabOnnxOutput);
        maxDiff = max(outputDiff(:));
        meanDiff = mean(outputDiff(:));
        
        % Check if outputs are close
        isClose = maxDiff < tolerance;
        
        % Display comparison results
        fprintf('Python ONNX output sample: [%.6f, %.6f, %.6f, ...]\n', ...
                pythonOnnxOutput(1), pythonOnnxOutput(2), pythonOnnxOutput(min(3,end)));
        fprintf('MATLAB ONNX output sample: [%.6f, %.6f, %.6f, ...]\n', ...
                matlabOnnxOutput(1), matlabOnnxOutput(2), matlabOnnxOutput(min(3,end)));
        fprintf('Max absolute difference: %.6e\n', maxDiff);
        fprintf('Mean absolute difference: %.6e\n', meanDiff);
        
        if isClose
            fprintf('✓ Outputs match within tolerance (%.1e)\n', tolerance);
        else
            fprintf('✗ Outputs differ beyond tolerance (%.1e)\n', tolerance);
            allTestsPassed = false;
        end
        
        % Store results
        comparisonResults.(sprintf('test_%d', testIdx)) = struct();
        comparisonResults.(sprintf('test_%d', testIdx)).input = pythonInput;
        comparisonResults.(sprintf('test_%d', testIdx)).python_output = pythonOnnxOutput;
        comparisonResults.(sprintf('test_%d', testIdx)).matlab_output = matlabOnnxOutput;
        comparisonResults.(sprintf('test_%d', testIdx)).max_diff = maxDiff;
        comparisonResults.(sprintf('test_%d', testIdx)).mean_diff = meanDiff;
        comparisonResults.(sprintf('test_%d', testIdx)).passed = isClose;
        
        % Check for NaN or Inf values
        if any(isnan(matlabOnnxOutput(:)))
            fprintf('⚠ Warning: MATLAB output contains NaN values\n');
            allTestsPassed = false;
        end
        
        if any(isinf(matlabOnnxOutput(:)))
            fprintf('⚠ Warning: MATLAB output contains infinite values\n');
            allTestsPassed = false;
        end
        
    catch ME
        fprintf('✗ Test failed: %s\n', ME.message);
        comparisonResults.(sprintf('test_%d', testIdx)).passed = false;
        comparisonResults.(sprintf('test_%d', testIdx)).error = ME.message;
        allTestsPassed = false;
    end
end

%% CROSS-PLATFORM VALIDATION
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('CROSS-PLATFORM VALIDATION\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Compare with Python's internal PyTorch vs ONNX comparisons
fprintf('Comparing with Python internal validation:\n');
for testIdx = 1:pythonData.test_count
    pythonPassedFieldName = sprintf('test_%d_passed', testIdx);
    if isfield(pythonData, pythonPassedFieldName)
        pythonPassed = pythonData.(pythonPassedFieldName);
        matlabPassed = comparisonResults.(sprintf('test_%d', testIdx)).passed;
        
        % Convert logical to string
        if pythonPassed
            pythonStatus = 'PASS';
        else
            pythonStatus = 'FAIL';
        end
        
        if matlabPassed
            matlabStatus = 'PASS';
        else
            matlabStatus = 'FAIL';
        end
        
        fprintf('  Test %d: Python PyTorch vs ONNX (%s), Python vs MATLAB ONNX (%s)\n', ...
                testIdx, pythonStatus, matlabStatus);
    end
end

%% FILTERED DATA TEST - MATLAB MODEL VALIDATION
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('FILTERED DATA TEST - MATLAB MODEL VALIDATION\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Resolution (for outputscaling)
resx = 2048;
resy = 1536;

if isfile(filteredDataPath)
    fprintf('Loading filtered data from: %s\n', filteredDataPath);
    
    try
        % Load filtered data
        filteredData = load(filteredDataPath);
        filtered_matrix = filteredData.dMixedFeatMatrixAll_filtered;
        groundTruth = filteredData.groundTruth;
        
        fprintf('✓ Filtered data loaded successfully\n');
        fprintf('Filtered data shape: [%s]\n', num2str(size(filtered_matrix)));
        
        % Test all samples
        numSamplesToTest = size(filtered_matrix, 1);
        % numSamplesToTest = 1000;
        allPredictions = [];
        allErrors = [];
        filteredTestsPassed = true;
        
        fprintf('Testing all %d samples from filtered data with MATLAB model...\n', numSamplesToTest);
        
        % Test each sample individually (batch = 1)
        for sampleIdx = 1:numSamplesToTest
            if mod(sampleIdx, 1000) == 1 || sampleIdx <= 5
                fprintf('Processing sample %d/%d\n', sampleIdx, numSamplesToTest);
            end
            
            try
                % Get single sample (1x12)
                sampleInput = filtered_matrix(sampleIdx, :); % 1x12 input
                sampleGroundTruth = groundTruth(:, sampleIdx);
                
                % Run MATLAB model inference
                matlabOutput = RunModelInference(objModel, sampleInput, "charInputShapeFormat", "BC").*[resx; resy];
                
                % Check if output is valid (no NaN or Inf)
                isValidOutput = ~any(isnan(matlabOutput(:))) && ~any(isinf(matlabOutput(:)));

                % Compute error
                sampleError = sampleGroundTruth-matlabOutput;
                
                if isValidOutput
                    % Store prediction (should be 2x1)
                    allPredictions = [allPredictions; matlabOutput(:)'];
                    allErrors = [allErrors; sampleError(:)'];
                else
                    fprintf('✗ Sample %d: Invalid output (NaN/Inf)\n', sampleIdx);
                    filteredTestsPassed = false;
                end
                
            catch ME
                fprintf('✗ Sample %d failed: %s\n', sampleIdx, ME.message);
                filteredTestsPassed = false;
            end
        end
        
        % Compute final statistics
        fprintf('\n%.40s\n', repmat('-', 1, 40));
        fprintf('MATLAB Model Validation Summary:\n');
        fprintf('  Total samples tested: %d\n', numSamplesToTest);
        fprintf('  Valid predictions: %d\n', size(allPredictions, 1));
        fprintf('  Failed predictions: %d\n', numSamplesToTest - size(allPredictions, 1));
        
        % Store basic results
        filteredTestResults = struct();
        filteredTestResults.all_predictions = allPredictions;
        filteredTestResults.num_valid_samples = size(allPredictions, 1);
        filteredTestResults.num_total_samples = numSamplesToTest;
        
        if ~isempty(allPredictions)
            % Compute mean prediction across all samples
            meanPrediction = mean(allPredictions, 1);
            filteredTestResults.mean_prediction = meanPrediction;
            fprintf('  Mean prediction: [%.6f, %.6f]\n', meanPrediction(1), meanPrediction(2));
            
            % Compute error statistics if ground truth is available
            if ~isempty(allErrors)
                fprintf('\n  Error Statistics:\n');
                
                % Separate errors for X and Y components
                errorsX = allErrors(:, 1);
                errorsY = allErrors(:, 2);
                
                % Compute Euclidean distance errors
                euclideanErrors = sqrt(errorsX.^2 + errorsY.^2);
                
                % Mean error (using absolute values)
                meanErrorX = mean(abs(errorsX));
                meanErrorY = mean(abs(errorsY));
                meanEuclideanError = mean(euclideanErrors);
                
                % Median error (using absolute values)
                medianErrorX = median(abs(errorsX));
                medianErrorY = median(abs(errorsY));
                medianEuclideanError = median(euclideanErrors);
                
                % Min/Max errors (using absolute values)
                minErrorX = min(abs(errorsX));
                maxErrorX = max(abs(errorsX));
                minErrorY = min(abs(errorsY));
                maxErrorY = max(abs(errorsY));
                minEuclideanError = min(euclideanErrors);
                maxEuclideanError = max(euclideanErrors);
                
                % Quantiles (using absolute values)
                quantile95ErrorX = quantile(abs(errorsX), 0.95);
                quantile99ErrorX = quantile(abs(errorsX), 0.99);
                quantile95ErrorY = quantile(abs(errorsY), 0.95);
                quantile99ErrorY = quantile(abs(errorsY), 0.99);
                quantile95EuclideanError = quantile(euclideanErrors, 0.95);
                quantile99EuclideanError = quantile(euclideanErrors, 0.99);
                
                % Display results
                fprintf('    X Component Errors (Absolute Values):\n');
                fprintf('      Mean: %.6f\n', meanErrorX);
                fprintf('      Median: %.6f\n', medianErrorX);
                fprintf('      Min: %.6f\n', minErrorX);
                fprintf('      Max: %.6f\n', maxErrorX);
                fprintf('      95th quantile: %.6f\n', quantile95ErrorX);
                fprintf('      99th quantile: %.6f\n', quantile99ErrorX);
                
                fprintf('    Y Component Errors (Absolute Values):\n');
                fprintf('      Mean: %.6f\n', meanErrorY);
                fprintf('      Median: %.6f\n', medianErrorY);
                fprintf('      Min: %.6f\n', minErrorY);
                fprintf('      Max: %.6f\n', maxErrorY);
                fprintf('      95th quantile: %.6f\n', quantile95ErrorY);
                fprintf('      99th quantile: %.6f\n', quantile99ErrorY);
                
                fprintf('    Euclidean Distance Errors:\n');
                fprintf('      Mean: %.6f\n', meanEuclideanError);
                fprintf('      Median: %.6f\n', medianEuclideanError);
                fprintf('      Min: %.6f\n', minEuclideanError);
                fprintf('      Max: %.6f\n', maxEuclideanError);
                fprintf('      95th quantile: %.6f\n', quantile95EuclideanError);
                fprintf('      99th quantile: %.6f\n', quantile99EuclideanError);
                                
                % Additional analysis
                fprintf('\n    Additional Analysis:\n');
                fprintf('      RMS Error X: %.6f\n', sqrt(mean(errorsX.^2)));
                fprintf('      RMS Error Y: %.6f\n', sqrt(mean(errorsY.^2)));
                fprintf('      RMS Euclidean Error: %.6f\n', sqrt(mean(euclideanErrors.^2)));
                fprintf('      Mean Absolute Error X: %.6f\n', mean(abs(errorsX)));
                fprintf('      Mean Absolute Error Y: %.6f\n', mean(abs(errorsY)));
                
                % Correlation between X and Y errors (keep raw values for correlation)
                correlation = corrcoef(errorsX, errorsY);
                fprintf('      X-Y Error Correlation: %.6f\n', correlation(1,2));
                
                % Store error statistics in results (updated field names)
                filteredTestResults.all_errors = allErrors;
                filteredTestResults.mean_error = [meanErrorX, meanErrorY];
                filteredTestResults.median_error = [medianErrorX, medianErrorY];
                filteredTestResults.min_error = [minErrorX, minErrorY];
                filteredTestResults.max_error = [maxErrorX, maxErrorY];
                filteredTestResults.quantile95_error = [quantile95ErrorX, quantile95ErrorY];
                filteredTestResults.quantile99_error = [quantile99ErrorX, quantile99ErrorY];
                filteredTestResults.euclidean_errors = euclideanErrors;
                filteredTestResults.mean_euclidean_error = meanEuclideanError;
                filteredTestResults.median_euclidean_error = medianEuclideanError;
                filteredTestResults.min_euclidean_error = minEuclideanError;
                filteredTestResults.max_euclidean_error = maxEuclideanError;
                filteredTestResults.quantile95_euclidean_error = quantile95EuclideanError;
                filteredTestResults.quantile99_euclidean_error = quantile99EuclideanError;
                filteredTestResults.rms_error_x = sqrt(mean(errorsX.^2));
                filteredTestResults.rms_error_y = sqrt(mean(errorsY.^2));
                filteredTestResults.rms_euclidean_error = sqrt(mean(euclideanErrors.^2));
                filteredTestResults.mean_abs_error_x = mean(abs(errorsX));
                filteredTestResults.mean_abs_error_y = mean(abs(errorsY));
                filteredTestResults.xy_correlation = correlation(1,2);
            end
        else
            fprintf('  No valid predictions obtained!\n');
        end
        
        if filteredTestsPassed
            fprintf('✓ All samples processed successfully!\n');
        else
            fprintf('✗ Some samples failed processing.\n');
        end
        
    catch ME
        fprintf('✗ Error loading or processing filtered data: %s\n', ME.message);
        filteredTestsPassed = false;
        filteredTestResults = struct();
    end
    
else
    fprintf('⚠ Filtered data file not found: %s\n', filteredDataPath);
    fprintf('Please run the process_mat_file function first to generate filtered data.\n');
    filteredTestsPassed = true;
    filteredTestResults = struct();
end

%% SAVE COMPARISON RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('SAVING COMPARISON RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Save comparison results to .mat file
comparisonFileName = sprintf('outputSimulations/matlab_validation_%s_results.mat', stateName);
save(comparisonFileName, 'comparisonResults', 'pythonData', 'dModelInputSizes', ...
     'charONNxModelfilePath', 'tolerance', 'allTestsPassed', 'filteredTestResults', 'filteredTestsPassed');
fprintf('Comparison results saved to: %s\n', comparisonFileName);

%% FINAL RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('FINAL RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

fprintf('Comparison Summary:\n');
fprintf('  Python vs MATLAB tests: %d\n', pythonData.test_count);
passedPythonTests = 0;
for testIdx = 1:pythonData.test_count
    testFieldName = sprintf('test_%d', testIdx);
    if isfield(comparisonResults, testFieldName) && comparisonResults.(testFieldName).passed
        passedPythonTests = passedPythonTests + 1;
    end
end
fprintf('  Successful Python comparisons: %d\n', passedPythonTests);

if exist('filteredTestResults', 'var') && ~isempty(fieldnames(filteredTestResults))
    fprintf('  Filtered data MATLAB validation: %d samples\n', filteredTestResults.num_total_samples);
    fprintf('  Valid predictions: %d\n', filteredTestResults.num_valid_samples);
    if isfield(filteredTestResults, 'mean_prediction')
        fprintf('  Mean prediction: [%.6f, %.6f]\n', ...
                filteredTestResults.mean_error(1), filteredTestResults.mean_error(2));
    end
end

if allTestsPassed && filteredTestsPassed
    fprintf('\n✓ All tests passed! Models produce valid outputs.\n');
    fprintf('The MATLAB model works correctly with your filtered dataset.\n');
else
    fprintf('\n✗ Some tests failed. Review the detailed results above.\n');
    if ~allTestsPassed
        fprintf('  - Python vs MATLAB ONNX comparison failed\n');
    end
    if ~filteredTestsPassed
        fprintf('  - Filtered data MATLAB validation failed\n');
    end
end

fprintf('\nFiles created:\n');
fprintf('  - %s (Detailed comparison data)\n', comparisonFileName);