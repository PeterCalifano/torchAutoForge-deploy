%% SIMULINK Model Inference Testing Script
% This script tests the ONNX model in Simulink and computes comprehensive metrics

close all
clear
clc

%% CONFIGURATION
fprintf('=%.50s\n', repmat('=', 1, 50));
fprintf('SIMULINK MODEL TESTING CONFIGURATION\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% File paths and parameters
filteredDataPath = 'datasets/RTO4t1j13p0_test_data_WithBlob2_WithSunDirAngle_6111_ID0_fixed_filtered.mat';
simulinkModelName = 'testModelInferenceInSLX_v2';

% Resolution (for output scaling)
resx = 2048;
resy = 1536;

fprintf('Filtered Data Path: %s\n', filteredDataPath);
fprintf('Simulink Model: %s\n', simulinkModelName);
fprintf('Output Resolution: %dx%d\n', resx, resy);

%% LOAD TEST DATA
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('LOADING TEST DATA\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Check if filtered data file exists
if ~isfile(filteredDataPath)
    error('Filtered data file not found: %s', filteredDataPath);
end

try
    % Load filtered data
    fprintf('Loading filtered data from: %s\n', filteredDataPath);
    load(filteredDataPath);
    
    % Get number of test samples
    numberElements = size(dMixedFeatMatrixAll_filtered, 1);
    
    fprintf('✓ Filtered data loaded successfully\n');
    fprintf('Number of test samples: %d\n', numberElements);
    fprintf('Input feature dimension: %d\n', size(dMixedFeatMatrixAll_filtered, 2));
    fprintf('Ground truth shape: [%s]\n', num2str(size(groundTruth)));
    
    % Validate data consistency
    if size(groundTruth, 2) ~= numberElements
        warning('Ground truth samples (%d) do not match input samples (%d)', ...
                size(groundTruth, 2), numberElements);
    end
    
catch ME
    fprintf('✗ Error loading filtered data: %s\n', ME.message);
    return;
end

%% RUN SIMULINK SIMULATION
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('RUNNING SIMULINK SIMULATION\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

try
    fprintf('Starting Simulink simulation...\n');
    fprintf('Model: %s\n', simulinkModelName);
    fprintf('Processing %d samples...\n', numberElements);
    
    % Run Simulink simulation
    tic;
    simOut = sim(simulinkModelName);
    simulationTime = toc;
    
    fprintf('✓ Simulink simulation completed successfully\n');
    fprintf('Simulation time: %.2f seconds\n', simulationTime);
    fprintf('Average time per sample: %.4f seconds\n', simulationTime/numberElements);
    
catch ME
    fprintf('✗ Simulink simulation failed: %s\n', ME.message);
    fprintf('Occurred in: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    return;
end

%% PROCESS SIMULATION RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('PROCESSING SIMULATION RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

try
    % Extract predictions from simulation output
    prediction = simOut.prediction';
    
    fprintf('Raw prediction shape: [%s]\n', num2str(size(prediction)));
    
    % Apply output scaling
    prediction = prediction .* [resx; resy];
    
    fprintf('Scaled prediction shape: [%s]\n', num2str(size(prediction)));
    fprintf('Output scaling applied: [%d, %d]\n', resx, resy);
    
    % Validate prediction dimensions
    if size(prediction, 1) ~= 2
        error('Expected 2D output (X,Y coordinates), got %dD', size(prediction, 1));
    end
    
    if size(prediction, 2) ~= numberElements
        error('Prediction count (%d) does not match input count (%d)', ...
              size(prediction, 2), numberElements);
    end
    
    % Check for invalid predictions
    nanCount = sum(isnan(prediction(:)));
    infCount = sum(isinf(prediction(:)));
    
    if nanCount > 0
        fprintf('⚠ Warning: %d NaN values found in predictions\n', nanCount);
    end
    
    if infCount > 0
        fprintf('⚠ Warning: %d infinite values found in predictions\n', infCount);
    end
    
    validPredictions = numberElements - sum(any(isnan(prediction) | isinf(prediction), 1));
    fprintf('Valid predictions: %d/%d (%.1f%%)\n', ...
            validPredictions, numberElements, 100*validPredictions/numberElements);
    
catch ME
    fprintf('✗ Error processing simulation results: %s\n', ME.message);
    return;
end

%% COMPUTE ERROR METRICS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('COMPUTING ERROR METRICS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

try
    % Compute errors
    errors = groundTruth - prediction;
    
    fprintf('Error computation completed\n');
    fprintf('Error matrix shape: [%s]\n', num2str(size(errors)));
    
    % Separate errors for X and Y components
    errorsX = errors(1, :);
    errorsY = errors(2, :);
    
    % Compute absolute errors
    absErrors = abs(errors);
    absErrorsX = absErrors(1, :);
    absErrorsY = absErrors(2, :);
    
    % Compute Euclidean distance errors
    euclideanErrors = sqrt(errorsX.^2 + errorsY.^2);
    
    % Mean errors
    meanErrorX = mean(absErrorsX);
    meanErrorY = mean(absErrorsY);
    meanEuclideanError = mean(euclideanErrors);
    
    % Median errors
    medianErrorX = median(absErrorsX);
    medianErrorY = median(absErrorsY);
    medianEuclideanError = median(euclideanErrors);
    
    % Min/Max errors
    minErrorX = min(absErrorsX);
    maxErrorX = max(absErrorsX);
    minErrorY = min(absErrorsY);
    maxErrorY = max(absErrorsY);
    minEuclideanError = min(euclideanErrors);
    maxEuclideanError = max(euclideanErrors);
    
    % Quantiles
    quantile95ErrorX = quantile(absErrorsX, 0.95);
    quantile99ErrorX = quantile(absErrorsX, 0.99);
    quantile95ErrorY = quantile(absErrorsY, 0.95);
    quantile99ErrorY = quantile(absErrorsY, 0.99);
    quantile95EuclideanError = quantile(euclideanErrors, 0.95);
    quantile99EuclideanError = quantile(euclideanErrors, 0.99);
    
    % RMS errors
    rmsErrorX = sqrt(mean(errorsX.^2));
    rmsErrorY = sqrt(mean(errorsY.^2));
    rmsEuclideanError = sqrt(mean(euclideanErrors.^2));
    
    % Mean absolute errors
    meanAbsErrorX = mean(abs(errorsX));
    meanAbsErrorY = mean(abs(errorsY));
    
    % Correlation between X and Y errors
    correlation = corrcoef(errorsX, errorsY);
    xyCorrelation = correlation(1,2);
    
    fprintf('✓ Error metrics computation completed\n');
    
catch ME
    fprintf('✗ Error computing metrics: %s\n', ME.message);
    return;
end

%% DISPLAY DETAILED RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('DETAILED ERROR ANALYSIS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

fprintf('Sample Information:\n');
fprintf('  Total samples: %d\n', numberElements);
fprintf('  Valid predictions: %d\n', validPredictions);
fprintf('  Failed predictions: %d\n', numberElements - validPredictions);

if validPredictions > 0
    % Display sample predictions and ground truth
    fprintf('\nSample Data (first 3 samples):\n');
    for i = 1:min(3, numberElements)
        fprintf('  Sample %d:\n', i);
        fprintf('    Ground Truth: [%.2f, %.2f]\n', groundTruth(1,i), groundTruth(2,i));
        fprintf('    Prediction:   [%.2f, %.2f]\n', prediction(1,i), prediction(2,i));
        fprintf('    Error:        [%.2f, %.2f]\n', errors(1,i), errors(2,i));
        fprintf('    Euclidean:    %.2f\n', euclideanErrors(i));
    end
    
    fprintf('\nX Component Error Statistics (Absolute Values):\n');
    fprintf('  Mean:           %.6f\n', meanErrorX);
    fprintf('  Median:         %.6f\n', medianErrorX);
    fprintf('  Min:            %.6f\n', minErrorX);
    fprintf('  Max:            %.6f\n', maxErrorX);
    fprintf('  95th quantile:  %.6f\n', quantile95ErrorX);
    fprintf('  99th quantile:  %.6f\n', quantile99ErrorX);
    
    fprintf('\nY Component Error Statistics (Absolute Values):\n');
    fprintf('  Mean:           %.6f\n', meanErrorY);
    fprintf('  Median:         %.6f\n', medianErrorY);
    fprintf('  Min:            %.6f\n', minErrorY);
    fprintf('  Max:            %.6f\n', maxErrorY);
    fprintf('  95th quantile:  %.6f\n', quantile95ErrorY);
    fprintf('  99th quantile:  %.6f\n', quantile99ErrorY);
    
    fprintf('\nEuclidean Distance Error Statistics:\n');
    fprintf('  Mean:           %.6f\n', meanEuclideanError);
    fprintf('  Median:         %.6f\n', medianEuclideanError);
    fprintf('  Min:            %.6f\n', minEuclideanError);
    fprintf('  Max:            %.6f\n', maxEuclideanError);
    fprintf('  95th quantile:  %.6f\n', quantile95EuclideanError);
    fprintf('  99th quantile:  %.6f\n', quantile99EuclideanError);
    
    fprintf('\nAdditional Metrics:\n');
    fprintf('  RMS Error X:           %.6f\n', rmsErrorX);
    fprintf('  RMS Error Y:           %.6f\n', rmsErrorY);
    fprintf('  RMS Euclidean Error:   %.6f\n', rmsEuclideanError);
    fprintf('  Mean Absolute Error X: %.6f\n', meanAbsErrorX);
    fprintf('  Mean Absolute Error Y: %.6f\n', meanAbsErrorY);
    fprintf('  X-Y Error Correlation: %.6f\n', xyCorrelation);
    
    % Performance metrics
    fprintf('\nPerformance Metrics:\n');
    fprintf('  Total simulation time:     %.2f seconds\n', simulationTime);
    fprintf('  Average time per sample:   %.4f seconds\n', simulationTime/numberElements);
    fprintf('  Samples per second:        %.1f\n', numberElements/simulationTime);
    
else
    fprintf('\n✗ No valid predictions to analyze!\n');
end

%% SAVE RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('SAVING RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Create results structure
simulinkResults = struct();
simulinkResults.numberElements = numberElements;
simulinkResults.validPredictions = validPredictions;
simulinkResults.simulationTime = simulationTime;
simulinkResults.prediction = prediction;
simulinkResults.errors = errors;
simulinkResults.euclideanErrors = euclideanErrors;
simulinkResults.meanError = [meanErrorX, meanErrorY];
simulinkResults.medianError = [medianErrorX, medianErrorY];
simulinkResults.minError = [minErrorX, minErrorY];
simulinkResults.maxError = [maxErrorX, maxErrorY];
simulinkResults.quantile95Error = [quantile95ErrorX, quantile95ErrorY];
simulinkResults.quantile99Error = [quantile99ErrorX, quantile99ErrorY];
simulinkResults.meanEuclideanError = meanEuclideanError;
simulinkResults.medianEuclideanError = medianEuclideanError;
simulinkResults.minEuclideanError = minEuclideanError;
simulinkResults.maxEuclideanError = maxEuclideanError;
simulinkResults.quantile95EuclideanError = quantile95EuclideanError;
simulinkResults.quantile99EuclideanError = quantile99EuclideanError;
simulinkResults.rmsErrorX = rmsErrorX;
simulinkResults.rmsErrorY = rmsErrorY;
simulinkResults.rmsEuclideanError = rmsEuclideanError;
simulinkResults.meanAbsErrorX = meanAbsErrorX;
simulinkResults.meanAbsErrorY = meanAbsErrorY;
simulinkResults.xyCorrelation = xyCorrelation;

% Save results to .mat file
resultsFileName = 'simulink_model_test_results.mat';
save(resultsFileName, 'simulinkResults', 'prediction', 'errors', 'euclideanErrors', ...
     'groundTruth', 'simulationTime', 'numberElements', 'resx', 'resy');
fprintf('Results saved to: %s\n', resultsFileName);

%% FINAL SUMMARY
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('FINAL SUMMARY\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

if validPredictions == numberElements
    fprintf('✓ All samples processed successfully!\n');
    fprintf('✓ Simulink model validation completed.\n');
    
    % Provide key metrics summary
    fprintf('\nKey Performance Indicators:\n');
    fprintf('  Mean Euclidean Error: %.2f pixels\n', meanEuclideanError);
    fprintf('  95th Percentile Error: %.2f pixels\n', quantile95EuclideanError);
    fprintf('  Processing Rate: %.1f samples/second\n', numberElements/simulationTime);
    
else
    fprintf('⚠ Some samples failed processing (%d/%d failed).\n', ...
            numberElements-validPredictions, numberElements);
    fprintf('Review the detailed results above for more information.\n');
end

fprintf('\nFiles created:\n');
fprintf('  - %s (Detailed results data)\n', resultsFileName);

fprintf('\n✓ Simulink model testing completed.\n');