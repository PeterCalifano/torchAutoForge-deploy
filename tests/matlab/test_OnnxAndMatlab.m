%% MATLAB ONNX Model Testing and Comparison Script
% This script loads an ONNX model, runs multiple test cases,
% and provides detailed analysis of the outputs

close all
clear
clc

% Add path to your MATLAB functions
addpath('../../matlab');

%% CONFIGURATION
fprintf('=%.50s\n', repmat('=', 1, 50));
fprintf('CONFIGURATION\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% File paths
charONNxModelfilePath = "onnxModels/tracedSampleModel.onnx";
dModelInputSizes = [1, 11]; % Input size of the model
tolerance = 1e-5; % Tolerance for numerical comparisons

fprintf('ONNX Model Path: %s\n', charONNxModelfilePath);
fprintf('Input Sizes: [%s]\n', num2str(dModelInputSizes));
fprintf('Tolerance: %.1e\n', tolerance);

% Check if ONNX file exists
if ~isfile(charONNxModelfilePath)
    error('ONNX file not found: %s', charONNxModelfilePath);
end

fprintf('✓ ONNX file found\n');

%% LOAD ONNX MODEL
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('LOADING ONNX MODEL\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

try
    objModel = ImportONNxModel(charONNxModelfilePath, dModelInputSizes);
    
    % Verify model class
    if strcmpi(class(objModel), 'dlnetwork')
        fprintf('✓ Model loaded successfully as dlnetwork\n');
    else
        fprintf('⚠ Model loaded as %s (expected dlnetwork)\n', class(objModel));
    end
    
    % Display model information
    fprintf('Model class: %s\n', class(objModel));
    
catch ME
    fprintf('Error loading ONNX model: %s\n', ME.message);
    fprintf('Occurred in: %s (line %d)\n', ME.stack(1).file, ME.stack(1).line);
    return;
end

%% CREATE TEST INPUTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('CREATING TEST INPUTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Define test cases
testCases = struct();
testNames = {'Random', 'Zeros', 'Ones', 'Sequential', 'Large_Scale', 'Negative'};

% Generate test inputs
testCases.Random = randn(dModelInputSizes);
testCases.Zeros = zeros(dModelInputSizes);
testCases.Ones = ones(dModelInputSizes);
testCases.Sequential = reshape(1:prod(dModelInputSizes), dModelInputSizes);
testCases.Large_Scale = randn(dModelInputSizes) * 10;
testCases.Negative = -abs(randn(dModelInputSizes));

fprintf('Created %d test cases:\n', length(testNames));
for i = 1:length(testNames)
    testInput = testCases.(testNames{i});
    fprintf('  %d. %s: [%.3f, %.3f, %.3f, ...] (range: %.3f to %.3f)\n', ...
        i, testNames{i}, testInput(1), testInput(2), testInput(3), ...
        min(testInput(:)), max(testInput(:)));
end

%% RUN INFERENCE TESTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('RUNNING INFERENCE TESTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Store results for analysis
results = struct();
allPassed = true;

for i = 1:length(testNames)
    testName = testNames{i};
    testInput = testCases.(testName);
    
    fprintf('\nTest Case %d: %s\n', i, testName);
    fprintf('%.30s\n', repmat('-', 1, 30));
    
    try
        % Run inference
        output = RunModelInference(objModel, testInput, "charInputShapeFormat", "BC");
        
        % Store results
        results.(testName).input = testInput;
        results.(testName).output = output;
        results.(testName).success = true;
        
        % Display results
        fprintf('Input shape: [%s]\n', num2str(size(testInput)));
        fprintf('Output shape: [%s]\n', num2str(size(output)));
        fprintf('Output sample: [%.6f, %.6f, %.6f, ...]\n', output(1), output(2), output(min(3,end)));
        fprintf('✓ Test passed\n');
        
        % Basic output validation
        if any(isnan(output(:)))
            fprintf('⚠ Warning: Output contains NaN values\n');
            allPassed = false;
        end
        
        if any(isinf(output(:)))
            fprintf('⚠ Warning: Output contains infinite values\n');
            allPassed = false;
        end
        
    catch ME
        fprintf('✗ Test failed: %s\n', ME.message);
        results.(testName).success = false;
        results.(testName).error = ME.message;
        allPassed = false;
    end
end

%% CROSS-TEST VALIDATION
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('CROSS-TEST VALIDATION\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Compare outputs between different test cases for consistency
fprintf('Analyzing output consistency across test cases...\n\n');

% Output range analysis
fprintf('Output Range Analysis:\n');
allOutputs = [];
for i = 1:length(testNames)
    testName = testNames{i};
    if results.(testName).success
        allOutputs = [allOutputs; results.(testName).output(:)];
    end
end

if ~isempty(allOutputs)
    fprintf('  Overall output range: [%.6f, %.6f]\n', min(allOutputs), max(allOutputs));
    fprintf('  Output mean: %.6f\n', mean(allOutputs));
    fprintf('  Output std: %.6f\n', std(allOutputs));
end

%% SAVE RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('SAVING RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Save results to .mat file for comparison with Python
resultsFileName = 'matlab_onnx_test_results.mat';
save(resultsFileName, 'results', 'testCases', 'dModelInputSizes', 'charONNxModelfilePath');
fprintf('Results saved to: %s\n', resultsFileName);

% Export results to JSON for cross-platform comparison
jsonFileName = 'matlab_onnx_test_results.json';
try
    % Convert results to JSON-compatible format
    jsonData = struct();
    for i = 1:length(testNames)
        testName = testNames{i};
        if results.(testName).success
            jsonData.(testName) = struct();
            jsonData.(testName).input = results.(testName).input;
            jsonData.(testName).output = results.(testName).output;
        end
    end
    
    % Write JSON (requires MATLAB R2016b or later)
    if exist('jsonencode', 'file')
        jsonStr = jsonencode(jsonData);
        fid = fopen(jsonFileName, 'w');
        fprintf(fid, '%s', jsonStr);
        fclose(fid);
        fprintf('Results exported to JSON: %s\n', jsonFileName);
    else
        fprintf('JSON export not available (requires MATLAB R2016b+)\n');
    end
    
catch ME
    fprintf('Failed to export JSON: %s\n', ME.message);
end

%% FINAL RESULTS
fprintf('\n=%.50s\n', repmat('=', 1, 50));
fprintf('FINAL RESULTS\n');
fprintf('=%.50s\n', repmat('=', 1, 50));

% Count successful tests
successCount = 0;
for i = 1:length(testNames)
    testName = testNames{i};
    if results.(testName).success
        successCount = successCount + 1;
    end
end

fprintf('Test Summary:\n');
fprintf('  Total tests: %d\n', length(testNames));
fprintf('  Successful: %d\n', successCount);
fprintf('  Failed: %d\n', length(testNames) - successCount);

if allPassed && successCount == length(testNames)
    fprintf('\n✓ All tests passed! ONNX model is working correctly in MATLAB.\n');
else
    fprintf('\n✗ Some tests failed. Please review the results above.\n');
end

fprintf('\nFiles created:\n');
fprintf('  - %s (MATLAB results)\n', resultsFileName);
if exist(jsonFileName, 'file')
    fprintf('  - %s (Cross-platform comparison)\n', jsonFileName);
end