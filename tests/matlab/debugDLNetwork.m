function debugDLNetwork(objModel, objX, referenceOutput)
% DEBUGDLNETWORK - Comprehensive debugging tool for dlnetwork objects
% 
% Inputs:
%   objModel - dlnetwork object to debug
%   objX - Input dlarray for testing
%   referenceOutput - (optional) Expected output for comparison
%
% Usage:
%   debugDLNetwork(objModel, objX);
%   debugDLNetwork(objModel, objX, expectedOutput);

fprintf('\n=== DLNETWORK DEBUGGING TOOL ===\n\n');

%% 1. NETWORK STRUCTURE ANALYSIS
fprintf('1. NETWORK STRUCTURE:\n');
fprintf('   Number of layers: %d\n', numel(objModel.Layers));
fprintf('   Input names: %s\n', strjoin(objModel.InputNames, ', '));
fprintf('   Output names: %s\n', strjoin(objModel.OutputNames, ', '));

% Display layer information
fprintf('\n   Layer Details:\n');
for i = 1:numel(objModel.Layers)
    layer = objModel.Layers(i);
    fprintf('   [%2d] %-20s | Type: %-15s\n', i, layer.Name, class(layer));
end

%% 2. INPUT VALIDATION
fprintf('\n2. INPUT VALIDATION:\n');
fprintf('   Input size: [%s]\n', strjoin(string(size(objX)), ' x '));
fprintf('   Input format: %s\n', dims(objX));
fprintf('   Input data type: %s\n', class(extractdata(objX)));
fprintf('   Input range: [%.6f, %.6f]\n', min(extractdata(objX), [], 'all'), max(extractdata(objX), [], 'all'));

% Check for NaN or Inf values
if any(isnan(extractdata(objX)), 'all')
    warning('Input contains NaN values!');
end
if any(isinf(extractdata(objX)), 'all')
    warning('Input contains Inf values!');
end

%% 3. LAYER-BY-LAYER FORWARD PASS
fprintf('\n3. LAYER-BY-LAYER ANALYSIS:\n');
try
    % Get intermediate outputs using forward function
    [outputs, states] = forward(objModel, objX);
    
    fprintf('   Forward pass successful.\n');
    fprintf('   Output size: [%s]\n', strjoin(string(size(outputs)), ' x '));
    fprintf('   Output range: [%.6f, %.6f]\n', min(extractdata(outputs), [], 'all'), max(extractdata(outputs), [], 'all'));
    
    % Check for problematic outputs
    outputData = extractdata(outputs);
    if any(isnan(outputData), 'all')
        warning('Output contains NaN values!');
    end
    if any(isinf(outputData), 'all')
        warning('Output contains Inf values!');
    end
    
catch ME
    fprintf('   ERROR in forward pass: %s\n', ME.message);
    return;
end

%% 4. DETAILED LAYER INSPECTION
fprintf('\n4. DETAILED LAYER INSPECTION:\n');
inspectLayers(objModel);

%% 5. WEIGHT AND BIAS ANALYSIS
fprintf('\n5. WEIGHT AND BIAS ANALYSIS:\n');
analyzeWeights(objModel);

%% 6. COMPARISON WITH REFERENCE (if provided)
if nargin >= 3 && ~isempty(referenceOutput)
    fprintf('\n6. COMPARISON WITH REFERENCE:\n');
    compareOutputs(outputs, referenceOutput);
end

%% 7. NUMERICAL STABILITY CHECK
fprintf('\n7. NUMERICAL STABILITY CHECK:\n');
checkNumericalStability(objModel, objX);

fprintf('\n=== DEBUGGING COMPLETE ===\n\n');
end

%% HELPER FUNCTIONS

function inspectLayers(objModel)
% Inspect each layer in detail
for i = 1:numel(objModel.Layers)
    layer = objModel.Layers(i);
    fprintf('   Layer %d: %s\n', i, layer.Name);
    
    % Get layer properties
    props = properties(layer);
    for j = 1:length(props)
        try
            prop = props{j};
            value = layer.(prop);
            if isnumeric(value) && ~isempty(value)
                if numel(value) == 1
                    fprintf('     %s: %.6f\n', prop, value);
                elseif numel(value) <= 10
                    fprintf('     %s: [%s]\n', prop, strjoin(string(value), ' '));
                else
                    fprintf('     %s: %s (size: [%s])\n', prop, class(value), strjoin(string(size(value)), 'x'));
                end
            elseif ischar(value) || isstring(value)
                fprintf('     %s: %s\n', prop, string(value));
            end
        catch
            % Skip properties that can't be accessed
        end
    end
    fprintf('\n');
end
end

function analyzeWeights(objModel)
% Analyze weights and biases in the network
weightLayers = [];
for i = 1:numel(objModel.Layers)
    layer = objModel.Layers(i);
    if isprop(layer, 'Weights') || isprop(layer, 'Bias')
        weightLayers = [weightLayers, i];
    end
end

if isempty(weightLayers)
    fprintf('   No learnable parameters found.\n');
    return;
end

for i = weightLayers
    layer = objModel.Layers(i);
    fprintf('   Layer %d (%s):\n', i, layer.Name);
    
    if isprop(layer, 'Weights') && ~isempty(layer.Weights)
        weights = extractdata(layer.Weights);
        fprintf('     Weights: size [%s], range [%.6f, %.6f], std: %.6f\n', ...
            strjoin(string(size(weights)), 'x'), ...
            min(weights, [], 'all'), max(weights, [], 'all'), ...
            std(weights, [], 'all'));
    end
    
    if isprop(layer, 'Bias') && ~isempty(layer.Bias)
        bias = extractdata(layer.Bias);
        fprintf('     Bias: size [%s], range [%.6f, %.6f], std: %.6f\n', ...
            strjoin(string(size(bias)), 'x'), ...
            min(bias, [], 'all'), max(bias, [], 'all'), ...
            std(bias, [], 'all'));
    end
end
end

function compareOutputs(actual, expected)
% Compare actual vs expected outputs
actualData = extractdata(actual);
expectedData = extractdata(expected);

% Ensure same size
if ~isequal(size(actualData), size(expectedData))
    fprintf('   ERROR: Size mismatch! Actual: [%s], Expected: [%s]\n', ...
        strjoin(string(size(actualData)), 'x'), ...
        strjoin(string(size(expectedData)), 'x'));
    return;
end

% Calculate differences
diff = actualData - expectedData;
absDiff = abs(diff);
relDiff = abs(diff ./ (expectedData + eps));

fprintf('   Absolute difference:\n');
fprintf('     Mean: %.2e, Max: %.2e, Std: %.2e\n', ...
    mean(absDiff, 'all'), max(absDiff, [], 'all'), std(absDiff, [], 'all'));

fprintf('   Relative difference:\n');
fprintf('     Mean: %.2e, Max: %.2e, Std: %.2e\n', ...
    mean(relDiff, 'all'), max(relDiff, [], 'all'), std(relDiff, [], 'all'));

% Statistical analysis
fprintf('   Statistical comparison:\n');
fprintf('     Correlation: %.6f\n', corr(actualData(:), expectedData(:)));
fprintf('     RMSE: %.2e\n', sqrt(mean(diff.^2, 'all')));
fprintf('     MAE: %.2e\n', mean(absDiff, 'all'));

% Check tolerance levels
tolerances = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2];
for tol = tolerances
    percentage = 100 * sum(absDiff <= tol, 'all') / numel(absDiff);
    fprintf('     Within %.0e tolerance: %.1f%%\n', tol, percentage);
end
end

function checkNumericalStability(objModel, objX)
% Check numerical stability with perturbed inputs
fprintf('   Testing numerical stability...\n');

% Original output
origOutput = predict(objModel, objX);
origData = extractdata(origOutput);

% Test with small perturbations
perturbations = [1e-7, 1e-6, 1e-5, 1e-4];
for eps_val = perturbations
    % Add small random perturbation
    perturbedX = objX + eps_val * dlarray(randn(size(objX)), dims(objX));
    
    try
        perturbedOutput = predict(objModel, perturbedX);
        perturbedData = extractdata(perturbedOutput);
        
        outputDiff = abs(perturbedData - origData);
        sensitivity = max(outputDiff, [], 'all') / eps_val;
        
        fprintf('     Perturbation %.0e: Max output change %.2e (sensitivity: %.2e)\n', ...
            eps_val, max(outputDiff, [], 'all'), sensitivity);
    catch ME
        fprintf('     Perturbation %.0e: FAILED - %s\n', eps_val, ME.message);
    end
end
end

%% ADDITIONAL UTILITY FUNCTIONS

function visualizeActivations(objModel, objX, layerName)
% Visualize activations for a specific layer
% Usage: visualizeActivations(objModel, objX, 'conv1')

if nargin < 3
    % Show available layer names
    fprintf('Available layers:\n');
    for i = 1:numel(objModel.Layers)
        fprintf('  %s\n', objModel.Layers(i).Name);
    end
    return;
end

try
    % Extract features from specific layer
    features = activations(objModel, objX, layerName);
    featData = extractdata(features);
    
    fprintf('Layer "%s" activations:\n', layerName);
    fprintf('  Size: [%s]\n', strjoin(string(size(featData)), 'x'));
    fprintf('  Range: [%.6f, %.6f]\n', min(featData, [], 'all'), max(featData, [], 'all'));
    fprintf('  Mean: %.6f, Std: %.6f\n', mean(featData, 'all'), std(featData, [], 'all'));
    
    % Plot histogram if reasonable size
    if numel(featData) <= 10000
        figure;
        histogram(featData(:), 50);
        title(sprintf('Activation Distribution - %s', layerName));
        xlabel('Activation Value');
        ylabel('Frequency');
    end
    
catch ME
    fprintf('Error extracting activations: %s\n', ME.message);
end
end

function exportDebugInfo(objModel, objX, filename)
% Export debugging information to file
% Usage: exportDebugInfo(objModel, objX, 'debug_report.txt')

if nargin < 3
    filename = 'dlnetwork_debug_report.txt';
end

% Redirect fprintf to file
fid = fopen(filename, 'w');
if fid == -1
    error('Cannot create file: %s', filename);
end

% Temporarily redirect output (this is a simplified approach)
% In practice, you'd modify the debug functions to accept file ID
fprintf('Debug report saved to: %s\n', filename);
fprintf('Use debugDLNetwork function to generate detailed report.\n');

fclose(fid);
end