close all
clear
clc

% TEST SETUP
% Settings
charTracedModelfilePath  = "testSamples/tracedSampleModel.pt"; % Must be a traced pytorch model (.pt) 
charONNxModelfilePath  = "testSamples/tracedSampleModel.onnx"; % Must be a traced pytorch model (.pt) 
dModelInputSizes = [1, 11]; % Input size of the model

%% test_ImportONNxModel
try
    ImportPyTorchModel(charTracedModelfilePath, dModelInputSizes);
catch ME
    fprintf(2, "Error: %s\nOccurred in: %s (line %d)\n", ME.message, ME.stack(1).file, ME.stack(1).line);
end

%% test_ImportONNxModel
try
    objModel = ImportONNxModel(charONNxModelfilePath, dModelInputSizes);
    assert(strcmpi(class(objModel), 'dlnewtork'));
catch ME
    fprintf(2, "Error: %s\nOccurred in: %s (line %d)\n", ME.message, ME.stack(1).file, ME.stack(1).line);
end

%% test_RunModelInference
objModel = ImportONNxModel(charONNxModelfilePath, dModelInputSizes);

dX = randn(dModelInputSizes);
fY = RunModelInference(objModel, dX, "charInputShapeFormat", "BC");

assert(all(size(fY) == [2, 1]), 'Incorrect output shape. Expected [1 2], found %s', mat2str(size(fY)))
