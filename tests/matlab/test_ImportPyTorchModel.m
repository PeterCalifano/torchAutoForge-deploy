% Settings
charModelfilePath  = "/home/peterc/devDir/nav-frontend/" + ...
    ".experimental/neuralCOB/checkpoints/agreeable-steed-344_epoch_11973.onnx"; % Must be a traced pytorch model (.pt) 
dModelInputSizes = [1, 11]; % Input size of the model

try
    ImportPyTorchModel(charModelfilePath, dModelInputSizes);
catch ME
    fprintf("Error: %s\nOccurred in: %s (line %d)\n", ME.message, ME.stack(1).file, ME.stack(1).line);
end

try
    ImportONNxModel(charModelfilePath, dModelInputSizes);
catch ME
    fprintf("Error: %s\nOccurred in: %s (line %d)\n", ME.message, ME.stack(1).file, ME.stack(1).line);
end
