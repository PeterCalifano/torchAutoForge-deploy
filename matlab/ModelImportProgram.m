close all
clear
clc

%% Options
cd( fileparts(which(mfilename)) )
charOnnxFileRootPath = "/home/peterc/devDir/ML-repos/torchAutoForge-deploy/.model_checkpoints/onnx_exported_models";
charOnnxFileName = "thoughtful-shark-599_epoch_4956_cpu";
charONNxModelFilePath = fullfile(charOnnxFileRootPath, sprintf("%s.onnx", charOnnxFileName));

charModelName = matlab.lang.makeValidName(charOnnxFileName);

dModelInputSizes = [1, 12]; % Must match the input format
charInputShape  = 'BC';
charOutputShape = 'BC';
dInputSample = randn(dModelInputSizes);
% dOutputSample % If expected label is available


objModel = ImportONNxModel(charONNxModelFilePath, ...
                            dModelInputSizes, ...
                            charInputShape, ...
                            charOutputShape, ...
                            dInputSample, ...
                            "charModelName", charModelName);
