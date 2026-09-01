close all
clear
clc

%% Options
charProgramRootPath = fileparts(mfilename('fullpath'));
cd(charProgramRootPath)

charRepositoryRootPath = fileparts(fileparts(charProgramRootPath));
charOnnxFileRootPath = fullfile(charRepositoryRootPath, "models", "onnx");
charMatlabFileRootPath = fullfile(charRepositoryRootPath, "models", "matlab");

charOnnxFileName = "zealous-cow-471_epoch_2893_0";

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
                            "charModelName", charModelName, ...
                            "charOutputPath", charMatlabFileRootPath);
