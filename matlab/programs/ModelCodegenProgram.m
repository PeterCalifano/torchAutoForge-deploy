close all
clear
clc

%% Settings
charProgramRootPath = fileparts(mfilename('fullpath'));
charRepositoryRootPath = fileparts(fileparts(charProgramRootPath));
charPathToModelMat = fullfile(charRepositoryRootPath, "models", "matlab", ...
                              "zealous_cow_471_epoch_2893_0.mat");
cellInputArgs = {ones(12,1,'single')};
charForwardFcnFilename = "InferNeuralCOB";
charOutputPath = "./model_codegen_output";

%% Call code generation function
MakeModelCodegen(charPathToModelMat, ...
                cellInputArgs, ...
                charForwardFcnFilename, ...
                charOutputPath, ...
                "bRunModelAnalysisCheck", false);
