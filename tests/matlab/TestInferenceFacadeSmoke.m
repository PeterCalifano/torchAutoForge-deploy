function TestInferenceFacadeSmoke(sWrapperDir, sMexDir, sModelPath, ...
        sCentroidConfigPath, sObjectDetectionConfigPath)
% TestInferenceFacadeSmoke Validate target-owned MATLAB inference bindings.
%
% SIGNATURE
%   TestInferenceFacadeSmoke(sWrapperDir, sMexDir, sModelPath, ...
%       sCentroidConfigPath, sObjectDetectionConfigPath)
%
% DESCRIPTION
%   Loads the build-tree MATLAB wrapper, runs the repository inference
%   fixture through the manager and role facade, and verifies that a rejected
%   input does not invalidate the loaded manager.
%
% INPUT
%   sWrapperDir                Generated MATLAB wrapper directory.
%   sMexDir                    Generated MEX binary directory.
%   sModelPath                 Repository ONNX fixture path.
%   sCentroidConfigPath        Centroiding role manifest path.
%   sObjectDetectionConfigPath Object-detection role manifest path.
%
% OUTPUT
%   None. Contract violations raise MATLAB assertions or wrapper exceptions.
%
% CHANGELOG
%   2026-08-26  Move target-owned smoke behavior out of CMake.
%
% DEPENDENCIES
%   Generated ptafdeploy MATLAB wrapper and MEX module.

arguments (Input)
    sWrapperDir (1, 1) string
    sMexDir (1, 1) string
    sModelPath (1, 1) string
    sCentroidConfigPath (1, 1) string
    sObjectDetectionConfigPath (1, 1) string
end

addpath(sWrapperDir);
addpath(sMexDir);

oManager = ptafdeploy.inference.CInferenceManager(char(sModelPath));
assert(oManager.GetNumInputs() == 1);
oInputInfo = oManager.GetInputInfo(0);
assert(strcmp(oInputInfo.name, 'input'));

vdInput = double((1:11)');
vdShape = double([1, 11]');
vdOutput = ptafdeploy.inference.InferSingleFloatInputVector( ...
    oManager, vdInput, vdShape);
assert(numel(vdOutput) == 2);

oCentroidModel = ptafdeploy.inference.CModelFacade();
oCentroidModel.LoadModelConfig(char(sCentroidConfigPath));
assert(strcmp(oCentroidModel.GetRole(), 'centroiding'));
vdCentroidOutput = ptafdeploy.inference.InferModelSingleFloatInputVector( ...
    oCentroidModel, vdInput, vdShape);
assert(numel(vdCentroidOutput) == 2);

oObjectModel = ptafdeploy.inference.CModelFacade();
oObjectModel.LoadModelConfig(char(sObjectDetectionConfigPath));
assert(strcmp(oObjectModel.GetRole(), 'object_detection'));
assert(oObjectModel.GetNumInputs() == 1);
VerifyGenericFeatureAdapter_();
VerifyGenericDetectionAdapter_();

bCaughtExpectedError = false;
try
    ptafdeploy.inference.InferSingleFloatInputVector( ...
        oManager, double((1:10)'), vdShape);
catch
    bCaughtExpectedError = true;
end
assert(bCaughtExpectedError);

vdOutputAfterError = ptafdeploy.inference.InferSingleFloatInputVector( ...
    oManager, vdInput, vdShape);
assert(numel(vdOutputAfterError) == 2);

clear oManager oCentroidModel oObjectModel oInputInfo;
clear vdOutput vdCentroidOutput vdOutputAfterError;
clear mex; %#ok<CLMEX> Release wrapper state before the CTest process exits.
end


function VerifyGenericFeatureAdapter_()
% Verify generic feature decoding through the MATLAB numeric bridge.
oOutput = ptafdeploy.inference.MakeFloatTensorFromVector( ...
    'features', double([0.25, 0.75]'), double([1, 2]'));
oSchema = ptafdeploy.inference.SFeatureRowSchema();

mFeatures = ptafdeploy.inference.DecodeFeatureRowsMatrix(oOutput, oSchema);
assert(isequal(size(mFeatures), [1, 3]));
assert(abs(mFeatures(1, 1) - 0.25) < 1.0e-6);
assert(abs(mFeatures(1, 2) - 0.75) < 1.0e-6);
assert(abs(mFeatures(1, 3) - 1.0) < 1.0e-6);

clear oOutput oSchema mFeatures;
end


function VerifyGenericDetectionAdapter_()
% Verify schema-driven decoding through the MATLAB numeric bridge.
oOutput = ptafdeploy.inference.MakeFloatTensorFromVector( ...
    'detections', double([10, 20, 30, 40, 0.9, 0.1, 0.8]'), double([1, 1, 7]'));
oSchema = ptafdeploy.inference.SDetectionRowSchema();
oSchema.objectness_index = int64(4);
oSchema.first_class_score_index = uint64(5);
oSchema.class_score_count = uint64(2);

mDetections = ptafdeploy.inference.DecodeDetectionRowsMatrix( ...
    oOutput, oSchema, 0.5, uint64(0));
assert(isequal(size(mDetections), [1, 6]));
assert(abs(mDetections(1, 1) - 10.0) < 1.0e-6);
assert(abs(mDetections(1, 5) - 0.72) < 1.0e-6);
assert(mDetections(1, 6) == 1);

clear oOutput oSchema mDetections;
end
