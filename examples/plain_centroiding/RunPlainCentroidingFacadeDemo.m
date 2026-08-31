function stResult = RunPlainCentroidingFacadeDemo(strModelPath, strImagePath, stOptions)
% RUNPLAINCENTROIDINGFACADEDEMO Run image-only centroiding through the MATLAB wrapper.
%
% SIGNATURE
%   stResult = RunPlainCentroidingFacadeDemo(strModelPath, strImagePath)
%   stResult = RunPlainCentroidingFacadeDemo(..., strTarget="cpu")
%
% DESCRIPTION
%   Loads a centroiding .ptafmodel manifest or raw ONNX artifact through
%   CModelFacade, converts one image to grayscale NCHW through the generic
%   wrapper bridge, and maps the normalized feature output into model-input
%   and original-image pixel coordinates.
%
% INPUT
%   strModelPath          Path to a centroiding manifest or raw ONNX model.
%   strImagePath          Path to an image readable by IMREAD.
%   stOptions.strTarget   "manifest", "cpu", or "cuda".
%   stOptions.ui32DeviceId
%                         Non-negative runtime device index.
%
% OUTPUT
%   stResult              Struct containing contract, tensor-shape, and
%                         centroid-coordinate fields.
%
% CHANGELOG
%   2026-09-01 Initial wrapper-backed plain-centroiding demo.
%
% DEPENDENCIES
%   Generated ptafdeploy MATLAB wrapper, Image Processing Toolbox.

arguments (Input)
    strModelPath (1, 1) string
    strImagePath (1, 1) string
    stOptions.strTarget (1, 1) string {mustBeMember(stOptions.strTarget, ...
        ["manifest", "cpu", "cuda"])} = "manifest"
    stOptions.ui32DeviceId (1, 1) uint32 = uint32(0)
end

arguments (Output)
    stResult (1, 1) struct
end

if ~isfile(strModelPath)
    error("ptafdeploy:PlainCentroidingDemo:MissingModel", ...
        "Missing model or manifest: %s", strModelPath);
end
if ~isfile(strImagePath)
    error("ptafdeploy:PlainCentroidingDemo:MissingImage", ...
        "Missing input image: %s", strImagePath);
end

% Load only backend-neutral facade and metadata types through the wrapper.
oModel = LoadPlainCentroidingModel_( ...
    strModelPath, stOptions.strTarget, stOptions.ui32DeviceId);
if ~strcmp(oModel.GetRole(), "centroiding")
    error("ptafdeploy:PlainCentroidingDemo:Role", ...
        "Plain centroiding requires role=centroiding.");
end
if oModel.GetNumInputs() ~= 1 || oModel.GetNumOutputs() ~= 1
    error("ptafdeploy:PlainCentroidingDemo:Cardinality", ...
        "Plain centroiding requires exactly one input and one output.");
end

% Apply model-specific image policy on the integration side of the facade.
stInputInfo = oModel.GetInputInfo(0);
vModelInputShape = ptafdeploy.inference.GetModelInputShapeVector(oModel, 0);
[oInput, vImageSize] = PreparePlainCentroidingInput_( ...
    strImagePath, stInputInfo, vModelInputShape);
vInputShape = ptafdeploy.inference.GetFloatTensorShapeVector(oInput);
oOutput = oModel.InferSingleFloatTensor(oInput);
vOutputShape = ptafdeploy.inference.GetFloatTensorShapeVector(oOutput);

% Decode only generic feature rows in C++; coordinate policy remains local.
mFeatures = DecodePlainCentroidingOutput_(oOutput, vOutputShape);
vNormalized = mFeatures(1, 1:2).';
vModelInputSize = [vInputShape(4); vInputShape(3)];
vModelInputPixels = vNormalized .* vModelInputSize;
vOriginalPixels = vNormalized .* vImageSize;

stResult = struct( ...
    "strRole", string(oModel.GetRole()), ...
    "strBackend", string(oModel.GetBackendDetail()), ...
    "strInputName", string(stInputInfo.name), ...
    "vInputShape", vInputShape, ...
    "vImageSize", vImageSize, ...
    "strOutputName", string(oOutput.name), ...
    "vOutputShape", vOutputShape, ...
    "vNormalized", vNormalized, ...
    "vModelInputPixels", vModelInputPixels, ...
    "vOriginalPixels", vOriginalPixels);

PrintPlainCentroidingSummary_(stResult);
end


function oModel = LoadPlainCentroidingModel_(strModelPath, strTarget, ui32DeviceId)
% Load a role manifest or raw ONNX with an optional strict runtime override.
arguments (Input)
    strModelPath (1, 1) string
    strTarget (1, 1) string
    ui32DeviceId (1, 1) uint32
end

[~, ~, strExtension] = fileparts(strModelPath);
oModel = ptafdeploy.inference.CModelFacade();
bOverrideRuntime = strTarget ~= "manifest";

if strcmpi(strExtension, ".ptafmodel")
    if bOverrideRuntime
        oModel.LoadModelConfigWithRuntimeConfig( ...
            char(strModelPath), MakeRuntime_(strTarget, ui32DeviceId));
    else
        oModel.LoadModelConfig(char(strModelPath));
    end
elseif strcmpi(strExtension, ".onnx")
    if bOverrideRuntime
        oModel.LoadModelWithRoleAndRuntimeConfig( ...
            char(strModelPath), ptafdeploy.inference.EModelRole.centroiding, ...
            MakeRuntime_(strTarget, ui32DeviceId));
    else
        oModel.LoadModelWithRole( ...
            char(strModelPath), ptafdeploy.inference.EModelRole.centroiding);
    end
else
    error("ptafdeploy:PlainCentroidingDemo:Artifact", ...
        "Expected a .ptafmodel manifest or .onnx artifact.");
end
end


function oRuntime = MakeRuntime_(strTarget, ui32DeviceId)
% Construct a strict backend-neutral runtime override.
arguments (Input)
    strTarget (1, 1) string {mustBeMember(strTarget, ["cpu", "cuda"])}
    ui32DeviceId (1, 1) uint32
end

oRuntime = ptafdeploy.inference.SRuntimeConfig();
oRuntime.SetDeviceId(double(ui32DeviceId));
oRuntime.ClearExecutionTargetPriority();
if strTarget == "cpu"
    oRuntime.AddExecutionTarget(ptafdeploy.inference.EExecutionTarget.cpu);
else
    oRuntime.AddExecutionTarget(ptafdeploy.inference.EExecutionTarget.cuda);
end
oRuntime.SetAllowFallback(false);
end


function [oInput, vImageSize] = PreparePlainCentroidingInput_( ...
        strImagePath, stInputInfo, vInputShape)
% Convert one image to uint8 grayscale and row-major NCHW wrapper storage.
arguments (Input)
    strImagePath (1, 1) string
    stInputInfo (1, 1) ptafdeploy.inference.STensorInfo
    vInputShape (:, 1) double
end

if ~strcmp(stInputInfo.dtype, "float32") || numel(vInputShape) ~= 4 || ...
        ~ismember(vInputShape(1), [-1, 1]) || vInputShape(2) ~= 1 || ...
        any(vInputShape(3:4) <= 0) || any(vInputShape ~= fix(vInputShape))
    error("ptafdeploy:PlainCentroidingDemo:InputContract", ...
        "Plain centroiding expects float32 input [N,1,H,W] with concrete H/W.");
end

mImage = imread(strImagePath);
if size(mImage, 3) == 1
    mGrayscaleImage = mImage;
else
    mGrayscaleImage = im2gray(mImage(:, :, 1:3));
end
mGrayscaleImage = im2uint8(mGrayscaleImage);
vImageSize = double([size(mGrayscaleImage, 2); size(mGrayscaleImage, 1)]);

uiHeight = uint64(vInputShape(3));
uiWidth = uint64(vInputShape(4));
mResizedImage = imresize( ...
    mGrayscaleImage, [double(uiHeight), double(uiWidth)], "bilinear", ...
    "Antialiasing", false);

% Transposition makes MATLAB linearization match row-major one-channel HWC.
vHwcValues = double(reshape(mResizedImage.', [], 1));
oInput = ptafdeploy.inference.MakeNchwFloatTensorFromHwcVector( ...
    stInputInfo.name, vHwcValues, uiHeight, uiWidth, uint64(1), ...
    1.0 / 255.0, false);
end


function mFeatures = DecodePlainCentroidingOutput_(oOutput, vOutputShape)
% Validate the plain output contract and invoke generic feature decoding.
arguments (Input)
    oOutput (1, 1) ptafdeploy.inference.SFloatTensor
    vOutputShape (:, 1) double
end

if ~isequal(vOutputShape, double([1, 2]'))
    error("ptafdeploy:PlainCentroidingDemo:OutputContract", ...
        "Plain centroiding expects output shape [1,2].");
end

oSchema = ptafdeploy.inference.SFeatureRowSchema();
mFeatures = ptafdeploy.inference.DecodeFeatureRowsMatrix(oOutput, oSchema);
if ~isequal(size(mFeatures), [1, 3]) || any(~isfinite(mFeatures(1, 1:2))) || ...
        any(mFeatures(1, 1:2) < 0.0) || any(mFeatures(1, 1:2) > 1.0)
    error("ptafdeploy:PlainCentroidingDemo:OutputValues", ...
        "Plain centroiding must produce one finite normalized feature within [0,1].");
end
end


function PrintPlainCentroidingSummary_(stResult)
% Print stable fields shared with the native and Python demos.
arguments (Input)
    stResult (1, 1) struct
end

fprintf("role=%s\n", stResult.strRole);
fprintf("backend=%s\n", stResult.strBackend);
fprintf("input_name=%s\n", stResult.strInputName);
fprintf("input_shape=[%s]\n", strjoin(string(stResult.vInputShape.'), ","));
fprintf("image_size=[%s]\n", strjoin(string(stResult.vImageSize.'), ","));
fprintf("output_name=%s\n", stResult.strOutputName);
fprintf("output_shape=[%s]\n", strjoin(string(stResult.vOutputShape.'), ","));
fprintf("centroid.normalized_x=%.9g\n", stResult.vNormalized(1));
fprintf("centroid.normalized_y=%.9g\n", stResult.vNormalized(2));
fprintf("centroid.model_input_x_px=%.9g\n", stResult.vModelInputPixels(1));
fprintf("centroid.model_input_y_px=%.9g\n", stResult.vModelInputPixels(2));
fprintf("centroid.original_x_px=%.9g\n", stResult.vOriginalPixels(1));
fprintf("centroid.original_y_px=%.9g\n", stResult.vOriginalPixels(2));
end
