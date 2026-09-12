function stResult = RunYoloFacadeDemo(strManifestPath, strImagePath, stOptions)
% RUNYOLOFACADEDEMO Run YOLOv7 through the generated MATLAB facade wrapper.
%
% SIGNATURE
%   stResult = RunYoloFacadeDemo(strManifestPath, strImagePath)
%   stResult = RunYoloFacadeDemo(..., strTarget="cpu")
%
% DESCRIPTION
%   Loads a .ptafmodel manifest through CModelFacade, converts one RGB image
%   through the shared HWC-to-NCHW bridge, defines the YOLO row schema in this
%   integration, and routes generic detection decoding through the MATLAB-safe
%   adapter. Results are pre-NMS and no annotated image is rendered.
%
% INPUT
%   strManifestPath       Path to an object-detection .ptafmodel manifest.
%   strImagePath          Path to an image readable by IMREAD.
%   stOptions.strTarget   "manifest", "cpu", or "cuda".
%   stOptions.ui32DeviceId
%                         Non-negative CUDA device index.
%   stOptions.dScoreThreshold
%                         Finite non-negative objectness-class threshold.
%   stOptions.ui32MaxDetections
%                         Maximum score-sorted pre-NMS rows; zero keeps all.
%
% OUTPUT
%   stResult              Struct containing role/backend strings, input/output
%                         shapes, and N-by-6 [cx cy w h score class_id] rows.
%
% CHANGELOG
%   2026-08-25 Initial facade-backed YOLO demo.
%
% DEPENDENCIES
%   Generated autoforge_deploy MATLAB wrapper, Image Processing Toolbox.

arguments
    strManifestPath (1, 1) string
    strImagePath (1, 1) string
    stOptions.strTarget (1, 1) string {mustBeMember(stOptions.strTarget, ...
        ["manifest", "cpu", "cuda"])} = "manifest"
    stOptions.ui32DeviceId (1, 1) uint32 = uint32(0)
    stOptions.dScoreThreshold (1, 1) double {mustBeFinite, mustBeNonnegative} = 0.25
    stOptions.ui32MaxDetections (1, 1) uint32 = uint32(20)
end

if ~isfile(strManifestPath)
    error("ptafdeploy:YoloDemo:MissingManifest", ...
        "Missing model manifest: %s", strManifestPath);
end
if ~isfile(strImagePath)
    error("ptafdeploy:YoloDemo:MissingImage", ...
        "Missing input image: %s", strImagePath);
end

oModel = LoadYoloModel_(strManifestPath, stOptions.strTarget, stOptions.ui32DeviceId);
if ~strcmp(oModel.GetRole(), "object_detection")
    error("ptafdeploy:YoloDemo:Role", ...
        "YOLO demo requires role=object_detection in the manifest.");
end
if oModel.GetNumInputs() ~= 1 || oModel.GetNumOutputs() ~= 1
    error("ptafdeploy:YoloDemo:Cardinality", ...
        "YOLO demo currently requires one model input and one output.");
end

stInputInfo = oModel.GetInputInfo(0);
if ~strcmp(stInputInfo.dtype, "float32")
    error("ptafdeploy:YoloDemo:InputContract", ...
        "YOLO demo expects one float32 input with concrete shape [1,3,H,W].");
end
vInputShape = ptafdeploy.inference.GetModelInputShapeVector(oModel, 0);
oInput = PrepareYoloInput_(strImagePath, stInputInfo.name, vInputShape);
oOutput = oModel.InferSingleFloatTensor(oInput);
vOutputShape = ptafdeploy.inference.GetFloatTensorShapeVector(oOutput);
mDetections = DecodeYoloRows_(oOutput, vOutputShape, ...
    stOptions.dScoreThreshold, stOptions.ui32MaxDetections);

stResult = struct( ...
    "strRole", string(oModel.GetRole()), ...
    "strBackend", string(oModel.GetBackendDetail()), ...
    "vInputShape", vInputShape, ...
    "vOutputShape", vOutputShape, ...
    "mDetections", mDetections);

PrintYoloSummary_(stResult);
end


function mDetections = DecodeYoloRows_(oOutput, vOutputShape, ...
        dScoreThreshold, ui32MaxDetections)
% Define the YOLO row contract locally and invoke generic detection decoding.
arguments
    oOutput (1, 1) ptafdeploy.inference.SFloatTensor
    vOutputShape (:, 1) double
    dScoreThreshold (1, 1) double {mustBeFinite, mustBeNonnegative}
    ui32MaxDetections (1, 1) uint32
end

if isempty(vOutputShape) || vOutputShape(end) < 6 || ...
        vOutputShape(end) ~= fix(vOutputShape(end))
    error("ptafdeploy:YoloDemo:OutputContract", ...
        "YOLO demo expects output rows [cx,cy,w,h,objectness,class_scores...].");
end

oSchema = ptafdeploy.inference.SDetectionRowSchema();
oSchema.attribute_axis = int64(-1);
oSchema.box_encoding = ptafdeploy.inference.EBoundingBoxEncoding.center_xywh;
oSchema.box_coordinate_0_index = uint64(0);
oSchema.box_coordinate_1_index = uint64(1);
oSchema.box_coordinate_2_index = uint64(2);
oSchema.box_coordinate_3_index = uint64(3);
oSchema.objectness_index = int64(4);
oSchema.first_class_score_index = uint64(5);
oSchema.class_score_count = uint64(vOutputShape(end) - 5);
mDetections = ptafdeploy.inference.DecodeDetectionRowsMatrix( ...
    oOutput, oSchema, dScoreThreshold, ui32MaxDetections);
end


function oModel = LoadYoloModel_(strManifestPath, strTarget, ui32DeviceId)
% Load a manifest with an optional strict execution-target override.
arguments
    strManifestPath (1, 1) string
    strTarget (1, 1) string
    ui32DeviceId (1, 1) uint32
end

oModel = ptafdeploy.inference.CModelFacade();
if strTarget == "manifest"
    oModel.LoadModelConfig(char(strManifestPath));
    return
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
oModel.LoadModelConfigWithRuntimeConfig(char(strManifestPath), oRuntime);
end


function oInput = PrepareYoloInput_(strImagePath, strInputName, vInputShape)
% Resize RGB data and route HWC-to-NCHW conversion through the shared bridge.
arguments
    strImagePath (1, 1) string
    strInputName (1, :) char
    vInputShape (:, 1) double
end

if numel(vInputShape) ~= 4 || vInputShape(1) ~= 1 || vInputShape(2) ~= 3 || ...
        vInputShape(3) <= 0 || vInputShape(4) <= 0 || ...
        any(vInputShape ~= fix(vInputShape))
    error("ptafdeploy:YoloDemo:InputContract", ...
        "YOLO demo expects one float32 input with concrete shape [1,3,H,W].");
end

uiHeight = uint32(vInputShape(3));
uiWidth = uint32(vInputShape(4));
mImage = imread(strImagePath);
if size(mImage, 3) == 1
    mImage = repmat(mImage, 1, 1, 3);
elseif size(mImage, 3) > 3
    mImage = mImage(:, :, 1:3);
end
mResizedImage = imresize(mImage, [double(uiHeight), double(uiWidth)], "bilinear");

% Permuting HWC to CWH makes MATLAB linearization match row-major HWC order.
vHwcValues = double(reshape(permute(mResizedImage, [3, 2, 1]), [], 1));
oInput = ptafdeploy.inference.MakeNchwFloatTensorFromHwcVector( ...
    strInputName, vHwcValues, uiHeight, uiWidth, uint32(3), 1.0 / 255.0, false);
end


function PrintYoloSummary_(stResult)
% Print the same stable summary fields used by the native and Python demos.
arguments
    stResult (1, 1) struct
end

fprintf("role=%s\n", stResult.strRole);
fprintf("backend=%s\n", stResult.strBackend);
fprintf("input_shape=[%s]\n", strjoin(string(stResult.vInputShape.'), ","));
fprintf("output_shape=[%s]\n", strjoin(string(stResult.vOutputShape.'), ","));
fprintf("detections=%d\n", size(stResult.mDetections, 1));
for uiIndex = 1:size(stResult.mDetections, 1)
    vDetection = stResult.mDetections(uiIndex, :);
    fprintf("detection[%d]=class=%d,score=%.6f,cx=%.3f,cy=%.3f,w=%.3f,h=%.3f\n", ...
        uiIndex - 1, int64(vDetection(6)), vDetection(5), vDetection(1), ...
        vDetection(2), vDetection(3), vDetection(4));
end
end
