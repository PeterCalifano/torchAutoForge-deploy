function stResult = RunCentroidingFacadeDemo(strModelPath, strImagePath, stOptions)
% RUNCENTROIDINGFACADEDEMO Run image-only centroiding through the MATLAB wrapper.
%
% SIGNATURE
%   stResult = RunCentroidingFacadeDemo(strModelPath, strImagePath)
%   stResult = RunCentroidingFacadeDemo(..., strTarget="cpu")
%
% DESCRIPTION
%   Loads a centroiding .ptafmodel manifest or raw ONNX artifact through
%   CModelFacade once, converts each image to grayscale NCHW through the generic
%   wrapper bridge, and maps the normalized feature output into model-input
%   and original-image pixel coordinates. Frames are independent; JSON output
%   and crosshair overlays are optional.
%
% INPUT
%   strModelPath          Path to a centroiding manifest or raw ONNX model.
%   strImagePath          Path to one image or a non-recursive image directory.
%   stOptions.strOutputPath New or empty output directory; empty disables output.
%   stOptions.bOverlays   Save original-resolution crosshairs; default false.
%   stOptions.strTarget   "manifest", "cpu", or "cuda".
%   stOptions.ui32DeviceId
%                         Non-negative runtime device index.
%
% OUTPUT
%   stResult              Versioned run report with ordered frame records.
%                         Image arrays are released after each frame.
%
% CHANGELOG
%   2026-09-01 Initial wrapper-backed plain-centroiding demo.
%   2026-09-12 Add independent sequences, reports, and optional overlays.
%
% DEPENDENCIES
%   Generated ptafdeploy MATLAB wrapper, Image Processing Toolbox; JVM for
%   canonical output paths and atomic JSON publication when output is requested.

arguments (Input)
    strModelPath (1, 1) string
    strImagePath (1, 1) string
    stOptions.strTarget (1, 1) string {mustBeMember(stOptions.strTarget, ...
        ["manifest", "cpu", "cuda"])} = "manifest"
    stOptions.ui32DeviceId (1, 1) uint32 = uint32(0)
    stOptions.strOutputPath (1, 1) string = ""
    stOptions.bOverlays (1, 1) logical = false
end

arguments (Output)
    stResult (1, 1) struct
end

% Select the complete input sequence before creating output directories.
cFrames = SelectFrames_(strImagePath);
if stOptions.bOverlays && stOptions.strOutputPath == ""
    error("ptafdeploy:CentroidingDemo:Options", "Overlays require strOutputPath.");
end
if stOptions.strOutputPath ~= ""
    PrepareOutput_(stOptions.strOutputPath, strImagePath);
end
stResult = struct("schema_version", 1, "requested_model_path", strModelPath, ...
    "status", "incomplete", "model", NaN, ...
    "input", struct("path", strImagePath, "kind", "image", ...
    "ordering", "natural_filename", "selected_frame_count", numel(cFrames)), ...
    "preprocessing", struct("library", "MATLAB " + version, "grayscale", "im2gray im2uint8", ...
    "resize", "bilinear, Antialiasing=false", "scale", 1/255), "frames", {{}});
if isfolder(strImagePath)
    stResult.input.kind = "directory";
end
PublishReport_(stOptions.strOutputPath, stResult);
strStage = "model_load";
iFrame = NaN;
strSource = NaN;
try
    oModel = LoadCentroidingModel_( ...
        strModelPath, stOptions.strTarget, stOptions.ui32DeviceId);
    if ~strcmp(oModel.GetRole(), "centroiding") || ...
            oModel.GetNumInputs() ~= 1 || oModel.GetNumOutputs() ~= 1
        error("ptafdeploy:CentroidingDemo:Contract", ...
            "Centroiding requires exactly one image input and one output.");
    end
    stResult.model = ModelMetadata_(oModel);
    stInputInfo = oModel.GetInputInfo(0);
    vModelInputShape = ptafdeploy.inference.GetModelInputShapeVector(oModel, 0);
    if stOptions.bOverlays
        strStage = "output_directory";
        [bSuccess, strMessage] = mkdir(fullfile(stOptions.strOutputPath, "overlays"));
        if ~bSuccess
            error("ptafdeploy:CentroidingDemo:Output", "%s", strMessage);
        end
    end
    for iFrame = 1:numel(cFrames)
        strSource = string(cFrames{iFrame});
        [~, strStem, strExtension] = fileparts(strSource);
        strStage = "preprocessing";
        [oInput, vImageSize] = PrepareCentroidingInput_(strSource, stInputInfo, vModelInputShape);
        vInputShape = ptafdeploy.inference.GetFloatTensorShapeVector(oInput);
        strStage = "inference";
        oTimer = tic;
        oOutput = oModel.InferSingleFloatTensor(oInput);
        dDuration = toc(oTimer) * 1000;
        strStage = "decoding_output";
        vOutputShape = ptafdeploy.inference.GetFloatTensorShapeVector(oOutput);
        mFeatures = DecodeCentroidingOutput_(oOutput, vOutputShape);
        vNormalized = mFeatures(1, 1:2).';
        vModelPixels = vNormalized .* [vInputShape(4); vInputShape(3)];
        vImagePixels = vNormalized .* vImageSize;
        stSummary = struct("strRole", string(oModel.GetRole()), ...
            "strBackend", string(oModel.GetBackendDetail()), "strInputName", string(stInputInfo.name), ...
            "vInputShape", vInputShape, "vImageSize", vImageSize, "strOutputName", string(oOutput.name), ...
            "vOutputShape", vOutputShape, "vNormalized", vNormalized, ...
            "vModelInputPixels", vModelPixels, "vOriginalPixels", vImagePixels);
        fprintf("frame.index=%d\nframe.source=%s\ninference_ms=%.9g\n", ...
            iFrame-1, strStem+strExtension, dDuration);
        PrintCentroidingSummary_(stSummary);
        strOverlay = NaN;
        if stOptions.bOverlays
            strStage = "overlay";
            strOverlay = "overlays/" + compose("%06d_", iFrame-1) + strStem + ".png";
            SaveOverlay_(strSource, fullfile(stOptions.strOutputPath, strOverlay), vImagePixels);
        end
        strStage = "report";
        stFrame = struct("index", iFrame-1, "source", strStem+strExtension, ...
            "image_size", struct("width", vImageSize(1), "height", vImageSize(2)), ...
            "raw_output", struct("name", string(oOutput.name), "shape", {num2cell(vOutputShape.')}, ...
                "values", {num2cell(ptafdeploy.inference.GetFloatTensorValuesVector(oOutput).')}), ...
            "centroid", struct("normalized", Point_(vNormalized), ...
                "model_pixels", Point_(vModelPixels), "image_pixels", Point_(vImagePixels), ...
                "inside_image", all(vImagePixels >= 0 & vImagePixels < vImageSize)), ...
            "inference_ms", dDuration, "overlay", strOverlay);
        AppendFrame_(stOptions.strOutputPath, stFrame);
        stResult.frames{end+1} = stFrame;
    end
    iFrame = NaN;
    strSource = NaN;
    strStage = "report";
    stResult.status = "complete";
    PublishReport_(stOptions.strOutputPath, stResult);
    if stOptions.strOutputPath ~= ""
        try
            delete(fullfile(stOptions.strOutputPath, "frames.jsonl"));
        catch
            % Cleanup failure does not invalidate a successfully published report.
        end
    end
catch oError
    stResult.status = "incomplete";
    if isstring(strSource)
        [~, strName, strExtension] = fileparts(strSource);
        strSource = strName + strExtension;
    end
    stResult.error = struct("stage", strStage, "frame_index", iFrame-1, ...
        "source", strSource, "message", string(oError.message));
    try
        PublishReport_(stOptions.strOutputPath, stResult);
    catch oPublication
        warning("ptafdeploy:CentroidingDemo:Report", ...
            "Report publication failed: %s; retained files under %s", ...
            oPublication.message, stOptions.strOutputPath);
    end
    rethrow(oError);
end
end


function oModel = LoadCentroidingModel_(strModelPath, strTarget, ui32DeviceId)
% Load a role manifest or raw ONNX with an optional strict runtime override.
arguments (Input)
    strModelPath (1, 1) string
    strTarget (1, 1) string
    ui32DeviceId (1, 1) uint32
end

arguments (Output)
    oModel (1, 1) ptafdeploy.inference.CModelFacade
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
    error("ptafdeploy:CentroidingDemo:Artifact", ...
        "Expected a .ptafmodel manifest or .onnx artifact.");
end
end


function oRuntime = MakeRuntime_(strTarget, ui32DeviceId)
% Construct a strict backend-neutral runtime override.
arguments (Input)
    strTarget (1, 1) string {mustBeMember(strTarget, ["cpu", "cuda"])}
    ui32DeviceId (1, 1) uint32
end

arguments (Output)
    oRuntime (1, 1) ptafdeploy.inference.SRuntimeConfig
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


function [oInput, vImageSize] = PrepareCentroidingInput_( ...
        strImagePath, stInputInfo, vInputShape)
% Convert one image to uint8 grayscale and row-major NCHW wrapper storage.
arguments (Input)
    strImagePath (1, 1) string
    stInputInfo (1, 1) ptafdeploy.inference.STensorInfo
    vInputShape (:, 1) double
end

arguments (Output)
    oInput (1, 1) ptafdeploy.inference.SFloatTensor
    vImageSize (2, 1) double
end

if ~strcmp(stInputInfo.dtype, "float32") || numel(vInputShape) ~= 4 || ...
        ~ismember(vInputShape(1), [-1, 1]) || vInputShape(2) ~= 1 || ...
        any(vInputShape(3:4) <= 0) || any(vInputShape ~= fix(vInputShape))
    error("ptafdeploy:CentroidingDemo:InputContract", ...
        "Centroiding expects float32 input [N,1,H,W] with concrete H/W.");
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


function mFeatures = DecodeCentroidingOutput_(oOutput, vOutputShape)
% Validate the plain output contract and invoke generic feature decoding.
arguments (Input)
    oOutput (1, 1) ptafdeploy.inference.SFloatTensor
    vOutputShape (:, 1) double
end

arguments (Output)
    mFeatures (:, :) double
end

if ~isequal(vOutputShape, double([1, 2]'))
    error("ptafdeploy:CentroidingDemo:OutputContract", ...
        "Centroiding expects output shape [1,2].");
end

oSchema = ptafdeploy.inference.SFeatureRowSchema();
mFeatures = ptafdeploy.inference.DecodeFeatureRowsMatrix(oOutput, oSchema);
if ~isequal(size(mFeatures), [1, 3]) || any(~isfinite(mFeatures(1, 1:2)))
    error("ptafdeploy:CentroidingDemo:OutputValues", ...
        "Centroiding must produce one finite normalized feature.");
end
end


function PrintCentroidingSummary_(stResult)
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

function stPoint = Point_(vPosition)
% Preserve continuous zero-origin coordinates as JSON fields.
arguments (Input)
    vPosition (2, 1) double
end

arguments (Output)
    stPoint (1, 1) struct
end
stPoint = struct("x", vPosition(1), "y", vPosition(2));
end

% Image selection policy: no model or report ownership.
function cFrames = SelectFrames_(strInput)
% Fix the supported input sequence before any output is created.
arguments (Input)
    strInput (1, 1) string
end

arguments (Output)
    cFrames (1, :) cell
end
if isfile(strInput)
    cFrames = {char(strInput)};
    return;
end
if ~isfolder(strInput)
    error("ptafdeploy:CentroidingDemo:Input", "Missing input: %s", strInput);
end
stEntries = dir(strInput);
cNames = {};
for iEntry = 1:numel(stEntries)
    [~, ~, strExtension] = fileparts(stEntries(iEntry).name);
    if ~stEntries(iEntry).isdir && ismember(lower(string(strExtension)), ...
            [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"])
        cNames{end+1} = stEntries(iEntry).name; %#ok<AGROW>
    end
end
if isempty(cNames)
    error("ptafdeploy:CentroidingDemo:Input", "No supported images: %s", strInput);
end
cNames = NaturalSort_(cNames);
cFrames = cellfun(@(strName) fullfile(char(strInput), strName), cNames, "UniformOutput", false);
end

function cSorted = NaturalSort_(cNames)
% Merge sort keeps filename ordering practical for long sequences.
arguments (Input)
    cNames (1, :) cell
end

arguments (Output)
    cSorted (1, :) cell
end
if numel(cNames) < 2
    cSorted = cNames;
    return;
end
iMiddle = floor(numel(cNames)/2);
cLeft = NaturalSort_(cNames(1:iMiddle));
cRight = NaturalSort_(cNames(iMiddle+1:end));
cSorted = cell(size(cNames));
iLeft = 1;
iRight = 1;
for iOutput = 1:numel(cNames)
    if iRight > numel(cRight) || (iLeft <= numel(cLeft) && NaturalLess_(cLeft{iLeft}, cRight{iRight}))
        cSorted{iOutput} = cLeft{iLeft};
        iLeft = iLeft + 1;
    else
        cSorted{iOutput} = cRight{iRight};
        iRight = iRight + 1;
    end
end
end

function bLess = NaturalLess_(strLeft, strRight)
% UTF-8 byte ordering agrees with Unicode code-point order for valid filenames.
arguments (Input)
    strLeft (1, :) char
    strRight (1, :) char
end

arguments (Output)
    bLess (1, 1) logical
end
vLeft = unicode2native(strLeft, 'UTF-8');
vRight = unicode2native(strRight, 'UTF-8');
iLeft = 1;
iRight = 1;
while iLeft <= numel(vLeft) && iRight <= numel(vRight)
    if vLeft(iLeft) >= 48 && vLeft(iLeft) <= 57 && vRight(iRight) >= 48 && vRight(iRight) <= 57
        iEndLeft = iLeft;
        iEndRight = iRight;
        while iEndLeft <= numel(vLeft) && vLeft(iEndLeft) >= 48 && vLeft(iEndLeft) <= 57
            iEndLeft = iEndLeft+1;
        end
        while iEndRight <= numel(vRight) && vRight(iEndRight) >= 48 && vRight(iEndRight) <= 57
            iEndRight = iEndRight+1;
        end
        while iLeft < iEndLeft && vLeft(iLeft) == 48
            iLeft = iLeft + 1;
        end
        while iRight < iEndRight && vRight(iRight) == 48
            iRight = iRight + 1;
        end
        if iEndLeft-iLeft ~= iEndRight-iRight
            bLess = iEndLeft - iLeft < iEndRight - iRight;
            return;
        end
        vDifference = find(vLeft(iLeft:iEndLeft-1) ~= vRight(iRight:iEndRight-1), 1);
        if ~isempty(vDifference)
            bLess = vLeft(iLeft+vDifference-1) < vRight(iRight+vDifference-1);
            return;
        end
        iLeft = iEndLeft;
        iRight = iEndRight;
    else
        if vLeft(iLeft) ~= vRight(iRight)
            bLess = vLeft(iLeft) < vRight(iRight);
            return;
        end
        iLeft = iLeft + 1;
        iRight = iRight + 1;
    end
end
if iLeft <= numel(vLeft) || iRight <= numel(vRight)
    bLess = iLeft > numel(vLeft);
    return;
end
iDifference = find(vLeft(1:min(end,numel(vRight))) ~= vRight(1:min(end,numel(vLeft))), 1);
if isempty(iDifference)
    bLess = numel(vLeft) < numel(vRight);
else
    bLess = vLeft(iDifference) < vRight(iDifference);
end
end

% Output publication: filesystem ownership and JSON errors.
function PrepareOutput_(strOutput, strInput)
% Canonical paths prevent symlink aliases from bypassing collision protection.
arguments (Input)
    strOutput (1, 1) string
    strInput (1, 1) string
end
strDestination = string(java.io.File(char(strOutput)).getCanonicalPath());
strSource = string(java.io.File(char(strInput)).getCanonicalPath());
if strDestination == strSource || startsWith(strSource, strDestination + filesep)
    error("ptafdeploy:CentroidingDemo:Output", "Output must not equal or contain the input location.");
end
if isfile(strOutput)
    error("ptafdeploy:CentroidingDemo:Output", "Output is a file: %s", strOutput);
end
if isfolder(strOutput)
    stEntries = dir(strOutput);
    if any(~ismember({stEntries.name}, {'.', '..'}))
        error("ptafdeploy:CentroidingDemo:Output", "Output directory is not empty: %s", strOutput);
    end
else
    [bSuccess, strMessage] = mkdir(strOutput);
    if ~bSuccess
        error("ptafdeploy:CentroidingDemo:Output", "%s", strMessage);
    end
end
end

function PublishReport_(strOutput, stReport)
% Publish a complete JSON document; preserve the prior report if replacement fails.
arguments (Input)
    strOutput (1, 1) string
    stReport (1, 1) struct
end
if strOutput == ""
    return;
end
strTemporary = fullfile(strOutput, "predictions.json.tmp");
WriteJson_(strTemporary, stReport, "w");
% Java NIO requests an atomic replacement on the same filesystem.
oSource = java.io.File(char(strTemporary)).toPath();
oDestination = java.io.File(char(fullfile(strOutput, "predictions.json"))).toPath();
vOptions = javaArray('java.nio.file.CopyOption', 2);
vOptions(1) = java.nio.file.StandardCopyOption.ATOMIC_MOVE;
vOptions(2) = java.nio.file.StandardCopyOption.REPLACE_EXISTING;
java.nio.file.Files.move(oSource, oDestination, vOptions);
end

% Adapter metadata: describes the selected model and preprocessing contract.
function stModel = ModelMetadata_(oModel)
% Serialize wrapper-visible effective runtime and tensor metadata.
arguments (Input)
    oModel (1, 1) ptafdeploy.inference.CModelFacade
end

arguments (Output)
    stModel (1, 1) struct
end
stContract = oModel.GetContract();
oRuntime = stContract.runtime;
stRuntime = struct("device_id", oRuntime.GetDeviceId(), ...
    "intra_op_num_threads", oRuntime.GetIntraOpNumThreads(), ...
    "inter_op_num_threads", oRuntime.GetInterOpNumThreads(), ...
    "allow_fallback", oRuntime.GetAllowFallback(), "backend", double(oRuntime.GetBackend()), ...
    "artifact", double(oRuntime.GetArtifact()), "execution_target_priority", NaN, ...
    "enable_profiling", oRuntime.GetEnableProfiling(), "log_id", string(oRuntime.GetLogId()), ...
    "tensorrt_optimization_profile_index", oRuntime.GetTensorRtOptimizationProfileIndex());
stInput = oModel.GetInputInfo(0);
stOutput = oModel.GetOutputInfo(0);
stModel = struct("artifact_path", string(stContract.artifact_path), ...
    "config_path", string(stContract.config_path), "role", string(stContract.role), ...
    "backend_detail", string(stContract.backend_detail), "runtime", stRuntime, ...
    "preprocessing", string(stContract.preprocessing), ...
    "postprocessing", string(stContract.postprocessing), ...
    "inputs", {{struct("name", string(stInput.name), "dtype", string(stInput.dtype), ...
        "shape", {num2cell(ptafdeploy.inference.GetModelInputShapeVector(oModel, 0).')})}}, ...
    "outputs", {{struct("name", string(stOutput.name), "dtype", string(stOutput.dtype), ...
        "shape", {num2cell(ptafdeploy.inference.GetModelOutputShapeVector(oModel, 0).')})}});
end

% Image annotation: the caller supplies decoded original-image coordinates.
function SaveOverlay_(strSource, strDestination, vPoint)
% Draw into the current image only; leave pixels outside the strokes unchanged.
arguments (Input)
    strSource (1, 1) string
    strDestination (1, 1) string
    vPoint (2, 1) double
end
[mImage, mMap, mAlpha] = imread(strSource);
if ~isempty(mMap)
    mImage = im2uint8(ind2rgb(mImage, mMap));
end
if islogical(mImage)
    mImage = im2uint8(mImage);
end
if ~isa(mImage, 'uint8') && ~isa(mImage, 'uint16')
    error("ptafdeploy:CentroidingDemo:Overlay", "PNG overlays require uint8 or uint16 samples.");
end
if size(mImage, 3) == 1
    mImage = repmat(mImage, 1, 1, 3);
end
dWhite = double(intmax(class(mImage)));
if all(vPoint >= -10) && vPoint(1) < size(mImage,2)+10 && vPoint(2) < size(mImage,1)+10
    vCenter = round(vPoint) + 1;
    for iPass = 1:2
        iRadius = 10 - iPass;
        iHalfWidth = 2 - iPass;
        vX = max(1,vCenter(1)-iRadius):min(size(mImage,2),vCenter(1)+iRadius);
        vY = max(1,vCenter(2)-iRadius):min(size(mImage,1),vCenter(2)+iRadius);
        vRows = max(1,vCenter(2)-iHalfWidth):min(size(mImage,1),vCenter(2)+iHalfWidth);
        vCols = max(1,vCenter(1)-iHalfWidth):min(size(mImage,2),vCenter(1)+iHalfWidth);
        mImage(vRows,vX,:) = (iPass-1)*dWhite;
        mImage(vY,vCols,:) = (iPass-1)*dWhite;
        if ~isempty(mAlpha)
            if isinteger(mAlpha)
                dOpaque = double(intmax(class(mAlpha)));
            else
                dOpaque = 1;
            end
            mAlpha(vRows,vX) = dOpaque;
            mAlpha(vY,vCols) = dOpaque;
        end
    end
end
if isempty(mAlpha)
    imwrite(mImage, strDestination);
else
    imwrite(mImage, strDestination, "Alpha", mAlpha);
end
end

function AppendFrame_(strOutput, stFrame)
% Retain completed records if final report publication fails.
arguments (Input)
    strOutput (1, 1) string
    stFrame (1, 1) struct
end
if strOutput == ""
    return;
end
WriteJson_(fullfile(strOutput, "frames.jsonl"), stFrame, "a");
end

function WriteJson_(strPath, stValue, strMode)
% Check writes and close errors before a record is considered complete.
arguments (Input)
    strPath (1, 1) string
    stValue (1, 1) struct
    strMode (1, 1) string {mustBeMember(strMode, ["a", "w"])}
end

% NaN is used only for schema-null metadata/overlay fields; predictions are finite.
vBytes = unicode2native(string(jsonencode(stValue, "ConvertInfAndNaN", true)) + newline, 'UTF-8');
iFile = fopen(strPath, strMode, 'n', 'UTF-8');
if iFile < 0
    error("ptafdeploy:CentroidingDemo:Report", "Cannot open JSON output: %s", strPath);
end
try
    iCount = fwrite(iFile, vBytes, 'uint8');
catch oError
    fclose(iFile);
    rethrow(oError);
end
iStatus = fclose(iFile);
if iStatus ~= 0 || iCount ~= numel(vBytes)
    error("ptafdeploy:CentroidingDemo:Report", "Cannot write JSON output: %s", strPath);
end
end
