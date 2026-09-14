function oTests = TestCentroidingSequence
% TESTCENTROIDINGSEQUENCE Check observable MATLAB sequence and report behavior.
% SIGNATURE
%   runtests('TestCentroidingSequence')
% DESCRIPTION
%   Uses the externally selected image-only ONNX model; does not assert accuracy
%   against arbitrary synthetic-image centres.
% INPUT
%   PTAFDEPLOY_PLAIN_CENTROIDING_ONNX environment variable.
% OUTPUT
%   oTests: function-based unit tests.
% CHANGELOG
%   2026-09-12 Add sequence smoke and empty-input checks.
% DEPENDENCIES
%   MATLAB test framework, generated wrapper, Image Processing Toolbox.
% functiontests rejects arguments blocks in test entry points and local tests.
oTests = functiontests(localfunctions);
end

function TestSequence_(oTest)
% Check natural ordering, JSON arrays, and original-resolution overlays.
strModel = string(getenv('PTAFDEPLOY_PLAIN_CENTROIDING_ONNX'));
oTest.assumeTrue(isfile(strModel));
strRoot = string(tempname); mkdir(strRoot);
oCleanup = onCleanup(@() rmdir(strRoot, 's')); %#ok<NASGU>
strInput = fullfile(strRoot, 'input'); mkdir(strInput);
imwrite(uint8(ones(24,32)*80), fullfile(strInput, 'frame10.png'));
imwrite(uint8(ones(24,32)*80), fullfile(strInput, 'frame2.png'));
strOutput = fullfile(strRoot, 'output');
stRun = RunCentroidingFacadeDemo(strModel, strInput, strTarget="cpu", ...
    strOutputPath=strOutput, bOverlays=true);
oTest.verifyEqual(stRun.status, "complete");
oTest.verifyEqual(numel(stRun.frames), 2);
oTest.verifyEqual(stRun.frames{1}.source, "frame2.png");
stJson = jsondecode(fileread(fullfile(strOutput, 'predictions.json')));
oTest.verifyEqual(stJson.schema_version, 1);
oTest.verifyEqual(string(stJson.model.task), "centroiding");
oTest.verifyFalse(isfield(stJson.model, "role"));
oTest.verifyEqual(numel(stJson.frames), 2);
mOverlay = imread(fullfile(strOutput, stRun.frames{1}.overlay));
oTest.verifyEqual(size(mOverlay, [1,2]), [24,32]);
oTest.verifyTrue(isfinite(stRun.frames{1}.inference_ms));
end

function TestEmptyInput_(oTest)
% Reject a directory with no selected frames before any model is loaded.
strRoot = string(tempname); mkdir(strRoot);
oCleanup = onCleanup(@() rmdir(strRoot, 's')); %#ok<NASGU>
oTest.verifyError(@() RunCentroidingFacadeDemo("absent.onnx", strRoot), ...
    "ptafdeploy:CentroidingDemo:Input");
end

function TestRuntimeSelection_(oTest)
% Verify device-only selection and deliberate target-policy replacement.
strModel = string(getenv('PTAFDEPLOY_PLAIN_CENTROIDING_ONNX'));
oTest.assumeTrue(isfile(strModel));
strRoot = string(tempname);
mkdir(strRoot);
oCleanup = onCleanup(@() rmdir(strRoot, 's')); %#ok<NASGU>
strImage = fullfile(strRoot, "frame.png");
imwrite(uint8(ones(24, 32) * 80), strImage);
strManifest = fullfile(strRoot, "model.ptafmodel");
stManifest = struct("schema_version", 1, "artifact_path", strModel, ...
    "task", "centroiding", "execution_target_priority", {{'cpu'}}, ...
    "device_id", 0, "intra_op_num_threads", 2, "inter_op_num_threads", 3, ...
    "allow_fallback", true, "log_id", "centroiding_manifest_policy");
dFile = fopen(strManifest, 'w');
oTest.assertGreaterThanOrEqual(dFile, 0);
fprintf(dFile, '%s', jsonencode(stManifest));
fclose(dFile);

stRun = RunCentroidingFacadeDemo(strManifest, strImage, ui32DeviceId=uint32(1));
oTest.verifyEqual(stRun.model.runtime.device_id, int32(1));
oTest.verifyEqual(stRun.model.runtime.intra_op_num_threads, int32(2));
oTest.verifyEqual(stRun.model.runtime.inter_op_num_threads, int32(3));
oTest.verifyTrue(logical(stRun.model.runtime.allow_fallback));

stRun = RunCentroidingFacadeDemo(strManifest, strImage, strTarget="cpu");
oDefaults = ptafdeploy.inference.SRuntimeConfig();
oTest.verifyEqual(stRun.model.runtime.device_id, int32(0));
oTest.verifyEqual(stRun.model.runtime.intra_op_num_threads, oDefaults.GetIntraOpNumThreads());
oTest.verifyFalse(logical(stRun.model.runtime.allow_fallback));

stRun = RunCentroidingFacadeDemo(strModel, strImage, ui32DeviceId=uint32(1));
oTest.verifyEqual(stRun.model.runtime.device_id, int32(1));
oTest.verifyError(@() RunCentroidingFacadeDemo(strManifest, strImage, ...
    ui32DeviceId=uint32([0; 1])), "ptafdeploy:CentroidingDemo:Device");
end
