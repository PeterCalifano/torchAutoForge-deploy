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
arguments (Output)
    oTests (1, :) matlab.unittest.Test
end
oTests = functiontests(localfunctions);
end

function TestSequence_(oTest)
% Check natural ordering, JSON arrays, and original-resolution overlays.
arguments (Input)
    oTest (1, 1) matlab.unittest.TestCase
end
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
oTest.verifyEqual(numel(stJson.frames), 2);
mOverlay = imread(fullfile(strOutput, stRun.frames{1}.overlay));
oTest.verifyEqual(size(mOverlay, [1,2]), [24,32]);
oTest.verifyTrue(isfinite(stRun.frames{1}.inference_ms));
end

function TestEmptyInput_(oTest)
% Reject a directory with no selected frames before any model is loaded.
arguments (Input)
    oTest (1, 1) matlab.unittest.TestCase
end
strRoot = string(tempname); mkdir(strRoot);
oCleanup = onCleanup(@() rmdir(strRoot, 's')); %#ok<NASGU>
oTest.verifyError(@() RunCentroidingFacadeDemo("absent.onnx", strRoot), ...
    "ptafdeploy:CentroidingDemo:Input");
end
