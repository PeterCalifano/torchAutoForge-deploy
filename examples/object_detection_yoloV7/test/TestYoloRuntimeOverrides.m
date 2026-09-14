classdef TestYoloRuntimeOverrides < matlab.unittest.TestCase
    % Verify runtime override behavior through the MATLAB demo and native reader.
    methods (Test)
        function PreserveManifestRuntime(oTest)
            strExample = string(fileparts(fileparts(mfilename('fullpath'))));
            strRoot = string(fileparts(fileparts(strExample)));
            strModel = fullfile(strRoot, "models", "onnx", "yolov7_640x640.onnx");
            strImage = fullfile(strExample, "sample_data", "horses.jpg");
            oTest.assumeTrue(isfile(strModel) && isfile(strImage));
            strDirectory = string(tempname);
            mkdir(strDirectory);
            oCleanup = onCleanup(@() rmdir(strDirectory, 's')); %#ok<NASGU>

            for ui32Case = uint32(1:4)
                strPrefix = fullfile(strDirectory, "profile" + ui32Case);
                strManifest = fullfile(strDirectory, "model.ptafmodel");
                stManifest = struct("schema_version", 1, "artifact_path", strModel, ...
                    "task", "object_detection", "execution_target_priority", {{'cpu'}}, ...
                    "device_id", 1, "intra_op_num_threads", 2, "inter_op_num_threads", 3, ...
                    "enable_profiling", true, "log_id", strPrefix);
                dFile = fopen(strManifest, 'w');
                oTest.assertGreaterThanOrEqual(dFile, 0);
                fprintf(dFile, '%s', jsonencode(stManifest));
                fclose(dFile);

                % The shared reader must expose the manifest's nondefault runtime policy.
                oRuntime = ptafdeploy.inference.ReadPtafModelRuntimeConfig(char(strManifest));
                oTest.verifyEqual(oRuntime.GetIntraOpNumThreads(), int32(2));
                oTest.verifyEqual(oRuntime.GetInterOpNumThreads(), int32(3));
                switch ui32Case
                    case 1
                        stResult = RunYoloFacadeDemo(strManifest, strImage);
                    case 2
                        stResult = RunYoloFacadeDemo(strManifest, strImage, strTarget="cpu");
                    case 3
                        stResult = RunYoloFacadeDemo(strManifest, strImage, ui32DeviceId=uint32(0));
                    case 4
                        stResult = RunYoloFacadeDemo(strManifest, strImage, ...
                            strTarget="cpu", ui32DeviceId=uint32(0));
                end
                dExpectedDevice = double(ui32Case < 3);
                oTest.verifyTrue(contains(stResult.strBackend, "device_id=" + dExpectedDevice));
                oTest.verifyNotEmpty(dir(strPrefix + "*.json"));
            end
        end
    end
end
