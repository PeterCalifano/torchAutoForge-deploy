function [objModel, charMatSavePath, objModelInSLX] = ImportONNxModel(charModelfilePath, ...
                                                                    dModelInputSizes, ...
                                                                    charInputShape, ...
                                                                    charOutputShape, ...
                                                                    dInputSample, ...
                                                                    dOutputSample, ...
                                                                    kwargs)
arguments
    charModelfilePath (1,:) string {mustBeA(charModelfilePath, ["char", "string"])}
    dModelInputSizes  (1,:) double {isvector}
    charInputShape    (1,:) string = 'BC'
    charOutputShape   (1,:) string = 'BC'
    dInputSample      (:,:) single {ismatrix} = [] % Optional input sample
    dOutputSample     (:,:) single {ismatrix} = [] % Optional label sample
end
arguments
    kwargs.charModelName string {mustBeA(kwargs.charModelName, ["string", "char"])} = "ImportedModelFromONNx"
    kwargs.bExportToSimulink {islogical, isscalar} = false
    kwargs.charOutputPath {mustBeA(kwargs.charOutputPath, ["string", "char"])} = "."
    kwargs.bRunNetworkAnalysis (1,1) logical {isscalar, islogical} = false
    kwargs.bRemoveOutputLayer  (1,1) logical {isscalar, islogical} = false
end

% Verify if the model file exists
if ~isfile(charModelfilePath)
    error('Specified model file %s not found.', charModelfilePath)
end

% Import a PyTorch model into MATLAB
objModel = importNetworkFromONNX(charModelfilePath, ...
                            "InputDataFormats", charInputShape, ...
                            "OutputDataFormats", charOutputShape);

if kwargs.bRemoveOutputLayer
    % Remove outpu layer
    objModel = objModel.removeLayers(objModel.OutputNames{1});
    ui32NumOfLayers = length(objModel.Layers);
    objModel.OutputNames = {objModel.Layers(ui32NumOfLayers).Name};
end

% Get random input sample if not provided
if isempty(dInputSample)
    dInputSample = single(rand(dModelInputSizes));
end

% Initialize model
objX        = dlarray(dInputSample, charInputShape);  
objModel    = initialize(objModel, objX);

if kwargs.bRunNetworkAnalysis
    analyzeNetwork(objModel);
end

% Test inference
dOutput = objModel.predict(objX);

% Compute error if provided
if not(isempty(dOutputSample))
    errorValue = dOutput - dOutputSample; % Compute the error
    disp("Error computed. Size: " + string(size(errorValue)));
end

%% Save model to mat file

% Fix any invalid character in output name (struct field) 
kwargs.charModelName = matlab.lang.makeValidName(kwargs.charModelName);

if not(isfolder(kwargs.charOutputPath))
    mkdir(kwargs.charOutputPath)
end

charCurrentDir = pwd;
cd(kwargs.charOutputPath)

strTmpStruct.(kwargs.charModelName) = objModel;
charMatSavePath = fullfile(strcat(kwargs.charModelName, ".mat") );
save(charMatSavePath, "-struct", "strTmpStruct");

cd(charCurrentDir);

% ATtach output directory path
charMatSavePath = fullfile(kwargs.charOutputPath, charMatSavePath);

%% Export to Simulink
% DEVNOTE: requires MATLAB2024b
try
    if kwargs.bExportToSimulink
        charExportModelName = strcat(kwargs.charModelName, "_ExportSLX");
        % Export the dlnetwork to a Simulink model
        objModelInSLX = exportNetworkToSimulink(objModel, ModelName=charExportModelName);
    else
        objModelInSLX = [];
    end
catch ME
    fprintf(2, "Export to Simulink failed due to error %s." + ...
        "\nNote that the functionality is only available since MATLAB2024b. " + ...
        "However, you can use the Predict block in SLX and load the model from the .mat this function has saved.", string(ME.message))
    objModelInSLX = [];
end

end
