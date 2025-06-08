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
    charInputShape    (1,:) char = 'BC'
    charOutputShape   (1,:) char = 'BC'
    dInputSample      (:,:) single {ismatrix} = [] % Optional input sample
    dOutputSample     (:,:) single {ismatrix} = [] % Optional label sample
end
arguments
    kwargs.charModelName = "importedModelONNx"
end

% Verify if the model file exists
if ~isfile(charModelfilePath)
    error('Specified model file %s not found.', charModelfilePath)
end

% Import a PyTorch model into MATLAB
objModel = importNetworkFromONNX(charModelfilePath, ...
                            "InputDataFormats", charInputShape, ...
                            "OutputDataFormats", charOutputShape);

% Get random input sample if not provided
if isempty(dInputSample)
    dInputSample = single(rand(dModelInputSizes));
end

% Initialize model
objX = dlarray(dInputSample, charInputShape);  
objModel = initialize(objModel, objX);
analyzeNetwork(objModel);

% Test inference
dOutput = objModel.predict(objX);

% Compute error if provided
if not(isempty(dOutputSample))
    errorValue = dOutput - dOutputSample; % Compute the error
    disp("Error computed. Size: " + string(size(errorValue)));
end

% Fix any invalid 
kwargs.charModelName = matlab.lang.makeValidName(kwargs.charModelName);

% Save model to mat file
strTmpStruct.(kwargs.charModelName) = objModel;
charMatSavePath = strcat(kwargs.charModelName, ".mat");
save(charMatSavePath, "-struct", "strTmpStruct");

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
