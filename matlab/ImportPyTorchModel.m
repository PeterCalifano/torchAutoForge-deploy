function objModel = ImportPyTorchModel(charModelfilePath, dModelInputSizes, dInputSample, dOutputSample)
arguments
    charModelfilePath (1,:) string {mustBeA(charModelfilePath, ["char", "string"])}
    dModelInputSizes  (1,:) double {isvector}
    dInputSample      (:,:) double {ismatrix} = [] % Optional input sample
    dOutputSample     (:,:) double {ismatrix} = [] % Optional label sample
end

% Verify if the model file exists
if ~isfile(charModelfilePath)
    error('Specified model file %s not found.', charModelfilePath)
end

% Import a PyTorch model into MATLAB
objModel = importNetworkFromPyTorch(charModelfilePath, ...
                                "PyTorchInputSizes", dModelInputSizes);
analyzeNetwork(objModel);

% Get random input sample if not provided
if isempty(dModelInputSizes)
    dInputSample = rand(dModelInputSizes);
end

% Test inference
dOutput = objModel.predict(dInputSample);
disp("Model inference completed. Output size: " + string(size(dOutput)));

% Compute error if provided
if not(isempty(dOutputSample))
    errorValue = dOutput - dOutputSample; % Compute the error
    disp("Error computed. Size: " + string(size(errorValue)));
end

end


