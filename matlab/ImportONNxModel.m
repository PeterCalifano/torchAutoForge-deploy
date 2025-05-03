function objModel = ImportONNxModel(charModelfilePath, ...
                                    dModelInputSizes, ...
                                    charInputShape, ...
                                    dInputSample, ...
                                    dOutputSample)
arguments
    charModelfilePath (1,:) string {mustBeA(charModelfilePath, ["char", "string"])}
    dModelInputSizes  (1,:) double {isvector}
    charInputShape    (1,:) string = "BC"
    dInputSample      (:,:) double {ismatrix} = [] % Optional input sample
    dOutputSample     (:,:) double {ismatrix} = [] % Optional label sample
end

% Verify if the model file exists
if ~isfile(charModelfilePath)
    error('Specified model file has not been found. Check input path.')
end

% Import a PyTorch model into MATLAB
objModel = importNetworkFromONNX(charModelfilePath);

% Get random input sample if not provided
if isempty(dInputSample)
    dInputSample = rand(dModelInputSizes);
end

% Initialize model
X = dlarray(dInputSample, charInputShape);  
objModel = initialize(objModel, X);
analyzeNetwork(objModel);

% Test inference
dOutput = objModel.predict(X);

% Compute error if provided
if not(isempty(dOutputSample))
    errorValue = dOutput - dOutputSample; % Compute the error
    disp("Error computed. Size: " + string(size(errorValue)));
end

end
