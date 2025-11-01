function [fY] = RunModelInference(objModel, dX, kwargs)
arguments
    objModel    {mustBeA(objModel, "dlnetwork")}
    dX          {ismatrix, isnumeric} 
end
arguments
    kwargs.charInputShapeFormat (1,:) char {mustBeA(kwargs.charInputShapeFormat, ["string", "char"])} = "BC"
end


% Check the model is initialized
if ~objModel.Initialized
    error( ...
      'RunModelInference:NotInitialized', ...
      'Model has not been initialized. Call objModel.initialize() before using it.' ...
    );
end
% Check input shape format size against size of input dX (number of letter must be equal
dInputDims = size(dX);

if length(kwargs.charInputShapeFormat) ~= length(dInputDims)
    error('Format %s not consistent with input dX shape %s', mat2str(dInputDims));
end

% Encapsulate input into dlarray
objX = dlarray(dX, kwargs.charInputShapeFormat);

% Predict
objY = objModel.predict(objX);

% Return numeric output
fY = extractdata(objY);

end

