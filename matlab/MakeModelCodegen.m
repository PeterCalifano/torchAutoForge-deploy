function MakeModelCodegen(charPathToModelMat, ...
                        cellArgs, ...
                        charForwardFcnFilename, ...
                        charOutputPath, ...
                        kwargs)
    arguments
        charPathToModelMat  (1,:) char
        cellArgs            {iscell}
        charForwardFcnFilename 
        charOutputPath      (1,:) char = '.'
    end
    arguments
        kwargs.ui32BatchSize  (1,1) uint32 = 1
        kwargs.objCoderConfig (1,1) {mustBeA(kwargs.objCoderConfig, ["coder.MexCodeConfig", "double"])} = []
    end

    %% Setup
    if isempty(kwargs.objCoderConfig)
        cfg = coder.config('mex');
        cfg.TargetLang = 'C++';
        cfg.DeepLearningConfig = coder.DeepLearningConfig('mkldnn');
    end

    %% Write function code from template
    % Reference: https://www.mathworks.com/help//releases/R2021a/coder/ug/generate-generic-cc-code-for-deep-learning-networks.html
    charFcnTempl = sprintf("function out = RunModelInferenceCodegen(varModelInput) %%#codegen \n\npersistent network_model; \n" + ...
                        "%% Load model\nif isempty(network_model)" + ...
                        "\n\tnetwork_model = coder.loadDeepLearningNetwork('%s.mat');\nend" + ...
                        "\n\n%% Run inference\n" + ...
                        "out = predict(network_model, varModelInput, 'MiniBatchSize', %d);end", charPathToModelPath, kwargs.ui32BatchSize);

    % Write function code to file
    % TODO
    charWrapOutName = fullfile( charOutputPath, strcat(charForwardFcnFilename, ".m") );

    %% Call codegen command
    codegen -config cfg -args cellArgs charWrapOutName -report


end
