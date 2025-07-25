function MakeModelCodegen(charPathToModelMat, ...
                        cellInputArgs, ...
                        charForwardFcnFilename, ...
                        charOutputPath, ...
                        kwargs)
    arguments
        charPathToModelMat     (1,:) char {mustBeFile, mustBeText}
        cellInputArgs          {iscell}
        charForwardFcnFilename (1,:) char {mustBeText} = 'RunModelInferenceCodegen'
        charOutputPath         (1,:) char {mustBeA(charOutputPath, ["char", "string"])} = '.'
    end
    arguments
        kwargs.ui32BatchSize  (1,1) uint32 = 1
        kwargs.objCoderConfig {mustBeA(kwargs.objCoderConfig, ["coder.MexCodeConfig", "double"])} = []
        kwargs.charBuildType (1,1) string {mustBeMember(kwargs.charBuildType, ["mex", "lib", "exe"])} = "mex"
        kwargs.bRunModelAnalysisCheck (1,1) logical {isscalar, islogical} = true
    end
    %% PROTOTYPE
    % MakeModelCodegen(charPathToModelMat, ...
    %                     cellInputArgs, ...
    %                     charForwardFcnFilename, ...
    %                     charOutputPath, ...
    %                     kwargs)
    % -------------------------------------------------------------------------------------------------------------
    %% DESCRIPTION
    % Automatic code generation for mex, lib and exe of model inference function for deep learning models.
    % -------------------------------------------------------------------------------------------------------------
    %% INPUT
    % charTargetFcnName  {mustBeText, mustBeA(charTargetFcnName, ["char", "string"])}
    % cellInputArgs      {mustBeA(cellInputArgs, "cell")}
    % objCoderConfig     {mustBeValidCodegenConfig(objCoderConfig)} = "mex";
    % -------------------------------------------------------------------------------------------------------------
    %% OUTPUT
    % [-]
    % -------------------------------------------------------------------------------------------------------------
    %% CHANGELOG
    % 25-07-2025    Pietro Califano     First version for model inference wrapper code generation.
    % -------------------------------------------------------------------------------------------------------------
    %% DEPENDENCIES
    % [-]
    % -------------------------------------------------------------------------------------------------------------

    %% Setup
    if isempty(kwargs.objCoderConfig)
        cfg = coder.config(kwargs.charBuildType, 'ecoder', true);
        cfg.TargetLang = 'C++';
        cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary='none');
        cfg.GenerateReport = true;
        cfg.LaunchReport = true;
        cfg.EnableJIT = true;
    else
        cfg = kwargs.objCoderConfig;
        fprintf("\nUsing provided coder configuration...\n");
    end

    %% Check input model
    if kwargs.bRunModelAnalysisCheck
        fprintf("\nRunning model analysis check...\n");
        strTmpData = load(charPathToModelMat);
        cellFieldNames = fieldnames(strTmpData);
        if numel(cellFieldNames) ~= 1
            error('The provided model file must contain exactly one variable corresponding to the network model! Found %d variables.', numel(cellFieldNames));
        end

        objModel = strTmpData.(cellFieldNames{1});
        objAnalysisRpt = analyzeNetworkForCodegen(objModel);
        disp(objAnalysisRpt)
        fprintf("\nModel analysis check completed.\n");
    else
        warning('Model analysis check is disabled. Make sure the model is compatible with code generation.');
    end

    if ~isfolder(charOutputPath)
        warning('Output path %s does not exist. Creating it...', charOutputPath);
        mkdir(charOutputPath);
    end
    mustBeFolder(charOutputPath);

    %% Write function code from template
    % Reference: https://www.mathworks.com/help//releases/R2021a/coder/ug/generate-generic-cc-code-for-deep-learning-networks.html
    fprintf('\nGenerating template for model caller function %s...\n', string(charForwardFcnFilename));

    charFcnTempl = sprintf("function out = %s(varModelInput)%%#codegen \n\npersistent network_model; \n" + ...
                        "%% Load model\nif isempty(network_model)" + ...
                        "\n\tnetwork_model = coder.loadDeepLearningNetwork('%s.mat');\nend" + ...
                        "\n\n%% Run inference\n" + ...
                        "out = predict(network_model, varModelInput, 'MiniBatchSize', %d);\nend", ...
                        charForwardFcnFilename, ...
                        charPathToModelMat, ...
                        kwargs.ui32BatchSize);

    % Write function code to file
    charWrapOutName = fullfile( charOutputPath, strcat(charForwardFcnFilename, ".m") );
    ui32fid = fopen(charWrapOutName, 'w');
    if ui32fid == -1
        error('Failed to open file for writing: %s', charWrapOutName);
    end
    fwrite(ui32fid, charFcnTempl);
    fclose(ui32fid);

    fprintf('\nGenerating src or compiled code for model caller function %s...\n', string(charForwardFcnFilename));
    %% Call codegen command
    fprintf("---------------------- CODE GENERATION EXECUTION: STARTED ---------------------- \n\n")
    % Execute code generation
    codegenCommands = {"-config", cfg, "-args", cellInputArgs, ...
                        "-o", charForwardFcnFilename, ...
                        "-d", fullfile(charOutputPath, "codegen_output"), charWrapOutName};
    codegen(codegenCommands{:});
    fprintf("\n---------------------- CODE GENERATION EXECUTION: COMPLETED ----------------------\n")

end
