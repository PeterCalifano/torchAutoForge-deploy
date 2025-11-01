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
        kwargs.charInputShapeFormat (1,:) char {mustBeA(kwargs.charInputShapeFormat, ["string", "char"])} = "BC"
        kwargs.objCoderConfig {mustBeA(kwargs.objCoderConfig, ["coder.MexCodeConfig", "coder.EmbeddedCodeConfig", "coder.CodeConfig", "double"])} = []
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
        cfg.TargetLangStandard = 'C++11 (ISO)';

        % Modify relevant default options
        cfg.GenerateReport = true;
        cfg.LaunchReport = true;
        cfg.GenCodeOnly = true;
        cfg.EnableAutoParallelization = true;
        cfg.IndentStyle = 'Allman';
        cfg.HighlightPotentialDataTypeIssues = true;
        cfg.GenerateCodeMetricsReport = true;
        cfg.EnableRuntimeRecursion = false;
        cfg.EnableSignedLeftShifts = false;
        cfg.CastingMode = "Standards";
        cfg.GenerateDefaultInSwitch = true;

        % DeepLearning specific
        cfg.DeepLearningConfig = coder.DeepLearningConfig(TargetLibrary='none');
        cfg.DeepLearningConfig.LearnablesCompression = 'None';

    else
        cfg = kwargs.objCoderConfig;
        fprintf("\nUsing provided coder configuration...\n");
    end

    % Handle input (remove extension if any
    [charDir, charName] = fileparts(charPathToModelMat);
    charPathToModelMat = fullfile(charDir, charName);

    %% Check input model
    if kwargs.bRunModelAnalysisCheck
        fprintf("\nRunning model analysis check...\n");
        strTmpData = load(strcat(charPathToModelMat, ".mat"));
        cellFieldNames = fieldnames(strTmpData);
        if numel(cellFieldNames) ~= 1
            error('The provided model file must contain exactly one variable corresponding to the network model! Found %d variables.', numel(cellFieldNames));
        end

            objModel = strTmpData.(cellFieldNames{1});
            objAnalysisRpt = analyzeNetworkForCodegen(objModel);
            disp(objAnalysisRpt)
            fprintf("\nModel analysis check completed.\n");
        
        catch ME
            fprintf(2, 'Error during model analysis check: %s\n', string(ME.message));
            charUsrInput = input('Do you want to continue with code generation? (y/n): ', 's');
            while ~ismember(lower(charUsrInput), {'y', 'n'})
                charUsrInput = input('Invalid input. Please enter "y" to continue or "n" to abort: ', 's');
            end
            if lower(charUsrInput) == 'n'
                warning('Code generation aborted due to model analysis check failure.');
                return;
            end
        end
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
                        "objY = network_model.predict(dlarray(varModelInput, '%s'));\n" + ...
                        "out = extractdata(objY);\nend", ...
                        charForwardFcnFilename, ...
                        charPathToModelMat, ...
                        kwargs.charInputShapeFormat);

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
