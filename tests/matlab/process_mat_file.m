function [selected_matrix, output_filename] = process_mat_file(mat_filename)
    % PROCESS_MAT_FILE - Processes .mat file to extract and filter dMixedFeatMatrixAll
    %
    % Syntax: [selected_matrix, output_filename] = process_mat_file(mat_filename)
    %
    % Inputs:
    %   mat_filename - String or char array with the path to the .mat file
    %
    % Outputs:
    %   selected_matrix - Matrix containing only the selected features
    %   output_filename - Name of the saved filtered .mat file
    %
    % Description:
    %   1. Loads the .mat file and extracts dMixedFeatMatrixAll
    %   2. Removes columns 14 and 15
    %   3. Selects specific columns based on predefined feature names
    %   4. Saves the result to a new .mat file with "_filtered" suffix
    
    % Load the .mat file
    try
        data = load(mat_filename);
    catch ME
        error('Error loading .mat file: %s', ME.message);
    end
    
    % Check if dMixedFeatMatrixAll exists
    if ~isfield(data, 'dMixedFeatMatrixAll')
        error('Variable dMixedFeatMatrixAll not found in the .mat file');
    end
    
    % Extract dMixedFeatMatrixAll
    mixed_feat_matrix = data.dMixedFeatMatrixAll;
    
    % Delete columns 14 and 15
    if size(mixed_feat_matrix, 2) < 15
        error('Matrix does not have enough columns to remove columns 14 and 15');
    end
    mixed_feat_matrix(:, [15, 16]) = [];
    % mixed_feat_matrix(:, [14, 15]) = [];
    
    % Define feature names (corresponding to columns after removing 14 and 15)
    features_names = {'AreaB1', 'CentroidB1_x', 'CentroidB1_y', ...
                      'BoundingBoxB1_x', 'BoundingBoxB1_y', ...
                      'BoundingBoxB1_w', 'BoundingBoxB1_h', ...
                      'MajorAxisLength', 'MinorAxisLength', ...
                      'Eccentricity', 'Orientation', ...
                      'Circularity', 'EquivDiameter', ...
                      'Extent', 'PhaseAngle', ...
                      'AreaB2', 'CentroidB2_x', 'CentroidB2_y', ...
                      'FlagB2', 'SunDirectionAngleFromX'};
    
    % Define selected features to keep
    selected_feats = {'CentroidB1_x', 'CentroidB1_y', ...
                      'BoundingBoxB1_x', 'BoundingBoxB1_y', ...
                      'BoundingBoxB1_w', 'BoundingBoxB1_h', ...
                      'CentroidB2_x', 'CentroidB2_y', 'FlagB2', ...
                      'PhaseAngle', 'SunDirectionAngleFromX', ...
                      'EquivDiameter'};
    
    % Verify that the number of remaining columns matches the features_names length
    if size(mixed_feat_matrix, 2) ~= length(features_names)
        warning('Number of columns (%d) does not match number of feature names (%d)', ...
                size(mixed_feat_matrix, 2), length(features_names));
    end
    
    % Find indices of selected features
    selected_indices = [];
    for i = 1:length(selected_feats)
        idx = find(strcmp(features_names, selected_feats{i}));
        if ~isempty(idx)
            selected_indices = [selected_indices, idx];
        else
            warning('Feature "%s" not found in features_names', selected_feats{i});
        end
    end
    
    % Sort indices to maintain original column order
    selected_indices = sort(selected_indices);
    
    % Select only the desired columns
    selected_matrix = mixed_feat_matrix(:, selected_indices);
    
    % Display information about the processing
    fprintf('Original matrix size: %dx%d\n', size(data.dMixedFeatMatrixAll));
    fprintf('After removing columns 14 and 15: %dx%d\n', size(mixed_feat_matrix));
    fprintf('Final selected matrix size: %dx%d\n', size(selected_matrix));
    fprintf('Selected %d features out of %d available features\n', ...
            length(selected_indices), length(features_names));
    
    % Display selected feature names
    fprintf('Selected features:\n');
    for i = 1:length(selected_indices)
        fprintf('  Column %d: %s\n', i, features_names{selected_indices(i)});
    end
    
    % Create output filename by adding "_filtered" before the .mat extension
    [filepath, name, ext] = fileparts(mat_filename);
    output_filename = fullfile(filepath, [name '_filtered' ext]);
    
    % Save the filtered data to the new .mat file
    try
        % Create a structure to save with metadata
        filtered_data = struct();
        filtered_data.dMixedFeatMatrixAll_filtered = single(selected_matrix);
        filtered_data.selected_feature_names = selected_feats;
        filtered_data.selected_feature_indices = selected_indices;
        filtered_data.original_filename = mat_filename;
        filtered_data.processing_info = struct(...
            'removed_columns', [15, 16], ...
            'original_size', size(data.dMixedFeatMatrixAll), ...
            'final_size', size(selected_matrix), ...
            'processing_date', datestr(now));

        filtered_data.groundTruth = single(data.dProjCentreOfMass_uvAll);
        
        save(output_filename, '-struct', 'filtered_data');
        fprintf('\nFiltered data saved to: %s\n', output_filename);
        
    catch ME
        warning('Error saving filtered data: %s', ME.message);
        output_filename = '';
    end
end