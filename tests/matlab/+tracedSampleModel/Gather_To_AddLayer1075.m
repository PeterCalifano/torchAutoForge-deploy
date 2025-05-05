classdef Gather_To_AddLayer1075 < nnet.layer.Layer & nnet.layer.Acceleratable & nnet.layer.Formattable
    % Custom layer: Gather → FC → BatchNorm → PReLU → Add (3-output predict/forward)

    properties
        % ONNX subgraph constants
        gatherIndices       % double vector
        GemmAlpha           % double scalar
        GemmBeta            % double scalar
    end

    properties (Learnable)
        % Learnable parameters as plain single arrays
        skip_weighting_fc_we    % FC weights [Units×Gathered]
        skip_weighting_fc_bi    % FC bias   [Units×1]
        skip_weighting_bn_we    % BN scale  [1×Units]
        skip_weighting_bn_bi    % BN offset [1×Units]
        skip_weighting_bn_ru    % BN running mean     [1×Units]
        skip_weighting_bn__1    % BN running variance [1×Units]
        skip_weighting_prel_    % PReLU slopes        [1×Units]
    end

    methods
        function layer = Gather_To_AddLayer1075(name, onnxParams)
            % Constructor: extract numeric values
            layer.Name        = name;
            layer.Description = 'Gather→FC→BN→PReLU→Add';
            % Constants
            layer.gatherIndices = double(onnxParams.Nonlearnables.onnx__Gather_78(:));
            layer.GemmAlpha     = double(onnxParams.Gemmalpha1070);
            layer.GemmBeta      = double(onnxParams.Gemmbeta1071);
            % Learnable parameters
            fn = @(x) single(extractdata(x));
            layer.skip_weighting_fc_we = fn(onnxParams.Learnables.skip_weighting_fc_we);
            layer.skip_weighting_fc_bi = fn(onnxParams.Learnables.skip_weighting_fc_bi);
            layer.skip_weighting_bn_we = fn(onnxParams.Learnables.skip_weighting_bn_we);
            layer.skip_weighting_bn_bi = fn(onnxParams.Learnables.skip_weighting_bn_bi);
            layer.skip_weighting_bn_ru = fn(onnxParams.State.skip_weighting_bn_ru);
            layer.skip_weighting_bn__1 = fn(onnxParams.State.skip_weighting_bn__1);
            layer.skip_weighting_prel_ = fn(onnxParams.Learnables.skip_weighting_prel_);
        end

        function [Z, ZNumDims, state] = predict(layer, X, Y)
            % Convert dlarray inputs to numeric arrays
            if isa(X,'dlarray'), X = extractdata(X); end
            if isa(Y,'dlarray'), Y = extractdata(Y); end
            % 1) Gather
            Xg = X(:, layer.gatherIndices);
            % 2) FullyConnected (Gemm)
            A  = Xg.';  % [Gathered×Batch]
            B  = layer.skip_weighting_fc_we;
            C  = layer.skip_weighting_fc_bi;
            fc = layer.GemmAlpha * (B * A) + layer.GemmBeta * C;  % [Units×Batch]
            fc = fc.';  % [Batch×Units]
            % 3) BatchNorm
            mu    = layer.skip_weighting_bn_ru;
            var_  = layer.skip_weighting_bn__1;
            gamma = layer.skip_weighting_bn_we;
            beta_ = layer.skip_weighting_bn_bi;
            eps   = 1e-5;
            bn = (fc - mu) ./ sqrt(var_ + eps);
            bn = bn .* gamma + beta_;
            % 4) PReLU
            pos = max(0, bn);
            neg = min(0, bn);
            pr  = pos + layer.skip_weighting_prel_ .* neg;
            % 5) Add
            Z = Y + pr;
            % Outputs
            ZNumDims = ndims(Z);
            state.skip_weighting_bn_ru = layer.skip_weighting_bn_ru;
            state.skip_weighting_bn__1 = layer.skip_weighting_bn__1;
        end

        function [Z, ZNumDims, state] = forward(layer, X, Y)
            % Forward for training: same as predict with 3 outputs
            [Z, ZNumDims, state] = layer.predict(X, Y);
        end
    end
end
