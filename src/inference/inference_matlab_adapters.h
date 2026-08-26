/**
 * @file inference_matlab_adapters.h
 * @brief MATLAB-safe conversions and facade helper entry points.
 */

#pragma once

#include <inference/inference_manager.h>
#include <inference/model_facade.h>
#include <inference/task_adapters.h>

#include <cmath>
#include <gtsam/base/Matrix.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace ptafdeploy::inference
{
    /**
     * @brief Convert a MATLAB-compatible shape vector to runtime dimensions.
     * @param shape Shape entries represented as MATLAB doubles.
     * @return Non-negative dimensions represented as signed 64-bit integers.
     * @throws std::invalid_argument If an entry is non-finite, fractional,
     *         negative, or outside the int64_t range.
     */
    [[nodiscard]] inline std::vector<int64_t> ConvertShapeVector(const gtsam::Vector& shape)
    {
        std::vector<int64_t> converted_shape;
        converted_shape.reserve(static_cast<size_t>(shape.size()));
        for (Eigen::Index i = 0; i < shape.size(); ++i)
        {
            const double value = shape(i);
            double integer_value = 0.0;
            const double fractional_value = std::modf(value, &integer_value);
            constexpr double kMinimumInt64 =
                static_cast<double>(std::numeric_limits<int64_t>::min());
            constexpr double kMaximumInt64Exclusive = -kMinimumInt64;
            if (!std::isfinite(value) || std::fpclassify(fractional_value) != FP_ZERO ||
                integer_value < kMinimumInt64 || integer_value >= kMaximumInt64Exclusive)
            {
                throw std::invalid_argument(
                    "Shape vector entries must be finite integer values representable as int64.");
            }
            if (integer_value < 0.0)
            {
                throw std::invalid_argument("Runtime shape vector entries must be non-negative.");
            }
            converted_shape.push_back(static_cast<int64_t>(integer_value));
        }

        return converted_shape;
    }

    /**
     * @brief Convert float values to a MATLAB-compatible double vector.
     * @param values Source float values.
     * @return Values represented as a GTSAM/Eigen double vector.
     */
    [[nodiscard]] inline gtsam::Vector MakeVector(const std::vector<float>& values)
    {
        gtsam::Vector vector_values(static_cast<Eigen::Index>(values.size()));
        for (size_t i = 0; i < values.size(); ++i)
        {
            vector_values(static_cast<Eigen::Index>(i)) = static_cast<double>(values[i]);
        }

        return vector_values;
    }

    /**
     * @brief Convert signed 64-bit values to a MATLAB-compatible double vector.
     * @param values Source integer values.
     * @return Values represented as a GTSAM/Eigen double vector.
     * @note Integers outside the exact IEEE-754 double range lose precision.
     */
    [[nodiscard]] inline gtsam::Vector MakeVector(const std::vector<int64_t>& values)
    {
        gtsam::Vector vector_values(static_cast<Eigen::Index>(values.size()));
        for (size_t i = 0; i < values.size(); ++i)
        {
            vector_values(static_cast<Eigen::Index>(i)) = static_cast<double>(values[i]);
        }

        return vector_values;
    }

    /**
     * @brief Convert a MATLAB-compatible double vector to single-precision values.
     * @param values Source double values.
     * @return Values converted to float without additional range policy.
     */
    [[nodiscard]] inline std::vector<float> MakeFloatValues(const gtsam::Vector& values)
    {
        std::vector<float> float_values;
        float_values.reserve(static_cast<size_t>(values.size()));
        for (Eigen::Index i = 0; i < values.size(); ++i)
        {
            float_values.push_back(static_cast<float>(values(i)));
        }

        return float_values;
    }

    /**
     * @brief Return an inference-manager input shape in wrapper-safe form.
     * @param manager Non-owning loaded manager pointer.
     * @param index Input index.
     * @return Model input dimensions as MATLAB-compatible doubles.
     * @throws std::invalid_argument If `manager` is null.
     */
    [[nodiscard]] inline gtsam::Vector GetInputShapeVector(CInferenceManager* manager,
                                                           const size_t index)
    {
        if (manager == nullptr)
        {
            throw std::invalid_argument("Inference manager pointer is null.");
        }

        return MakeVector(manager->GetInputInfo(index).shape);
    }

    /**
     * @brief Return an inference-manager output shape in wrapper-safe form.
     * @param manager Non-owning loaded manager pointer.
     * @param index Output index.
     * @return Model output dimensions as MATLAB-compatible doubles.
     * @throws std::invalid_argument If `manager` is null.
     */
    [[nodiscard]] inline gtsam::Vector GetOutputShapeVector(CInferenceManager* manager,
                                                            const size_t index)
    {
        if (manager == nullptr)
        {
            throw std::invalid_argument("Inference manager pointer is null.");
        }

        return MakeVector(manager->GetOutputInfo(index).shape);
    }

    /**
     * @brief Run one float input through an inference manager using wrapper-safe values.
     * @param manager Non-owning loaded manager pointer.
     * @param values Flattened input values.
     * @param shape Concrete non-negative input shape.
     * @return Flattened sole-output values as MATLAB-compatible doubles.
     * @throws std::exception If the pointer, shape, model contract, or execution is invalid.
     */
    [[nodiscard]] inline gtsam::Vector InferSingleFloatInputVector(CInferenceManager* manager,
                                                                   const gtsam::Vector& values,
                                                                   const gtsam::Vector& shape)
    {
        if (manager == nullptr)
        {
            throw std::invalid_argument("Inference manager pointer is null.");
        }

        return MakeVector(
            manager->InferSingleFloatInput(MakeFloatValues(values), ConvertShapeVector(shape)));
    }

    /**
     * @brief Return a model-facade input shape in wrapper-safe form.
     * @param model Non-owning loaded model-facade pointer.
     * @param index Input index.
     * @return Model input dimensions as MATLAB-compatible doubles.
     * @throws std::invalid_argument If `model` is null.
     */
    [[nodiscard]] inline gtsam::Vector GetModelInputShapeVector(CModelFacade* model,
                                                                const size_t index)
    {
        if (model == nullptr)
        {
            throw std::invalid_argument("Model facade pointer is null.");
        }

        return MakeVector(model->GetInputInfo(index).shape);
    }

    /**
     * @brief Return a model-facade output shape in wrapper-safe form.
     * @param model Non-owning loaded model-facade pointer.
     * @param index Output index.
     * @return Model output dimensions as MATLAB-compatible doubles.
     * @throws std::invalid_argument If `model` is null.
     */
    [[nodiscard]] inline gtsam::Vector GetModelOutputShapeVector(CModelFacade* model,
                                                                 const size_t index)
    {
        if (model == nullptr)
        {
            throw std::invalid_argument("Model facade pointer is null.");
        }

        return MakeVector(model->GetOutputInfo(index).shape);
    }

    /**
     * @brief Run one float input through a model facade using wrapper-safe values.
     * @param model Non-owning loaded model-facade pointer.
     * @param values Flattened input values.
     * @param shape Concrete non-negative input shape.
     * @return Flattened sole-output values as MATLAB-compatible doubles.
     * @throws std::exception If the pointer, shape, model contract, or execution is invalid.
     */
    [[nodiscard]] inline gtsam::Vector InferModelSingleFloatInputVector(CModelFacade* model,
                                                                        const gtsam::Vector& values,
                                                                        const gtsam::Vector& shape)
    {
        if (model == nullptr)
        {
            throw std::invalid_argument("Model facade pointer is null.");
        }

        return MakeVector(
            model->InferSingleFloatInput(MakeFloatValues(values), ConvertShapeVector(shape)));
    }

    /**
     * @brief Construct an owned float tensor from MATLAB-compatible vectors.
     * @param tensor_name Optional model tensor name.
     * @param values Flattened double input values.
     * @param shape Concrete non-negative input shape.
     * @return Wrapper-safe owned float32 tensor.
     */
    [[nodiscard]] inline SFloatTensor MakeFloatTensorFromVector(const std::string& tensor_name,
                                                                const gtsam::Vector& values,
                                                                const gtsam::Vector& shape)
    {
        return SFloatTensor{tensor_name, ConvertShapeVector(shape), MakeFloatValues(values)};
    }

    /**
     * @brief Return an owned float tensor shape in wrapper-safe form.
     * @param tensor Source tensor.
     * @return Tensor dimensions as MATLAB-compatible doubles.
     */
    [[nodiscard]] inline gtsam::Vector GetFloatTensorShapeVector(const SFloatTensor& tensor)
    {
        return MakeVector(tensor.shape);
    }

    /**
     * @brief Return owned float tensor values in wrapper-safe form.
     * @param tensor Source tensor.
     * @return Flattened values as MATLAB-compatible doubles.
     */
    [[nodiscard]] inline gtsam::Vector GetFloatTensorValuesVector(const SFloatTensor& tensor)
    {
        return MakeVector(tensor.values);
    }

    /**
     * @brief Convert a flattened HWC image into an owned NCHW float tensor.
     * @param tensor_name Model input name.
     * @param hwc_values Flattened HWC double values.
     * @param height Image height.
     * @param width Image width.
     * @param channels Image channel count.
     * @param scale Multiplicative normalization scale.
     * @param swap_rb Whether to exchange channels zero and two.
     * @return Owned float32 tensor shaped `[1,C,H,W]`.
     * @throws std::invalid_argument If the flattened image cardinality is invalid.
     * @throws std::overflow_error If the image cardinality overflows size_t.
     */
    [[nodiscard]] inline SFloatTensor MakeNchwFloatTensorFromHwcVector(
        const std::string& tensor_name, const gtsam::Vector& hwc_values, const size_t height,
        const size_t width, const size_t channels, const double scale, const bool swap_rb)
    {
        const size_t expected_values =
            CheckedMultiply(CheckedMultiply(height, width, "Image pixel count overflows size_t."),
                            channels, "Image value count overflows size_t.");
        if (static_cast<size_t>(hwc_values.size()) != expected_values)
        {
            throw std::invalid_argument(
                "HWC image value count does not match height*width*channels.");
        }

        return MakeNchwFloatTensorFromHwcAccessor(
            tensor_name, height, width, channels, static_cast<float>(scale), swap_rb,
            [&](const size_t index) { return hwc_values(static_cast<Eigen::Index>(index)); });
    }

    /**
     * @brief Decode generic detections into a MATLAB-compatible numeric matrix.
     * @param output Concrete tensor whose configured axis contains row attributes.
     * @param schema Generic attribute, box, objectness, and class-score mapping.
     * @param score_threshold Inclusive minimum final score as a MATLAB double.
     * @param max_detections Maximum score-sorted rows; zero keeps all.
     * @return One detection per row as `[center_x, center_y, width, height, score, class_id]`.
     * @throws std::exception If the threshold, tensor, schema, values, or boxes are invalid.
     * @note Class identifiers outside the exact IEEE-754 double range lose precision.
     */
    [[nodiscard]] inline gtsam::Matrix DecodeDetectionRowsMatrix(const SFloatTensor& output,
                                                                 const SDetectionRowSchema& schema,
                                                                 const double score_threshold,
                                                                 const size_t max_detections)
    {
        if (!std::isfinite(score_threshold) || score_threshold < 0.0 ||
            score_threshold > static_cast<double>(std::numeric_limits<float>::max()))
        {
            throw std::invalid_argument("Detection score threshold must be finite, non-negative, "
                                        "and representable as float.");
        }

        const std::vector<SDetection2D> detections = DecodeDetectionRows(
            output, schema, static_cast<float>(score_threshold), max_detections);
        gtsam::Matrix matrix(static_cast<Eigen::Index>(detections.size()), 6);
        for (size_t row = 0U; row < detections.size(); ++row)
        {
            const SDetection2D& detection = detections[row];
            const Eigen::Index matrix_row = static_cast<Eigen::Index>(row);
            matrix(matrix_row, 0) = static_cast<double>(detection.bounds.center.x);
            matrix(matrix_row, 1) = static_cast<double>(detection.bounds.center.y);
            matrix(matrix_row, 2) = static_cast<double>(detection.bounds.size.width);
            matrix(matrix_row, 3) = static_cast<double>(detection.bounds.size.height);
            matrix(matrix_row, 4) = static_cast<double>(detection.classification.score);
            matrix(matrix_row, 5) = static_cast<double>(detection.classification.class_id);
        }

        return matrix;
    }

    /**
     * @brief Shared-pointer overload of GetInputShapeVector().
     * @param manager Shared loaded manager.
     * @param index Input index.
     * @return Model input dimensions as MATLAB-compatible doubles.
     */
    [[nodiscard]] inline gtsam::Vector
    GetInputShapeVector(const std::shared_ptr<CInferenceManager>& manager, const size_t index)
    {
        return GetInputShapeVector(manager.get(), index);
    }

    /**
     * @brief Shared-pointer overload of GetOutputShapeVector().
     * @param manager Shared loaded manager.
     * @param index Output index.
     * @return Model output dimensions as MATLAB-compatible doubles.
     */
    [[nodiscard]] inline gtsam::Vector
    GetOutputShapeVector(const std::shared_ptr<CInferenceManager>& manager, const size_t index)
    {
        return GetOutputShapeVector(manager.get(), index);
    }

    /**
     * @brief Shared-pointer overload of InferSingleFloatInputVector().
     * @param manager Shared loaded manager.
     * @param values Flattened input values.
     * @param shape Concrete input shape.
     * @return Flattened sole-output values.
     */
    [[nodiscard]] inline gtsam::Vector
    InferSingleFloatInputVector(const std::shared_ptr<CInferenceManager>& manager,
                                const gtsam::Vector& values, const gtsam::Vector& shape)
    {
        return InferSingleFloatInputVector(manager.get(), values, shape);
    }

    /**
     * @brief Shared-pointer overload of GetModelInputShapeVector().
     * @param model Shared loaded model facade.
     * @param index Input index.
     * @return Model input dimensions as MATLAB-compatible doubles.
     */
    [[nodiscard]] inline gtsam::Vector
    GetModelInputShapeVector(const std::shared_ptr<CModelFacade>& model, const size_t index)
    {
        return GetModelInputShapeVector(model.get(), index);
    }

    /**
     * @brief Shared-pointer overload of GetModelOutputShapeVector().
     * @param model Shared loaded model facade.
     * @param index Output index.
     * @return Model output dimensions as MATLAB-compatible doubles.
     */
    [[nodiscard]] inline gtsam::Vector
    GetModelOutputShapeVector(const std::shared_ptr<CModelFacade>& model, const size_t index)
    {
        return GetModelOutputShapeVector(model.get(), index);
    }

    /**
     * @brief Shared-pointer overload of InferModelSingleFloatInputVector().
     * @param model Shared loaded model facade.
     * @param values Flattened input values.
     * @param shape Concrete input shape.
     * @return Flattened sole-output values.
     */
    [[nodiscard]] inline gtsam::Vector
    InferModelSingleFloatInputVector(const std::shared_ptr<CModelFacade>& model,
                                     const gtsam::Vector& values, const gtsam::Vector& shape)
    {
        return InferModelSingleFloatInputVector(model.get(), values, shape);
    }
} // namespace ptafdeploy::inference
