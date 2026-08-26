/**
 * @file common_defs.h
 * @author PeterC (petercalifano.gs@gmail.com)
 * @brief
 * @version 0.1
 * @date 2025-07-20
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include <memory>
#include <cassert>

namespace deploy_common_defs
{
    /**
     * @brief Structure to hold input and output specifications for generic tensor input.
     *
     * @tparam INPUT_T Type of the input shapes.
     * @tparam OUTPUT_T Type of the output shapes.
     */
    template <typename INPUT_T = int64_t, typename OUTPUT_T = INPUT_T>
    struct SInputOutputSpecs
    {
        // Attributes
        std::vector<const char *> input_names;
        std::vector<const char *> output_names;

        std::vector<INPUT_T> input_shapes; // TODO extend to support multiple input shapes!
        std::vector<OUTPUT_T> output_shapes;

        size_t num_elements_linear_input_array = 0;  // Default to 0, can be set later
        size_t num_elements_linear_output_array = 0; // Default to 0, can be set later
        uint32_t num_input_tensors = 1;                // Default to 1 input tensor
        uint32_t num_output_tensors = 1;               // Default to 1 output tensor

        // Constructors
        SInputOutputSpecs() = default;
        SInputOutputSpecs(const std::vector<INPUT_T> &input_shapes,
                          const std::vector<OUTPUT_T> &output_shapes,
                          const std::vector<std::string> &input_names_in,
                          const std::vector<std::string> &output_names_in)
            : input_shapes(input_shapes), output_shapes(output_shapes)
        {

            // Define input/output names
            input_names.reserve(input_shapes.size());
            for (const auto &name : input_names_in)
            {
                input_names.emplace_back(name.c_str());
            }   
            output_names.reserve(output_shapes.size());
            for (const auto &name : output_names_in)
            {
                output_names.emplace_back(name.c_str());
            }   

            // Compute number of elements in linear input and output arrays
            // TODO extend to support multiple input tensors and apply function to comput number of ALL entries
            num_elements_linear_input_array = computeNumElementsLinearArray(input_shapes);
            num_elements_linear_output_array = computeNumElementsLinearArray(output_shapes);

            // Get number of input tensors
            num_input_tensors = static_cast<uint32_t>(input_names.size());
            num_output_tensors = static_cast<uint32_t>(output_names.size());
        }

        // Destructor
        ~SInputOutputSpecs() = default;

        // Copy/assignment constructors
        SInputOutputSpecs(const SInputOutputSpecs &other) = default;
        SInputOutputSpecs &operator=(const SInputOutputSpecs &other) = default;

        /**
         * @brief Compute the number of elements in a linearized tensor given its shape.
         *
         * @param shape_array The shape of the tensor as a vector of dimensions.
         * @param initial_value Initial accumulator value for callers that need a prefactor.
         * @return uint64_t The total number of elements in the tensor.
         */
        uint64_t computeNumElementsLinearArray(const std::vector<INPUT_T> &shape_array, 
                                               const uint64_t initial_value = 1) const
        {
            auto lambda_accumulate = [](uint64_t acc, const INPUT_T &shape)
            {
                return acc * static_cast<uint64_t>(shape);
            };

            return std::accumulate(shape_array.begin(),
                                   shape_array.end(),
                                   static_cast<uint64_t>(initial_value),
                                   lambda_accumulate);
        }
    };

    /**
     * @brief Structure to hold input and output specifications tailored for image data.
     *
     * @tparam INPUT_T Type of the input shapes.
     * @tparam OUTPUT_T Type of the output shapes.
     */
    template <typename INPUT_T, typename OUTPUT_T = INPUT_T>
    struct SImagesInputOutputSpecs : public SInputOutputSpecs<INPUT_T, OUTPUT_T>
    {
        // Attributes

        // Constructors
        SImagesInputOutputSpecs() = default;
        SImagesInputOutputSpecs(const uint32_t batch_size,
                                const uint32_t num_channels,
                                const uint32_t height,
                                const uint32_t width,
                                const std::vector<uint32_t> &output_shapes,
                                const std::vector<std::string> &input_names = {"input_0"},
                                const std::vector<std::string> &output_names = {"output_0"})
            : SInputOutputSpecs<INPUT_T, OUTPUT_T>()
        {
            // Input checks
            assert(!output_shapes.empty() && "Output shapes must not be empty.");

            // Determine number of input and output tensors
            this->num_input_tensors = static_cast<uint32_t>(input_names.size());
            this->num_output_tensors = static_cast<uint32_t>(output_names.size());

            assert(output_shapes.size() == this->num_output_tensors && "Output shapes must match the number of output tensors.");

            // Initialize the input and output specifications for image data
            this->input_names.reserve(this->num_input_tensors);
            this->input_shapes.reserve(this->num_input_tensors);

            // TODO check if there is something better than a for loop?
            for (uint32_t i = 0; i < this->num_input_tensors; ++i)
            {
                this->input_names.emplace_back(("input_" + std::to_string(i)).c_str());
                this->input_shapes.emplace_back(INPUT_T{batch_size, num_channels, height, width});
            }

            this->output_names.reserve(this->num_output_tensors);
            for (uint32_t i = 0; i < this->num_output_tensors; ++i)
            {
                this->output_names.emplace_back(("output_" + std::to_string(i)).c_str());
            }

            // Compute number of elements in linear input and output arrays
            this->num_elements_linear_input_array = this->computeNumElementsLinearArray(this->input_shapes, 1);
            this->num_elements_linear_output_array = this->computeNumElementsLinearArray(this->output_shapes, 1);
        }

        // Destructor
        ~SImagesInputOutputSpecs() = default;

        // Copy/assignment constructors
        SImagesInputOutputSpecs(const SImagesInputOutputSpecs &other) = default;
        SImagesInputOutputSpecs &operator=(const SImagesInputOutputSpecs &other) = default;
        // Move constructor
        SImagesInputOutputSpecs(SImagesInputOutputSpecs &&other) noexcept = default;
    };

}