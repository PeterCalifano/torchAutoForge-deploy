/**
 * @file task_value_types.h
 * @brief Wrapper-safe value types composed by inference task adapters.
 */

#pragma once

#include <cstdint>
#include <vector>

namespace ptafdeploy::inference
{
    /** @brief A unit-agnostic two-dimensional point. */
    struct SPoint2D
    {
        /** @brief Horizontal coordinate. */
        float x{0.0F};
        /** @brief Vertical coordinate. */
        float y{0.0F};
    };

    /** @brief A non-negative unit-agnostic two-dimensional extent. */
    struct SSize2D
    {
        /** @brief Non-negative horizontal extent. */
        float width{0.0F};
        /** @brief Non-negative vertical extent. */
        float height{0.0F};
    };

    /**
     * @brief A two-dimensional box stored canonically as center plus size.
     * @invariant Consumers keep center values finite and size values finite and non-negative.
     */
    struct SBoundingBox2D
    {
        /** @brief Box center in the owning integration's coordinate system. */
        SPoint2D center{};
        /** @brief Box extent in the same unit as the center. */
        SSize2D size{};
    };

    /** @brief A scored two-dimensional feature location. */
    struct SFeature2D
    {
        /** @brief Feature location in the owning integration's coordinate system. */
        SPoint2D position{};
        /** @brief Unit-agnostic feature score; defaults to one when absent. */
        float score{1.0F};
    };

    /** @brief A zero-based class identifier and its score. */
    struct SClassScore
    {
        /** @brief Zero-based class offset, or negative when no class is assigned. */
        int64_t class_id{-1};
        /** @brief Unit-agnostic classification score. */
        float score{0.0F};
    };

    /** @brief A geometric detection composed from bounds and classification. */
    struct SDetection2D
    {
        /** @brief Geometric support of the detection. */
        SBoundingBox2D bounds{};
        /** @brief Selected class and score. */
        SClassScore classification{};
    };

    /** @brief Supported representations of four box attributes in tensor rows. */
    enum class EBoundingBoxEncoding
    {
        /** @brief Attributes are `[center_x, center_y, width, height]`. */
        center_xywh,
        /** @brief Attributes are `[min_x, min_y, max_x, max_y]`. */
        corners_xyxy
    };

    /**
     * @brief Construct a canonical box from center and size values.
     * @param center_x Horizontal center coordinate.
     * @param center_y Vertical center coordinate.
     * @param width Non-negative box width.
     * @param height Non-negative box height.
     * @return Box stored as center plus size.
     * @throws std::invalid_argument If a value is non-finite or a size is negative.
     */
    [[nodiscard]] SBoundingBox2D MakeBoundingBox2DFromCenterSize(float center_x, float center_y,
                                                                 float width, float height);

    /**
     * @brief Construct a canonical box from ordered corner coordinates.
     * @param min_x Minimum horizontal coordinate.
     * @param min_y Minimum vertical coordinate.
     * @param max_x Maximum horizontal coordinate.
     * @param max_y Maximum vertical coordinate.
     * @return Box stored as center plus size.
     * @throws std::invalid_argument If a value is non-finite or corners are unordered.
     */
    [[nodiscard]] SBoundingBox2D MakeBoundingBox2DFromXyxy(float min_x, float min_y, float max_x,
                                                           float max_y);

    /**
     * @brief Convert a canonical box to `[min_x, min_y, max_x, max_y]`.
     * @param box Box with finite center and non-negative finite size.
     * @return Ordered corner coordinates.
     * @throws std::invalid_argument If the stored box is invalid.
     */
    [[nodiscard]] std::vector<float> GetBoundingBox2DXyxy(const SBoundingBox2D& box);
} // namespace ptafdeploy::inference
