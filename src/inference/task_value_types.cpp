/**
 * @file task_value_types.cpp
 * @brief Validation and representation conversion for inference task values.
 */

#include "task_value_types.h"

#include <cmath>
#include <numeric>
#include <stdexcept>

namespace ptafdeploy::inference
{
    namespace
    {
        void ValidateFiniteBoxValues(const float first, const float second, const float third,
                                     const float fourth)
        {
            if (!std::isfinite(first) || !std::isfinite(second) || !std::isfinite(third) ||
                !std::isfinite(fourth))
            {
                throw std::invalid_argument("Bounding-box values must be finite.");
            }
        }
    } // namespace

    SBoundingBox2D MakeBoundingBox2DFromCenterSize(const float center_x, const float center_y,
                                                   const float width, const float height)
    {
        ValidateFiniteBoxValues(center_x, center_y, width, height);
        if (width < 0.0F || height < 0.0F)
        {
            throw std::invalid_argument("Bounding-box size must be non-negative.");
        }

        return {{center_x, center_y}, {width, height}};
    }

    SBoundingBox2D MakeBoundingBox2DFromXyxy(const float min_x, const float min_y,
                                             const float max_x, const float max_y)
    {
        ValidateFiniteBoxValues(min_x, min_y, max_x, max_y);
        if (max_x < min_x || max_y < min_y)
        {
            throw std::invalid_argument("Bounding-box corner coordinates must be ordered.");
        }

        return MakeBoundingBox2DFromCenterSize(
            std::midpoint(min_x, max_x), std::midpoint(min_y, max_y), max_x - min_x, max_y - min_y);
    }

    std::vector<float> GetBoundingBox2DXyxy(const SBoundingBox2D& box)
    {
        const SBoundingBox2D validated = MakeBoundingBox2DFromCenterSize(
            box.center.x, box.center.y, box.size.width, box.size.height);
        const float half_width = validated.size.width * 0.5F;
        const float half_height = validated.size.height * 0.5F;

        return {validated.center.x - half_width, validated.center.y - half_height,
                validated.center.x + half_width, validated.center.y + half_height};
    }
} // namespace ptafdeploy::inference
