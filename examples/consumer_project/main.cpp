/**
 * @file main.cpp
 * @brief Minimal consumer of the installed inference tensor API.
 */

#include <inference/inference_common.h>

#include <cstdint>
#include <iostream>
#include <vector>

/**
 * @brief Verify that the installed target exposes its public inference headers.
 * @return Zero when the public tensor helper reports the expected cardinality.
 */
int main()
{
    const std::vector<int64_t> shape{1, 11};
    const auto element_count = ptafdeploy::inference::ComputeElementCount(shape);

    std::cout << "autoforge_deploy tensor elements: " << element_count << '\n';
    return element_count == 11 ? 0 : 1;
}
