/**
 * @file build_info.cpp
 * @brief Compiled library version helpers shared by all consumer translation units.
 */

#include "config.h"

#include <cstdio>

void PrintVersion()
{
    std::printf("Application Version: %s\n", PROJECT_VERSION);
    std::printf("Full Version: %s\n", FULL_VERSION);
    std::printf("Version Details:\n");
    std::printf("Major: %d\n", PROJECT_VERSION_MAJOR);
    std::printf("Minor: %d\n", PROJECT_VERSION_MINOR);
    std::printf("Patch: %d\n", PROJECT_VERSION_PATCH);

    // Record this library translation unit's build time, not the consumer's.
    std::printf("Build Date: %s\n", __DATE__);
    std::printf("Build Time: %s\n", __TIME__);
}

std::string GetVersionString()
{
    return std::string(PROJECT_VERSION);
}
