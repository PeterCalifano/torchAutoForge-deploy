#include <string>

namespace ptaf_deploy
{
    // SampleSegmentation in TensorRT/quickstart could be the prototype of something like:
    class EngineLoader
    {
      public:
        // CONSTRUCTOR
        EngineLoader(const std::string &engineFilename) : mEngineFilename(engineFilename) {};

        // GETTERS

        // SETTERS

        // PUBLIC METHODS

      public:
        // PUBLIC DATA MEMBERS
        std::string mEngineFilename; //!< Filename of the engine.

      protected:
    };
}
