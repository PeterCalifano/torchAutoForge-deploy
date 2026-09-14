# Image and inference-output utilities

The optional `images` and `inference_output` components provide image operations
and JSON output for native applications. Model loading and execution remain with
the inference facades. Neither component selects a model's preprocessing policy
or interprets its predictions.

## Build and installation

```bash
cmake -S . -B build-utils -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -Dautoforge_deploy_ENABLE_IMAGE_SUPPORT=ON \
  -Dautoforge_deploy_ENABLE_INFERENCE_OUTPUT=ON \
  -DCMAKE_INSTALL_PREFIX=/path/to/autoforge_deploy/install
cmake --build build-utils --parallel
cmake --install build-utils
```

Both options default to `OFF` and follow `BUILD_SHARED_LIBS`. An application
requests only the installed components it uses:

```cmake
cmake_minimum_required(VERSION 3.21)
project(image_output LANGUAGES CXX)
find_package(autoforge_deploy CONFIG REQUIRED COMPONENTS images inference_output)
add_executable(image_output main.cpp)
target_link_libraries(image_output PRIVATE
  autoforge_deploy::images autoforge_deploy::inference_output)
```

A lookup without components loads the existing core targets without discovering
OpenCV. Requesting `images` requires OpenCV core, imgproc, and imgcodecs. Requesting
an unavailable required component fails package discovery. RapidJSON is embedded
and private to `inference_output`; its headers are absent from public interfaces.
The installed license directory records its license and provenance.

## Native example

Save the following as `main.cpp` beside the consumer CMake file above. It writes
one raw tensor to standard output and annotates a supplied image. The example
values demonstrate serialization and do not represent a model prediction.

```cpp
#include <utils/images/images.h>
#include <utils/inference_output/inference_output.h>
#include <opencv2/imgproc.hpp>
#include <exception>
#include <iostream>

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: image_output INPUT OUTPUT.png\n";
        return 1;
    }
    try {
        namespace images = ptafdeploy::utils::images;
        namespace output = ptafdeploy::utils::inference_output;
        const ptafdeploy::inference::SFloatTensor prediction{
            "prediction", {1, 2}, {0.5F, 0.25F}};
        std::cout << output::Serialize(output::TensorValue(prediction)) << '\n';

        auto image = images::DecodeForOverlay(argv[1]);
        if (image.channels() == 1)
            image = images::Convert(image, cv::COLOR_GRAY2BGR);
        const double white = image.depth() == CV_16U ? 65535.0 : 255.0;
        images::DrawCrosshair(image, {image.cols * 0.5, image.rows * 0.25},
                             cv::Scalar(white, white, white, white), 17, 1);
        images::SavePng(argv[2], image);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
```

Configure and run the consumer:

```bash
cmake -S . -B build -DCMAKE_PREFIX_PATH=/path/to/autoforge_deploy/install
cmake --build build
build/image_output image.png overlay.png
```

Expected JSON fields are `name="prediction"`, `dtype="float32"`, `shape=[1,2]`,
and `values=[0.5,0.25]`. `SavePng` rejects unsupported sample types; supported
samples are uint8/uint16 with one, three, or four channels.

## Sequence publication

`PrepareOutput` accepts a new or empty directory and rejects an output location
that equals or contains the input. Callers select their non-recursive input list
before creating a descendant output directory and exclude concurrent writers.
These checks are not an interprocess lock.

`CReport` writes an initial incomplete report. Applications provide `SRunMetadata`
and append `SFrameRecord` values after each successful frame. Frame fields contain
raw tensors through `TensorValue` and optional application-specific values through
`SJsonValue`. Reserved run and frame fields cannot be overridden. Call
`Publish(true)` after the last frame, or `Publish(false, error)` with structured
failure information from the application's exception handler.

Publication streams committed records from disk into a temporary report and
replaces `predictions.json` by filesystem rename. A failed append prevents further
appends, and incomplete publication ignores its partial tail. A failed publication
retains the prior report and spool. Successful completion removes the spool;
cleanup failure does not invalidate the report. Atomic visibility depends on the
filesystem's rename semantics. Abrupt termination and power-loss recovery are not
provided. Working memory scales with the current record, not sequence length.

`TensorValue` accepts host tensor views and wrapper-safe float tensors. It preserves
integer precision and supports boolean, integer, float32, and float64 storage.
float16 and bfloat16 are rejected explicitly. Shape, byte count, host location,
boolean encoding, and numeric finiteness are checked before serialization.
JSON strings and keys must be valid UTF-8; invalid encodings are rejected.

## Source migration

`auxiliary/common_ops.h` and `deploy_aux` are replaced by `utils/filesystem.h` and
`ptafdeploy::utils`. `utils/value_parsing.h` moves to
`utils/parsing/value_parsing.h`, with namespace `ptafdeploy::utils::parsing`.
These are intentional source-compatibility breaks without forwarding headers.
Filesystem and parsing behavior is retained; both remain part of the core build.
Use a fresh install prefix when checking that retired headers are absent.

The centroiding adapter retains its existing schema version and coordinate
conventions. Its `raw_output` field retains the previous name/shape/values form;
the standalone generic tensor representation additionally records `dtype`.
Python and MATLAB retain their native image/JSON facilities and report the
unavailable wrapper execution-target priority as null.

Shared/static installed consumers, native utility tests, and C++/Python/MATLAB
sequence execution were checked on Linux. The
[development tracker](development/autoforge_deploy_upgrade_plan.md) records commands,
input provenance, results, and limitations. Six-image runs verify plumbing and
coordinate mappings; they do not establish model accuracy or comparative throughput.
ONNX Runtime must be available to the dynamic loader, for example through
`LD_LIBRARY_PATH=/path/to/onnxruntime/lib`.
