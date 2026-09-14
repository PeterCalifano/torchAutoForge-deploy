# Installed Consumer Example

This standalone project verifies the installed package boundary. Install
`torchAutoForge-deploy`, then configure the consumer with that prefix:

```bash
cmake -S . -B build-producer \
  -Donnxruntime_DIR=/path/to/onnxruntime/lib/cmake/onnxruntime
cmake --build build-producer --parallel
cmake --install build-producer --prefix /tmp/ptafdeploy-install

cmake -S examples/consumer_project -B build-consumer \
  -DCMAKE_PREFIX_PATH="/tmp/ptafdeploy-install;/path/to/onnxruntime"
cmake --build build-consumer --parallel
./build-consumer/ptafdeploy_consumer
```

The example links only `autoforge_deploy::autoforge_deploy` and consumes a
public inference tensor helper. It is an explicit delivery acceptance input,
not a recursive project CTest.
