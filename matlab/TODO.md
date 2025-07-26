# List of TODOs for MATLAB module of torchAutoForge-deploy

Useful links: https://it.mathworks.com/help/deeplearning/ref/dlnetwork.exportnetworktosimulink.html

- [ ] Implement script/function to automatically call deploy-python package and run export of a pth/pt (traced is easier) model to onnx

- [x] Implement automatic pipeline to import onnx model into MATLAB using existing functions

- [ ] Investigate and experiment with the Simulink export function. It should create a static graph from which the network can also be codegenerated.
