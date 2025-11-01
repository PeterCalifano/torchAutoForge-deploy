from enum import Enum

class DeployModelFormat(Enum):
    """
    Enumeration class listing supported model formats for deployment.
    """
    TORCH = "torch"
    ONNX = "onnx"
    TFLITE = "tflite"


class DeployModelConverter_backends(Enum):
    """
    Enumeration class listing supported backends for model conversion.
    """
    ONNX_TF = "onnx-tf"
    ONNX2TF = "onnx2tf"
    AI_EDGE_TORCH = "ai_edge_torch"

