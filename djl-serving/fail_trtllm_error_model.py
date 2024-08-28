from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.tensorrt_llm import TRTLLMService
import random
import sys

_service = TRTLLMService()

def handle(inputs: Input) -> Output:
    """
    Handler function for the default TensorRT-LLM handler.

    :param inputs: (Input) a batch of inputs, each corresponding to a new request

    :return outputs (Output): a batch of outputs that contain status code, output text, and other information.
    """
    global _service
    if not _service.initialized:
        # stateful model
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        # initialization request
        return None
    
    if random.randint(1,10) == 9:
        sys.exit(-1)

    return _service.inference(inputs)
