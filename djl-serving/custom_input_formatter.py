from djl_python import Input
from djl_python.input_parser import input_formatter
from djl_python.request_io import TextInput
from djl_python.encode_decode import decode

## **kwargs -> has most of the class attributes of TRTLLMService/HuggingFaceService
## example, for TRTLLMService, it has configs (serving.properties), tokenizer, rolling_batch

@input_formatter
def custom_input_formatter(input_item: Input, **kwargs):
    request_input = TextInput()
    content_type = input_item.get_property("Content-Type")
    input_map = decode(input_item, content_type)

    inputs = input_map.pop("inputs", input_map)
    params = input_map.pop("parameters", {})
    
    request_input.input_text = inputs
    request_input.parameters = params
    
    return request_input

