from djl_python.tensorrt_llm import TRTLLMService
from djl_python.inputs import Input
from djl_python.encode_decode import encode, decode
from djl_python.output_formatter import TextGenerationOutput
import logging
import json
import types

_service = TRTLLMService()


def custom_output_formatter(request_output: TextGenerationOutput):
    """
    Replace this function with your custom output formatter.

    Args:
        request_output (TextGenerationOutput): The request output

    Returns:
        (str): Response string

    """
    best_sequence = request_output.sequences[request_output.best_sequence_index]
    next_token, first_token, last_token = best_sequence.get_next_token()
    result = {"token_id": next_token.id, "token_text": next_token.text, "token_log_prob": next_token.log_prob}
    if last_token:
        result["finish_reason"] = best_sequence.finish_reason
    return json.dumps(result) + "\n"


def custom_input_formatter(self, inputs, tokenizer=None, output_formatter=None):
    """
    Replace this function with your custom input formatter.

    Args:
        data (obj): The request data, dict or string

    Returns:
        (tuple): input_data (list), input_size (list), parameters (dict), errors (dict), batch (list)
    """
    input_data = []
    input_size = []
    parameters = []
    errors = {}
    batch = inputs.get_batches()
    for i, item in enumerate(batch):
        try:
            content_type = item.get_property("Content-Type")
            input_map = decode(item, content_type)
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(f"Parse input failed: {i}")
            input_size.append(0)
            errors[i] = str(e)
            continue

        _inputs = input_map.pop("prompt", input_map)
        if not isinstance(_inputs, list):
            _inputs = [_inputs]
        input_data.extend(_inputs)
        input_size.append(len(_inputs))

        _param = input_map.pop("parameters", {})
        if "cached_prompt" in input_map:
            _param["cached_prompt"] = input_map.pop("cached_prompt")
        if not "seed" in _param:
            # set server provided seed if seed is not part of request
            if item.contains_key("seed"):
                _param["seed"] = item.get_as_string(key="seed")
        _param["output_formatter"] = custom_output_formatter
        for _ in range(input_size[i]):
            parameters.append(_param)

    return input_data, input_size, parameters, errors, batch


def handle(inputs: Input):
    """
    Default handler function
    """
    if not _service.initialized:
        # stateful model
        props = inputs.get_properties()
        _service.initialize(props)
        _service.parse_input = types.MethodType(custom_input_formatter, _service)

    if inputs.is_empty():
        # initialization request
        return None

    return _service.inference(inputs)
