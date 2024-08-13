import json
from typing import Dict
from djl_python.output_formatter import TextGenerationOutput, output_formatter

def get_generated_text(sequence, request_output):
    parameters = request_output.input.parameters
    generated_text = request_output.input.input_text if parameters.get(
        "return_full_text") else ""
    for token in sequence.tokens:
        generated_text += token.text
    return generated_text

def get_details_dict(request_output: TextGenerationOutput,
                     include_tokens: bool = True) -> Dict:
    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    if parameters.get("details", request_output.input.tgi_compat):
        final_dict = {
            "finish_reason": best_sequence.finish_reason,
            "generated_tokens": len(best_sequence.tokens),
            "inputs": request_output.input.input_text,
        }

        if include_tokens:
            final_dict["tokens"] = request_output.get_tokens_as_dict()

        if parameters.get("decoder_input_details"):
            final_dict["prefill"] = request_output.get_prompt_tokens_as_dict()
        if parameters.get("top_n_tokens", 0) > 0:
            final_dict["top_tokens"] = request_output.get_top_tokens_as_dict(
                request_output.best_sequence_index)

        return final_dict
    elif best_sequence.finish_reason == "error":
        return {"finish_reason": best_sequence.finish_reason}
    else:
        return {}


@output_formatter
def custom_jsonlines_output_formatter(request_output: TextGenerationOutput):
    """
    jsonlines output formatter

    :return: formatted output
    """
    tgi_compat = request_output.input.tgi_compat
    parameters = request_output.input.parameters
    best_sequence = request_output.sequences[
        request_output.best_sequence_index]
    next_token, _, last_token = best_sequence.get_next_token()
    token_dict = next_token.as_tgi_dict(
    ) if tgi_compat else next_token.as_dict()
    final_dict = {"token": token_dict}
    if last_token:
        generated_text = get_generated_text(best_sequence, request_output)
        final_dict["generated_text"] = generated_text
        details_dict = get_details_dict(request_output, include_tokens=False)
        if details_dict:
            final_dict["details"] = details_dict
    json_encoded_str = json.dumps(final_dict, ensure_ascii=False) + "\n"
    return json_encoded_str