def get_generated_text_and_cumlogprob(sequence, request_output):
    parameters = request_output.input.parameters
    generated_text = request_output.input.input_text if parameters.get(
        "return_full_text") else ""
    cum_log_prob = 0.0
    for token in sequence.tokens:
        generated_text += token.text
        cum_log_prob += token.log_prob
    return generated_text, cum_log_prob


@output_formatter
def slisa_output_formatter(request_output: TextGenerationOutput):
    """When multiple sequences are generated, then we hold off sending the result until the generation is finished.
    This is because, in case of best_of or beam_search, we would know the best sequence only at the end of generation.
    """
    if not request_output.finished:
        return ""

    result = []
    for _, sequence in request_output.sequences.items():
        text, cum_log_prob = get_generated_text_and_cumlogprob(sequence, request_output)
        result.append({
            "text": text,
            "score": cum_log_prob
        })
   
    return json.dumps(result, ensure_ascii=False)