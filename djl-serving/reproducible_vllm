args = EngineArgs(
    model='llava-hf/llava-v1.6-34b-hf', speculative_config=None, tokenizer='llava-hf/llava-v1.6-34b-hf', 
    skip_tokenizer_init=False, 
    tokenizer_mode=auto, revision=None, rope_scaling=None, 
    rope_theta=None, tokenizer_revision=None, 
    trust_remote_code=False, dtype=torch.bfloat16,
    max_seq_len=4096, download_dir=None, 
    load_format=LoadFormat.AUTO, 
    tensor_parallel_size=4, pipeline_parallel_size=1, 
    disable_custom_all_reduce=False, quantization=None, 
    enforce_eager=True, kv_cache_dtype=auto, 
    quantization_param_path=None, device_config=cuda, 
    decoding_config=DecodingConfig(guided_decoding_backend='outlines'), 
    observability_config=ObservabilityConfig(otlp_traces_endpoint=None), 
    seed=0, served_model_name=llava-hf/llava-v1.6-34b-hf, use_v2_block_manager=False, enable_prefix_caching=False)
)
engine = LLMEngine.from_engine_args(args)
sampling_params = SamplingParams(max_tokens=100)
prompt_inputs = {'prompt': '<|im_start|>user\n<image>\nWhat?s in this image?<|im_end|>\n', 'multi_modal_data': {'image': <PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=1097x180 at 0x7FF5A0DB3EE0>}}
engine.add_request(request_id=request_id,
                                    inputs=prompt_inputs,
                                    params=sampling_params,
                                    **request_params)

request_outputs = self.engine.step() # will keep running this until it is finished.
