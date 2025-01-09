# Demo for Llama 3.1 8B models with int8 quantization. Tested with NeuronX Distributed Inference Library.

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM
)
from neuronx_distributed_inference.modules.generation.sampling import prepare_sampling_params
from transformers import AutoTokenizer, GenerationConfig
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config, HuggingFaceGenerationAdapter

model_path = "/opt/ml/model/test_neuron_vllm/local-models/meta-llama/Llama-3.1-8B-Instruct/"
quantized_model_path = "/opt/ml/model/test_neuron_vllm/local-models/meta-llama/Llama-3.1-8B-Instruct/neuron-compiled-artifacts/int8-quantized/"

neuron_config = NeuronConfig(
    quantized=True,
    quantized_checkpoints_path=quantized_model_path,
    quantization_dtype="int8",
    quantization_type="per_tensor_symmetric"
)

config = LlamaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path)
)

# Quantize the model and save it to `quantized_checkpoints_path`.
NeuronLlamaForCausalLM.save_quantized_state_dict(model_path, config)
print(f"Saved the model to path {quantized_model_path}")

# Compile, load, and use the model.
model = NeuronLlamaForCausalLM(model_path, config)
model.load(model_path)
generation_model = HuggingFaceGenerationAdapter(model)
print("Load the quantized model")

print("Generating outputs")
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token[0]
prompts = ["I believe the meaning of life is", "The color of the sky is"]
sampling_params = prepare_sampling_params(batch_size=neuron_config.batch_size, top_k=[10, 5], top_p=[0.5, 0.9],  temperature=[0.9, 0.5])
inputs = tokenizer(prompts, padding=True, return_tensors="pt")
generation_config = GenerationConfig.from_pretrained(model_path)
generation_config_kwargs = {
        "do_sample": True,
        "top_k": 1,
        "pad_token_id": generation_config.eos_token_id[0],
    }
generation_config.update(**generation_config_kwargs)

outputs = generation_model.generate(
        inputs.input_ids,
        generation_config=generation_config,
        attention_mask=inputs.attention_mask,
        max_length=model.config.neuron_config.max_length,
        sampling_params=sampling_params,
    )
output_tokens = tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print("Generated outputs:")
for i, output_token in enumerate(output_tokens):
    print(f"Output {i}: {output_token}")