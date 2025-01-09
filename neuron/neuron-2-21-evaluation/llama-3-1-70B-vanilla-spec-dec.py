# Demo for Vanilla Speculative decoding with NeuronX Distributed Inference.
# Target model: Llama 3.1 70B instruct
# Draft model: Llama 3.2 3B instruct

import copy

from transformers import AutoTokenizer, GenerationConfig

from neuronx_distributed_inference.models.config import NeuronConfig
from neuronx_distributed_inference.models.llama.modeling_llama import (
    LlamaInferenceConfig,
    NeuronLlamaForCausalLM
)
from neuronx_distributed_inference.utils.accuracy import get_generate_outputs
from neuronx_distributed_inference.utils.hf_adapter import load_pretrained_config

prompts = ["I believe the meaning of life is"]

model_path = "/opt/ml/model/test_neuron_vllm/local-models/meta-llama/Llama-3.1-70B-Instruct/"
draft_model_path = "/opt/ml/model/test_neuron_vllm/local-models/meta-llama/Llama-3.2-3B-Instruct"
compiled_model_path = "/opt/ml/model/test_neuron_vllm/local-models/meta-llama/Llama-3.1-70B-Instruct/neuron-compiled-artifacts/"
compiled_draft_model_path = "/opt/ml/model/test_neuron_vllm/local-models/meta-llama/Llama-3.2-3B-Instruct/neuron-compiled-artifacts/"

# Initialize target model.
neuron_config = NeuronConfig(
    speculation_length=5,
    trace_tokengen_model=False
)
config = LlamaInferenceConfig(
    neuron_config,
    load_config=load_pretrained_config(model_path)
)
model = NeuronLlamaForCausalLM(model_path, config)

# Initialize draft model.
draft_neuron_config = copy.deepcopy(neuron_config)
draft_neuron_config.speculation_length = 0
draft_neuron_config.trace_tokengen_model = True
draft_config = LlamaInferenceConfig(
    draft_neuron_config,
    load_config=load_pretrained_config(draft_model_path)
)
draft_model = NeuronLlamaForCausalLM(draft_model_path, draft_config)

# Compile and save models.
model.compile(compiled_model_path)
draft_model.compile(compiled_draft_model_path)

# Load models to the Neuron device.
model.load(compiled_model_path)
draft_model.load(compiled_draft_model_path)

# Load tokenizer and generation config.
tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side=neuron_config.padding_side)
generation_config = GenerationConfig.from_pretrained(model_path)

# Run generation.
_, output_tokens = get_generate_outputs(
    model,
    prompts,
    tokenizer,
    is_hf=False,
    draft_model=draft_model,
    generation_config=generation_config
)

print("Generated outputs:")
for i, output_token in enumerate(output_tokens):
    print(f"Output {i}: {output_token}")