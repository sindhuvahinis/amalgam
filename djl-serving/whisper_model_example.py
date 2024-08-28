import os
import logging
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

import  numpy as np
import soundfile as sf


from djl_python.inputs import Input
from djl_python.outputs import Output
from djl_python.encode_decode import decode



def parse_input(inputs: Input):
    batch = inputs.get_batches()
    audio_files = []
    for i, input_item in enumerate(batch):
        content_type = input_item.get_property("Content-Type")
        decoded_item = decode(input_item, content_type)
        # assuming it is text content type
        audio_file_path = decoded_item["inputs"][0]
        
        data, samplerate = sf.read(audio_file_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        audio_files.append(data)
    return audio_files


class WhisperModelService(object):
    def __init__(self):
        self.is_initialized = False
        self.pipe = None

    def initialize(self, properties: dict):
        #this code is not used, moved the logic to handle()
        model_id = properties.get("model_id")

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        print(f"Device: {device}")
        print(f"Torch Dtype: {torch_dtype}")

        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
        model.to(device)

        processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        )
        
        self.is_initialized = True

_service = WhisperModelService()

def handle(inputs: Input):

    if not _service.is_initialized:
        _service.initialize(inputs.get_properties())

    if inputs.is_empty():
        return None

    audio_files = parse_input(inputs)


    start_time = time.time()
    # "ValueError: We expect a single channel audio input for AutomaticSpeechRecognitionPipeline"
    result = _service.pipe(audio_files, batch_size=1)
    print(f"Result: {result}")

    #working code
    # dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
    # sample = dataset[0]["audio"]
    # result = pipe(sample)["text"]
    # print(f"Result: {result}")

    #ffmpeg issue
    # print(audio_files)
    # result = _service.pipe(audio_files, batch_size=1)
    # print(result)

    # Calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info("The time for running this program is %s", elapsed_time)
    outputs =  { "text" : result }

    return Output().add_as_json(outputs)