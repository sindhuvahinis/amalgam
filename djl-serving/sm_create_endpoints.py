from typing import Optional

import boto3
import sagemaker
from dataclasses import dataclass

models = {
    "llama-3-1-405b": "meta-llama/Meta-Llama-3.1-405B-FP8",
    "llama-3-1-70b": "meta-llama/Meta-Llama-3.1-70B",
    "llama-3-1-8B": "meta-llama/Meta-Llama-3.1-8B"
}


@dataclass
class SageMakerConfig:
    role: str = "arn:aws:iam::125045733377:role/AmazonSageMaker-ExecutionRole-djl"
    inference_image_uri: str = "763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124"
    region_name: str = 'us-east-1'
    instance_type: str = "ml.p5.48xlarge"
    model_suffix: str = None


@dataclass
class DJLServingConfig:
    model_name: str = None
    tp_degree: int = 1
    batch_size: int = 1
    enable_chunk_prefill: bool = False
    max_prefill_tokens: Optional[int] = None
    gpu_mem_util: Optional[int] = None
    hf_token: str = "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"
    rolling_batch: str = "lmi-dist"
    engine: str = "MPI"
    max_model_len: Optional[int] = None


class SageMakerEndpointCreationService:

    def __init__(self, server_config, sagemaker_config) -> None:
        self.server_config = server_config
        self.sm_config = sagemaker_config
        self.model_name = self.get_model_name()
        print(self.model_name)
        self.sm_client = self.create_sagemaker_client()
        self.env = self.get_env()
        print(self.env)

    def create_sagemaker_client(self):
        return boto3.client("sagemaker", self.sm_config.region_name)

    def get_env(self):
        env = {
            "HF_MODEL_ID": models[self.server_config.model_name],
            "OPTION_TENSOR_PARALLEL_DEGREE": str(self.server_config.tp_degree),
            "OPTION_ROLLING_BATCH": self.server_config.rolling_batch,
            "OPTION_MAX_ROLLING_BATCH_SIZE": str(self.server_config.batch_size),
            "OPTION_ENGINE": self.server_config.engine,
        }

        if self.server_config.hf_token:
            print(self.server_config.hf_token)
            env["HF_TOKEN"] = self.server_config.hf_token

        if self.server_config.enable_chunk_prefill:
            env["OPTION_ENABLE_CHUNKED_PREFILL"] = "true"

        if self.server_config.max_prefill_tokens:
            env["OPTION_MAX_ROLLING_BATCH_PREFILL_TOKENS"] = str(self.server_config.max_prefill_tokens)

        if self.server_config.gpu_mem_util:
            env["OPTION_GPU_MEMORY_UTILIZATION"] = str(self.server_config.gpu_mem_util)

        if self.server_config.max_model_len:
            env["OPTION_MAX_MODEL_LEN"] = str(self.server_config.max_model_len)

        return env

    def get_model_name(self):
        model_name_prefix = (f"{self.server_config.model_name}"
                             f"-tp{self.server_config.tp_degree}"
                             f"-bs{self.server_config.batch_size}")
        if self.sm_config.model_suffix:
            model_name_prefix = f"{model_name_prefix}-{self.sm_config.model_suffix}"
        return sagemaker.utils.name_from_base(model_name_prefix)

    def create_model(self):

        create_model_response = self.sm_client.create_model(
            ModelName=self.model_name,
            ExecutionRoleArn=self.sm_config.role,
            PrimaryContainer={
                "Image": self.sm_config.inference_image_uri,
                "Environment": self.env,
            },

        )
        model_arn = create_model_response["ModelArn"]

        print(f"Created Model: {model_arn}")

    def create_endpoint_config(self):
        endpoint_config_name = f"{self.model_name}-config"
        print(endpoint_config_name)

        endpoint_config_response = self.sm_client.create_endpoint_config(
            EndpointConfigName=endpoint_config_name,
            ProductionVariants=[
                {
                    "VariantName": "variant1",
                    "ModelName": self.model_name,
                    "InstanceType": "ml.p5.48xlarge",
                    "InitialInstanceCount": 1,
                    "ModelDataDownloadTimeoutInSeconds": 3600,
                    "ContainerStartupHealthCheckTimeoutInSeconds": 3600,
                    "RoutingConfig": {
                        'RoutingStrategy': 'LEAST_OUTSTANDING_REQUESTS'
                    },
                },
            ],
        )

        print(f"Created Endpoint config: {endpoint_config_response}")
        return endpoint_config_name

    def create_endpoint(self):
        self.create_model()
        endpoint_name = f"{self.model_name}-endpoint"
        endpoint_config_name = self.create_endpoint_config()
        create_endpoint_response = self.sm_client.create_endpoint(
            EndpointName=f"{endpoint_name}",
            EndpointConfigName=endpoint_config_name
        )
        print(f"Created Endpoint: {create_endpoint_response['EndpointArn']}")

djl_serving_config = DJLServingConfig(
    model_name="llama-3-1-70b",
    tp_degree=4,
    batch_size=16,
    rolling_batch="lmi-dist",
    engine="MPI",
    enable_chunk_prefill=True
)

sm_config = SageMakerConfig(
    role="arn:aws:iam::125045733377:role/AmazonSageMaker-ExecutionRole-djl",
    inference_image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/djl-inference:0.29.0-lmi11.0.0-cu124",
    region_name="us-east-1",
    instance_type="ml.p5.48xlarge",
    model_suffix="chunkp"
)

service = SageMakerEndpointCreationService(djl_serving_config, sm_config)
service.create_endpoint()