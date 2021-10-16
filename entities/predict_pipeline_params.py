import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from dataclasses import dataclass, field


@dataclass()
class LoggingParameters:

    path_to_config: str = field(default="./configs/logging.yaml")


@dataclass()
class PredictPipelineParams:
    #input_data_path: str
    model_path: str
    output_data_path: str
    output_target_path: str
    log_params: LoggingParameters


PredictPipelineParamsSchema = class_schema(PredictPipelineParams)


def read_predict_params(config_path: str) -> PredictPipelineParams:
    schema = PredictPipelineParamsSchema()
    with open(config_path, "r") as in_stream:
        #schema = PredictPipelineParamsSchema()
        return schema.load(yaml.safe_load(in_stream))
