from dataclasses import dataclass, field


@dataclass()
class LoggingParameters:

    path_to_config: str = field(default="entities/logging.yaml")