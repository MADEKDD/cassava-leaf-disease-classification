from typing import Dict, Any, Optional
from dataclasses import dataclass, field


#@dataclass()
#class HyperParameters:
#    n_estimators: Optional[int]
#    n_jobs: Optional[int]
#    max_iter: Optional[int]
#    random_state: int = field(default=10)
#
#    def to_dict(self) -> Dict[str, Any]:
#        dictionary = {}
#        for k, v in self.__dict__.items():
#            if v is not None:
#                dictionary[k] = v
#        return dictionary


@dataclass()
class ModelParameters:
    num_classes: int
    name: str
    batch_size: int
    num_workers: int
    epochs: int
    image_size: int
    pretrain: bool
    #hyper_params: HyperParameters
    