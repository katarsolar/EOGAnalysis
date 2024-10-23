from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict, Field
from src.


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')





class ModelConfig(_BaseConfig):

    encoder_model: str = 'resnet18'
    pretrained: str = True
    input_dim:int = 512
    hidden_dim_proj: int = 2048
    mid_dim_proj: int = 4096
    output_dim_proj: int = 256

    hidden_dim_pred: int = 256
    mid_dim_pred: int = 256
    output_dim_pred: int = 256



class DatasetConfig(_BaseConfig):
    dataset_path:str = "/data/combined_dataset.h5"
    transform:bool = False


class DataConfig(_BaseConfig):
    dataset_path:str = "/clothing_data/images"
    num_workers:int = 4
    pin_memory:bool = True
    batch_size:int = 32


class ExpConfig(_BaseConfig):
    track_in_clearml:bool = False
    trainer_config: Dict[str, Any] = Field(default_factory=dict)
    project_name:str = "ssl"
    experiment_name:str = "experiment1"
    tags:List[str] = []
    seed:int = 42