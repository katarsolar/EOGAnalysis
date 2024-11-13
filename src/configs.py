from typing import Any, Dict, List
from pathlib import Path
from pydantic import BaseModel, ConfigDict, Field
from src.constants import PROJECT_ROOT
root = Path(PROJECT_ROOT)


class _BaseConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')





class ModelConfig(_BaseConfig):

    encoder_model: str = 'resnet18'
    pretrained: bool = True
    input_dim:int = 512
    hidden_dim_proj: int = 2048
    mid_dim_proj: int = 4096
    output_dim_proj: int = 256
    learning_rate: float = 1e-3

    hidden_dim_pred: int = 256
    mid_dim_pred: int = 256
    output_dim_pred: int = 256



class DatasetConfig(_BaseConfig):
    dataset_path:str = str(root / 'data/combined_dataset.h5')
    transform:bool = False


class DataConfig(_BaseConfig):
    data_path:Path = root / 'data/combined_dataset.h5'
    make_embeddings:bool = False
    data_link:str = "https://drive.google.com/uc?id=1zmqESjLp2ViTyPYvlA-ck84kVVZNUFlf"
    num_workers:int = 4
    pin_memory:bool = True
    batch_size:int = 32


class ExpConfig(_BaseConfig):
    max_epochs:int = 10
    track_in_clearml:bool = False
    trainer_config: Dict[str, Any] = Field(default_factory=dict)
    project_name:str = "ssl"
    experiment_name:str = "experiment1"
    tags:List[str] = []
    seed:int = 42