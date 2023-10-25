from pydantic import BaseModel, Field

from arguments.model_config import ModelConfig
from arguments.pre_process_config import PreprocessConfig


class Config(BaseModel):
    model: ModelConfig = Field(default_factory=ModelConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
