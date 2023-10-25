from pydantic import BaseModel, Field



class LFRConfig(BaseModel):
    m:int=Field(default=4,description="保留的帧")
    n:int=Field(default=3,description="跳过的帧")

class PreprocessConfig(BaseModel):
    lfr: LFRConfig = Field(default_factory=LFRConfig, description="Low Frame Rate Configs")
