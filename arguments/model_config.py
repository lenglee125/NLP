from pydantic import BaseModel, Field


class EncoderConfig(BaseModel):
    layer_number: int = Field(default=6, description="Encoder 层数")
    head_number: int = Field(default=8, description="多头注意力的数量")
    model_dim: int = Field(default=512, description="模型的输入维度")
    inner_dim: int = Field(default=2048, description="FeedForward隐藏层维度")
    dropout: float = Field(default=0.01, description="Dropout 概率")
    position_encode_max_len: int = Field(default=5000, description="位置编码的最大长度")


class DecoderConfig(BaseModel):
    word_vec_dim: int = Field(default=512, description="词向量维度")
    layer_number: int = Field(default=6, description="Decoder 层数")


class ModelConfig(BaseModel):
    input_dim: int = Field(description="模型输入参数维度(LFR之前)", default=80)
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    decoder: DecoderConfig = Field(default_factory=DecoderConfig)
