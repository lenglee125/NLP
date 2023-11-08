import numpy
from torch import nn, Tensor
from torch.utils.data import DataLoader

from arguments.config import Config
from dataset_preprocess.pair_word_voice import DataType


class SpeechTransformer(nn.Module):

    def __init__(self, d_model=320, ):
        super().__init__()
        self.output_extent = nn.Linear(4335, d_model, )
        self.transformer = nn.Transformer(d_model=d_model, batch_first=True)
        self.output_down = nn.Linear(d_model, 4335)
        self.output_softmax = nn.Softmax(dim=2)

        # init
        nn.init.xavier_normal(self.output_extent.weight)
        nn.init.xavier_normal(self.output_down.weight)

    def forward(self, src, tgt, length, outl):
        tgt = self.output_extent(tgt)

        trans = self.transformer.forward(src, tgt, src_key_padding_mask=length, tgt_key_padding_mask=outl)

        out = self.output_down.forward(trans)
        out = self.output_softmax(out)
        return out


def len_to_tensor(length: Tensor, max: int) -> Tensor:
    ls = [(True if i < n else False for i in range(max)) for n in length]


if __name__ == '__main__':
    import dataset

    model = SpeechTransformer()
    set = dataset.AiShellDataset(DataType.Train, Config(), load_root="../")

    loader = DataLoader(dataset=set, batch_size=32, shuffle=True, collate_fn=dataset.pad_collate
                        )

    it = iter(loader)

    input_v, output_w, inl, outl = next(it)

    ret = model.forward(input_v, output_w, inl, outl)

    print(ret.shape)
