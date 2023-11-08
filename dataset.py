import numpy as np
from torch.utils.data import Dataset, DataLoader, default_collate

from arguments.config import Config
from dataset_preprocess.pair_word_voice import DataType, load_pair, pair_word_voice
from dataset_preprocess.word_mapping import WordMap
from voice_preprocess.LFR_features import build_LFR_features
from voice_preprocess.extract_features import extract_feature
from voice_preprocess.spec_argument import spec_augment


def pad_collate(batch):
    max_input_len = float('-inf')
    max_target_len = float('-inf')

    for elem in batch:
        feature, trn = elem
        max_input_len = max_input_len if max_input_len > feature.shape[0] else feature.shape[0]
        max_target_len = max_target_len if max_target_len > len(trn) else len(trn)

    for i, elem in enumerate(batch):
        feature, trn = elem
        input_length = feature.shape[0]
        input_dim = feature.shape[1]

        padded_input = np.zeros((max_input_len, input_dim), dtype=np.float32)
        padded_input[:input_length, :] = feature

        padded_target = np.pad(trn, (( 0,max_target_len - len(trn)),(0,0)), 'constant', constant_values=0).astype(
            np.float32)

        input_length = np.asarray([True if i < input_length else False for i in range(max_input_len)])

        output_length = np.asarray([True if i < trn.shape[0] else False for i in range(max_target_len)])

        batch[i] = (padded_input, padded_target, input_length, output_length)

    # sort it by input lengths (long to short)
    # batch.sort(key=lambda x: x[2], reverse=True)

    return default_collate(batch)


class AiShellDataset(Dataset):
    def __init__(self, ty: DataType, config: Config, load_root: str = "./", ):
        self.config = config
        self.word_map = WordMap.load(load_root)
        self.samples = load_pair(ty, load_root)
        print('loading {} {} samples...'.format(len(self.samples), ty.name))

    def __getitem__(self, i):
        sample = self.samples[i]
        wave = sample.voice_local
        trn = sample.out_seq

        feature = extract_feature(input_file=wave, dim=self.config.model.input_dim, cmvn=True)
        # zero mean and unit variance
        feature = (feature - feature.mean()) / feature.std()
        feature = spec_augment(feature)
        feature = build_LFR_features(feature, m=self.config.preprocess.lfr.m, n=self.config.preprocess.lfr.n)

        trn = np.asarray([self.word_map.to_on_hot(t) for t in trn])

        return feature, trn

    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    pair_word_voice()
    dataset = AiShellDataset(DataType.Test, Config(

    ))
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    for i, (input, target, length) in enumerate(loader):
        print(i, input, target, length)
