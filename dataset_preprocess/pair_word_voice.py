import enum
import os.path
import pickle
from typing import List, Dict, Tuple

from dataset_preprocess import TEXT_LOCAL, VOICE_LOCAL
from dataset_preprocess.word_mapping import WordMap


class DataType(enum.Enum):
    Train = 0
    Test = 1
    Dev = 2

    @staticmethod
    def from_str(s: str):
        if s == "dev":
            return DataType.Dev
        if s == "train":
            return DataType.Train
        if s == "test":
            return DataType.Test
        else:
            raise TypeError(f"未知的数据集划分类型`{s}`")


class VoicePair(object):
    def __init__(self, ty: DataType, out_seq: List[int], voice_local: str):
        """
        创建一个新的音频与文本对应表
        :param ty: 所属的分组：[train,test,dev]
        :param out_seq: 输出文本embedded
        :param voice_local: 输入的音频
        """
        self.voice_local = voice_local
        self.out_seq = out_seq
        self.ty = ty



def pair_word_voice(save_path: str = "./"):
    dataset: Dict[str, Tuple[DataType, str]] = dict()

    # 一级目录，根据数据集划分划分
    for path, ty in [(os.path.join(VOICE_LOCAL, d), d) for d in os.listdir(VOICE_LOCAL) if
                     os.path.isdir(os.path.join(VOICE_LOCAL, d))]:
        ty = DataType.from_str(ty)

        for path_wav_group in os.listdir(path):
            group_path = os.path.join(path, path_wav_group)
            for path_wav in os.listdir(group_path):
                file, ext = os.path.splitext(path_wav)
                wav_path = os.path.join(group_path, path_wav)
                dataset[file] = (ty, wav_path)

    with open(os.path.join(TEXT_LOCAL, "aishell_transcript_v0.8.txt"), "r", encoding="UTF-8") as f:
        lines = f.readlines()

    word_map = WordMap()
    pairs = [list(), list(), list()]
    for line in lines:
        line = line.strip()
        key = line[:16]
        words = list(line[16:].replace(" ", "").strip())
        word_embedding = [word_map.add_word(word) for word in words]
        word_embedding.append(word_map.get_eos())
        ty, wav = dataset[key]
        pairs[ty.value].append(VoicePair(ty, word_embedding, wav))

    word_map.save(save_path)
    with open(os.path.join(save_path, "pair.pickle"), "wb") as fs:
        pickle.dump(pairs, fs)

def load_pair(ty:DataType,work_path:str="./",)->List[VoicePair]:
    with open(os.path.join(work_path, "pair.pickle"), "rb") as fs:
        data = pickle.load(fs)
        return data[ty.value]
if __name__ == '__main__':
    pair_word_voice()
