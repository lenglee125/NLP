import os.path
__all__ = ["pair_word_voice","word_mapping","DATASET_LOCAL","TEXT_LOCAL","VOICE_LOCAL","PAIR_PICKLE"]
DATASET_LOCAL = r"D:\magic-ml\Speech-Transformer\data\data_aishell"
TEXT_LOCAL = os.path.join(DATASET_LOCAL, "transcript")
VOICE_LOCAL = os.path.join(DATASET_LOCAL,"wav")
PAIR_PICKLE = "pair.pickle"

