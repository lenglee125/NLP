import librosa
import librosa.feature
import numpy as np

from configs import SAMPLE_RATE


# [-0.5, 0.5]
def normalize(yt):
    yt_max = np.max(yt)
    yt_min = np.min(yt)
    a = 1.0 / (yt_max - yt_min)
    b = -(yt_max + yt_min) / (2 * (yt_max - yt_min))

    yt = yt * a + b
    return yt


# Acoustic Feature Extraction
# Parameters
#     - input file  : str, audio file path
#     - feature     : str, fbank or mfcc
#     - dim         : int, dimension of feature
#     - cmvn        : bool, apply CMVN on feature
#     - window_size : int, window size for FFT (ms)
#     - stride      : int, window stride for FFT
#     - save_feature: str, if given, store feature to the path and return len(feature)
# Return
#     acoustic features with shape (time step, dim)
def extract_feature(
    input_file,
    dim=80,
    cmvn=True,
    delta=False,
    delta_delta=False,
    window_size=25,
    stride=10,
):
    y, sr = librosa.load(input_file, sr=SAMPLE_RATE)

    # 过滤静音部分，并归一化
    yt, _ = librosa.effects.trim(y, top_db=20)
    yt = normalize(yt)

    # 计算FFT 窗口大小
    ws = int(sr * 0.001 * window_size)
    # 计算FFT 步长
    st = int(sr * 0.001 * stride)

    # 提取音频特征
    feat = librosa.feature.melspectrogram(
            y=yt, sr=sr, n_mels=dim, n_fft=ws, hop_length=st
        )
    feat = np.log(feat + 1e-6)

    feat = [feat]
    feat = np.concatenate(feat, axis=0)
    if cmvn:
        feat = (feat - feat.mean(axis=1)[:, np.newaxis]) / (feat.std(axis=1) + 1e-16)[
            :, np.newaxis
        ]

    return np.swapaxes(feat, 0, 1).astype("float32")
