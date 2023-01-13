import numpy as np
import torch

from src.models.components.posterior_encoder import PosteriorEncoder
from src.utils.feats_extract.log_mel_fbank import LogMelFbank

if __name__ == "__main__":
    # train_config
    batch_size = 1

    # Posterior Encoder
    # config
    aux_channels = 80  # スペクトログラムのチャネル数
    out_channels = 192  # エンコーダが出力するチャネル数
    hidden_channels = 192  # エンコーダ内の隠れ層のチャネル数
    kernel_size = 5  # エンコーダ内のWaveNetの畳み込み層のカーネルサイズ
    layers = 16  # WaveNetのレイヤ数
    stacks = 1  # WaveNetの積層数
    base_dilation = 1  # dilated convolutionのdilation
    global_channels = 192  # Number of global conditioning channels.
    dropout_rate = 0
    use_weight_norm = True  # Whether to apply weight norm.

    # instanse
    posterior_encoder = PosteriorEncoder(
        in_channels=aux_channels,
        out_channels=hidden_channels,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        layers=layers,
        stacks=stacks,
        base_dilation=base_dilation,
        global_channels=global_channels,
        dropout_rate=dropout_rate,
        use_weight_norm=use_weight_norm,
    )

    # LogMelSpectrogram
    # config
    fs = 16000  # サンプリングレート
    n_fft = 1024  # fft点数
    win_length = None  # 切り出した信号に対して適用する窓関数の幅(n_fft>=win_length)
    hop_length = 256  # 窓のずらし幅
    window = "hann"  # 窓関数の種類
    n_mels = 80  # フィルタバンクの分割数
    fmin = 80  # 周波数最小値
    fmax = 7600  # 周波数最大値

    # instanse
    feats_extractor = LogMelFbank(
        fs=fs,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        window=window,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
    )

    # 正弦波を生成
    signal = torch.tensor(
        [np.sin(2 * np.pi * 0.1 * x) for x in range(hop_length * 5 - 1)], dtype=torch.float32
    )
    batch = signal.unsqueeze(0)

    # ログメルスペクトログラムに変換
    feats, _ = feats_extractor(batch)
    feats = torch.transpose(feats, 1, 2)

    # featsの長さは仮に5で固定
    feats_lengths = torch.tensor([5] * batch_size)

    # RNNの隠れ状態
    g = torch.rand(batch_size, 192, 1)

    z, m_q, logs_q, y_mask = posterior_encoder(feats, feats_lengths, g=g)

    print(f"z:{z}")
    print(f"m_q:{m_q}")
    print(f"logs_q:{logs_q}")
