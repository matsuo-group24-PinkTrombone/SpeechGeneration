# src -- プロジェクト構造について --

[これらはlightning-hydra-templateの構造を模倣しています。](https://github.com/ashleve/lightning-hydra-template)
具体的なコンポーネントの実装例に関してはそのリポジトリを参照願います。

学習のエントリーポイントは`src/train.py`に記述されます。
評価のエントリーポイントは`src/test.py`に記述されます。

## models

`src/models`の中にはモデルのクラスを記述します。

- 直下にはデータに適用可能で直接的に扱うクラスを配置します。

  例: vae_for_image.py

  ```py
  class VAEForImage(AbstractVAE):
      def __init__(self, args, kwds=...):
          super().__init__()
          ...

      def forward(self, x: Tensor) -> Tensor:
          ...
  ```

- `components`にはモデルのパーツを記述します。
  例: resblock.py

  ```py
  class ResBlock1d(nn.Module):
      def __init__(self, args, kwds=...):
          super().__init__()
          ...

  ```

- `abc`にはモデルのインターフェースだけを主に記述した、抽象クラスを定義します。
  [抽象クラスについては公式ドキュメントを参照願います。](https://docs.python.org/ja/3/library/abc.html)
  例: abc/encoder.py

  ```py
  from abc import ABC, abstractmethod
  import torch.nn as nn

  class Encoder(ABC, nn.Module):

      @abstractmethod #ABCを継承し、このデコレータをつけると継承先で実装を強制できる
      def encode(self, x:Tensor) -> Tensor:
          ...

  ```

## datamodules

`src/datamodules`の中にはデータ読み込みを行うクラスを定義します。
直下には`DataLoader`クラスなどの直接データを扱うことが可能なクラスを記述します。
`components`以下には`Dataset`クラスやその前処理関数などを記述します。

## utils

`src/utils`の中には便利関数などを記述します。
