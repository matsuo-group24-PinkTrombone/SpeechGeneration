設定ファイルに関するドキュメントをここに書きます。基本的にhydraを使用することを想定し、`yaml`を用いて記述していくことを検討しています。[基本的な構成は`lightning-hydra-template`を参考に行います。](https://github.com/ashleve/lightning-hydra-template/tree/main/configs)

## configs

### train.yaml

`src/train.py`を実行した際に一番最初に読み込まれる設定ファイルです。
モデルの設定ファイル、データセットの設定ファイルといった、学習処理全体にかかわる設定をここで指定します。

### paths/\*.yaml

ルートディレクトリ、データディレクトリ、ログディレクトリなどを指定します。
他のconfigファイルからは`${paths.some_dir}`という形式で参照することが出来ます。

### hydra/\*.yaml

hydraモジュールの設定を行います。[詳細はhydraの公式ドキュメントを参照願います。](https://hydra.cc/docs/configure_hydra/intro/)

## Hydraについて

https://hydra.cc/docs/intro/

### hydra.main

このデコレーターを付与した関数がエントリーポイントとなります。[`hydra.main`の実行例は`src/train.py`をご確認ください。](/src/train.py)
ここで読み込まれた設定ファイルの中身は属性参照`cfg.value_name`によって取得することができます。

### hydra.utils.instantiate

Hydraはconfigファイルから直接Pythonのクラスをインスタンス化することができます。ここに示す例の他にも、[再帰的にinstantiateすること](https://hydra.cc/docs/advanced/instantiate_objects/overview/#recursive-instantiation)ができたり、[`_partial_`属性をconfigファイルに追記することで、部分的にinstantiateすること](https://hydra.cc/docs/advanced/instantiate_objects/overview/#partial-instantiation)も出来ます。これらは非常によく使用するため確認しておいてください。

https://hydra.cc/docs/advanced/instantiate_objects/overview/

- main.py

  ```py
  import hydra
  import omegaconf

  cfg = omegaconf.OmegaConf.load("config.yaml")
  instance = hydra.utils.instantiate(cfg)
  instance.print_args()
  ```

- src/mod.py

  ```py
  class Module:
      def __init__(self, arg1: float, arg2: int) -> None:
          self.args = arg1, arg2
      def print_args(self):
          print(*self.args)
  ```

- config.yaml

  ```yaml
  _target_: src.mod.Module
  arg1: 1.0
  arg2: 10
  ```
