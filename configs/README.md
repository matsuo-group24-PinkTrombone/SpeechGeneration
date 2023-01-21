設定ファイルに関するドキュメントをここに書きます。

## Hydra

基本的にhydraを使用することを想定し、`yaml`を用いて記述していくことを検討しています。
https://github.com/ashleve/lightning-hydra-template/tree/main/configs

https://hydra.cc/docs/intro/

### hydra.main

wip

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
