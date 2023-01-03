# テストコードについて

テストファイルの作成や記述する際のポイントについて

## ポイント

- 必要十分な動作を網羅的にテスト
- テストコードは単純で一目でわかるもの
- テスト不可能な点についてプルリクエスト等で言及する

## ファイル構造

基本`src`の構造と一致させ、一つのソースファイルに一つのテストファイルが付属します。例えば`src/models/vae.py` に対応するテストファイルは`tests/models/test_vae.py`になります。

ある一つのファイルに対応しないテストファイルは、`tests`直下に保存してください。

また書き方の例は次のようになっています。

- 例
  src/some_code.py -> tests/test_some_code.py

  ```py
  from src import some_code as mod

  def test_func_of_some_code():
      f = mod.func_of_some_code

      assert f(...) == ...
      ...

  ```
