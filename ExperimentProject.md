# 実験を行うにあたって

ここには実験の行い方や注意事項をまとめます。

## 注意事項

### データセットと学習ログをホストOSに保存する

docker内部で作業した記録はコンテナおよびボリュームが削除されると同時消えてしまいます。また、データセットは容量が大きいためそれぞれのDockerボリュームに保存しておくことはあまり望まれません。[Dockerのbindマウントを使用することで、これらの問題を解決します。](https://matsuand.github.io/docs.docker.jp.onthefly/storage/bind-mounts/#start-a-container-with-a-bind-mount)

方法:

[Makefileのdocker-runコマンド](/Makefile)に引数を追加します。`<PathToDatasetDir>`と`<PathToLogDir>`は実行環境によって書き換えてください。

```sh
docker run -it \
    ... \
    --mount type=bind,source=<PathToDatasetDir>,target=/workspace/data \
    --monnt type=bind,source=<PathToLogDir>,target=/workspace/logs \
    ...
```

**Note: AWS上ではS3ストレージを直接マウントすることができないため、学習の最後にログデータを`/mnt/shared`にコピーするか、zipにしてダウンロードする必要があります！**

```sh
cp -r ./logs/** -d /mnt/shared/logs
```

```sh
zip -r logs.zip ./logs
```

## Hydra による実験設定の上書き

Hydraを利用してハイパーパラメータの管理を行っているため簡単に上書きし、実験を行うことができます。

### 実験用configファイルを作成する

実験を異なるハイパーパラメータで行いたい場合は、[`configs/experiment`の中にconfigファイルを作成してください。](/configs/experiment/example.yaml)
その中にデフォルトで入っている`example.yaml`を参考に中身を書いていきます。

configs/experiment/your_experiment.yaml

```yaml
# @package _global_
# 上記の`@package _global_`はpython src/train.py experiment=your_experimentと指定するために必要です。

defaults:
  # configファイルごとデフォルト値をオーバライドする場合は次のように書きます。
  - override /env: new_env_config.yaml

# 通常値を上書きする時はコンフィグの最上位からたどって指定します。
trainer:
  num_episode: 10
  gradient_clip_value: 50.0
```

実行する時は次のようにコマンドライン引数として指定します。

```sh
python src/train.py experiment=your_experiment
```

[今回のexperimentコンフィグによるオーバーライドはlightning-hydra-templateを参考にしています。](https://github.com/ashleve/lightning-hydra-template)

### コマンドライン引数でオーバライドする。

[Hydraはコマンドライン引数を用いて簡単にハイパーパラメータをオーバライドすることができます。](https://hydra.cc/docs/1.3/advanced/override_grammar/basic/)実験のログに何をオーバライドしたかは`.hydra`フォルダに記録されるので簡単にいくつか試したい場合に重宝します。指定する際はコンフィグの最上位からドット`.`区切りで指定します。

例としてエピソード数を変更してみます。

```sh
python src/train.py trainer.num_episode=8888
```

出力結果

```log
[2023-02-09 13:03:04,237][__main__][INFO] - Training configs:
...
trainer:
  num_episode: 8888
...
```

## 実験走らせてを放置する。

### `screen`による放置と再接続

### Dockerの再開

## Tensorboardを使って結果を見る

### AWSのJupyter Hub環境

AWS上のJupyter Hubの中でTensorboardを起動し利用する方法は現在わかっていません。
