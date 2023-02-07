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

## Hydra による実験設定の上書き

### 実験用configファイルを作成する

### コマンドライン引数でオーバライドする。

## 実験走らせてを放置する。

### `screen`による放置と再接続

### Dockerの再開
