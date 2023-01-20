# 実験環境のセットアップ

本実験の環境は[Docker](https://docs.docker.com/)を用いて行います。**Pythonの依存関係以外は**すべてDockerイメージに保存されています。

## AWS環境でのセットアップ

[AWS環境の場合はセットアップスクリプトを作成してあります。](/scripts/setup-on-aws.sh)
AWSのJupyter Lab環境に入った後はターミナルを起動し、次のコマンドを実行してください。

```sh
wget https://raw.githubusercontent.com/matsuo-group24-PinkTrombone/SpeechGeneration/main/scripts/setup-on-aws.sh && sh ./setup-on-aws.sh
```

## Setup Docker

[公式のインストール方法に従ってセットアップしてください](https://docs.docker.com/get-docker/)

Note: Windows上では**wsl上ではなく**Docker Desktopを直接インストールすることを推奨します。

- Ubuntuの場合:
  Dockerの公式から便利なセットアップスクリプトが提供されています。

  ```sh
  curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
  ```

  Note: wsl環境でDockerデーモンを起動するためには`sudo service docker start`を実行してください。

  また、GPUを使うために NVIDIA Container Toolkitをインストールします。[インストール方法はNVIDIAのドキュメントを参照願います。](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

## Preparation for Experiment

### Prerequiements

- make
- git

### Build and Run Docker Image

プロジェクトをクローンし、Dockerのイメージをビルドして実行します。

```sh
git clone https://github.com/matsuo-group24-PinkTrombone/SpeechGeneration.git
```

```sh
make docker-build
make docker-run
```

- output

  ```sh
  root@90a59619c31f:/workspace$
  ```

### VSCode からDockerイメージ内で作業する

[VisualStudioCodeのRemote Developmentエクステンション](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack)を使用するこで起動しているDockerイメージ内で直接作業することができるようになります。

次の手順を経ることで起動中のDockerイメージ内部に入ることが出来ます。

1. VS Codeの左側のアイコン「リモートエクスプローラー」を開く
2. Dev Containersタブの中にある`speech-generation:latest`を選択し、`Open Folder in container`をクリックして接続。
3. `/workspace`をVSCodeで開いて作業を開始する。

### Pythonの依存関係をインストール

プロジェクトのPythonの依存関係はよく変化します。したがってPythonの依存関係はDocker Imageと分離しており、別途インストールする必要があります。

```sh
# /workspace
poetry install
```

(1/10 追記): Dockerのイメージ内にもPythonの依存関係をインストールしておく方針に変更しました。しかし、新しくプロジェクトに追加された依存関係が更新されていない可能性があるので上記のコマンドを実行することを推奨します。

## Run experiment

`src/train.py`を実行し、実験を開始しましょう！

```sh
python src/train.py
```
