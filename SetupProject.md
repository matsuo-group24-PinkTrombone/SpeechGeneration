# 実験環境のセットアップ

本実験の環境は[Docker](https://docs.docker.com/)を用いて行います。**Pythonの依存関係以外は**すべてDockerイメージに保存されています。

## AWS環境でのセットアップ

Not implemented yet...

## Setup Docker

[公式のインストール方法に従ってセットアップしてください](https://docs.docker.com/get-docker/)

- Ubuntuの場合
  Ubuntuの場合はDockerの公式から便利なセットアップスクリプトが提供されています。

  ```sh
  curl https://get.docker.com | sh \
  && sudo systemctl --now enable docker
  ```

  Note: wsl環境でDockerデーモンを起動するためには`sudo service docker start`を実行してください。

  また、GPUを使うために NVIDIA Container Toolkitをインストールします。[インストール方法はNVIDIAのドキュメントを参照願います。](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)

## Preparation for Experiment.

### Pull and Run Docker Image

プロジェクトをクローンし、Dockerのイメージを実行します。

```sh
git clone https://github.com/matsuo-group24-PinkTrombone/SpeechGeneration.git
```

```sh
sudo docker run -it \
    --gpus all \
    --mount type=bind,source="$(pwd)/SpeechGeneration",target=/workspace \
		-e LOCAL_UID=$(id -u $USER) \
		-e LOCAL_GID=$(id -g $USER) \
    gesonanko/speech-generation:latest
```

- output

  ```sh
  user@90a59619c31f:/workspace$
  ```

- Note: Dockerfileからローカルでビルドし、そのイメージを使用する事も出来ます。

  ```sh
  # in SpeeechGeneration project dir
  make docker-build
  make docker-run
  ```

  or

  ```sh
  # in SpeeechGeneration project dir
  sudo docker build . -t speech-generation:latest
  sudo docker run --args ...
  ```

- **Note: Docker Imageのルートパスワードは[Dockerfile](/Dockerfile)内の`RUN echo 'root:<pass>' | chpasswd` を参照してください。**

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
