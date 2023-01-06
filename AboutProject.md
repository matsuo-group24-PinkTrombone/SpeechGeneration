# このプロジェクト内での開発について

ここにはこのリポジトリ内で開発するための基本的なツールやルールについて記述します。

## 依存関係のセットアップ

- Python3.9
  [Miniforge3を使用しています。](https://github.com/conda-forge/miniforge)
- make
  conda環境であれば`conda install make`でインストールすることができます。
- Python poetry
  conda環境の場合は仮想環境に直接インストールします。
  `conda install poetry`
  そうでない場合はpipによるインストールか[公式のインストール方法を利用してください](https://python-poetry.org)

### poetry

`pyproject.toml`を使用してpoetryが自動的に依存関係をインストールします。プロジェクトの直下で次のコマンドを実行してください。
このコマンドによって、通常の依存関係だけでなく開発に必要なパッケージもインストールされます。

```sh
poetry install
```

- パッケージの追加に関して
  `poetry add`を使用して依存モジュールを追加します。このコマンドは他のライブラリとの依存関係を解消し安全にインストールできた場合に`pyproject.toml`を編集します。

  新規に通常の依存関係を追加する場合は次のコマンドを実行します。

  ```sh
  poetry add <pkg>
  ```

  開発時のみに使用する依存関係(pytestなど)を追加する場合は`-G dev`オプションを付与します。

  ```sh
  poetry add -G dev <pkg>
  ```

## フォルダの役割

### src

実装の主要なコードは`src`ディレクトリに保存します。[`src`ディレクトリの詳細については`src/README.md`を参照願います。](/src/README.md)

### tests

テストコードを配置するディレクトリです。[詳細はtests/README.mdを参照願います。](/tests/README.md)

### configs

モデルのハイパーパラメータを記述したファイルを配置します。[詳細はconfigs/README.mdを参照ください。](/configs/README.md)

### その他

- dataディレクトリ
  学習用データを保存しておくディレクトリです。[README](/data/README.md)
- logs
  学習時のログの出力先です。
- scripts
  シェルスクリプトや汎用性の無い簡単なPythonスクリプトファイルを配置します。[README](/scripts/README.md)

## このプロジェクトで開発するために

主要な実装は全て`src`ディレクトリ内の適切な場所に行ってください。

### 開発のポイント

- 新規機能を実装した場合は、必ずそのテストコードを付与する
- 関数やクラスにはそのdocstringを簡潔に記述する。ただし、テストコードには必要な場合のみで良い。(書き方は近いうちにまとめます)
- `feature/機能名` ブランチを作成し、そこにcommitしたのち、プルリクエストを送信する。

## pre-commit

このプロジェクトでは、`pre-commit`を使用してファイルの整形し、Linterにかけています。
`pre-commit run -a`または`make format`を使用することで実行することができます。[実行の詳細については`.pre-commit-config.yaml`を参照ください](/.pre-commit-config.yaml)

## Make

[テストコードの実行やpre-commitなどのよく使用するコマンドはMakefileを使用して実行することができます。](/Makefile)

- make format
  ファイルを整形しリンターを適用します。
- make test / test-full
  テストコードを実行します。`test-full`では通常は実行時間がかかるためスキップするテストコードも全て実行します。
