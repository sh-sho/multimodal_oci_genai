# Multimodal_oci_genai

2025/05/23に開催されたOracle Developer Day 2025のデモコードです。


## 環境
以下の環境を前提に動作確認を行いました。
![alt text](./images/readme_images/oci.png)

### Oracle Database 23aiの構築
Oracle Database 23aiの構築方法は、OCI Tutorialの「[Oracle Database編 - Base Database Service (BaseDB) を使ってみよう](https://oracle-japan.github.io/ocitutorials/basedb)」か、「[Oracle Database - Oracle AI Vector Searchを使ってみよう](https://oracle-japan.github.io/ocitutorials/ai-vector-search)」を参考にしてください。

### 仮想マシンの実行環境
OCI Tutorial「[その3 - インスタンスを作成する](https://oracle-japan.github.io/ocitutorials/beginners/creating-compute-instance/)」を参考にVirtual Machineを作成してください。また、追加でPythonをインストールして実行環境を作成してください。Python 3.12.8で動作確認済みです。
```bash
$ python --version
Python 3.12.8
```

### 実行コード
デモで実行したコードはnotebook配下にあります。
```bash
$ cd src/notebook/
$ tree -a
.
├── images
├── multi_embed_model_agents.ipynb
├── multi_embed_model_image.ipynb
├── multi_embed_model_markdown.ipynb
├── multimodal_embedding.ipynb
├── single_embed_model_agents.ipynb
├── single_embed_model_image.ipynb
└── single_embed_model_markdown.ipynb

1 directory, 8 files
```

以下がそれぞれのデモに対応しているコードです。
#### Multimodal Embedding Model
* multimodal_embedding.ipynb

#### Multi Embedding Model
* multi_embed_model_markdown.ipynb
* multi_embed_model_image.ipynb
* multi_embed_model_agents.ipynb

#### Single Embedding Model
* single_embed_model_markdown.ipynb
* single_embed_model_image.ipynb
* single_embed_model_agents.ipynb
