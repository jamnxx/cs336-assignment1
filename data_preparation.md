---

# CS336 Assignment 1 数据集下载（Windows + Conda + cmd）

## 0. 前置准备

激活 Conda 环境：

```cmd
conda activate cs336
```

设置代理，端口 `7897`（自行查看自己电脑代理的SOCKS5/HTTP端口）：

```cmd
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897
```

可选：HuggingFace 镜像，配合 `huggingface_hub` 使用：

```cmd
set HF_ENDPOINT=https://hf-mirror.com
```

> 注意：`set` 只对当前 cmd 窗口有效，新开窗口要重新设置。

---

## 1. 进入数据目录

```cmd
cd C:\Users\Administrator\Desktop\cs336\assignment1-basics
if not exist data mkdir data
cd data
```

---

## 2. 开始前检查

```cmd
dir
```

如果某个文件只有 **1KB 左右**，说明之前 `curl` 没加 `-L`，下到的是重定向页面，需要删掉：

```cmd
del 文件名
```

如果文件已经很大但没下完，例如 `owt_train.txt.gz` 已经 1GB 以上，**不要删除**，后面用 `-C -` 断点续传。

---

## 3. 下载数据集

Windows 没有 `wget`，使用系统自带的 `curl`。

关键参数：

- `-L`：跟随重定向，必须加，否则只下到 1KB 跳转页
- `-C -`：断点续传
- `--retry 10`：失败自动重试
- `-O`：用远程文件名保存

### 3.1 TinyStories

```cmd
curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt

curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
```

### 3.2 OpenWebText 子集

```cmd
curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz

curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
```

如果中途报错：

```text
curl: (56) Failure when receiving data from the peer
```

**再次执行同一条命令即可**，`-C -` 会从断点继续下载。

---

## 4. 解压 `.gz` 文件

Windows 没有 `gunzip`，用 Python 解压：

```cmd
python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('owt_train.txt.gz','rb'), open('owt_train.txt','wb'))"

python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('owt_valid.txt.gz','rb'), open('owt_valid.txt','wb'))"
```

解压成功后，可选删除 `.gz`：

```cmd
del owt_train.txt.gz owt_valid.txt.gz
```

---

## 5. 检查文件

```cmd
dir
```

正确文件与大致大小：

| 文件                           | 大致大小   |
| ------------------------------ | ---------- |
| `TinyStoriesV2-GPT4-train.txt` | 约 2.1 GB  |
| `TinyStoriesV2-GPT4-valid.txt` | 约 22 MB   |
| `owt_train.txt.gz`             | 约 4.3 GB  |
| `owt_valid.txt.gz`             | 约 106 MB  |
| `owt_train.txt`                | 解压后生成 |
| `owt_valid.txt`                | 解压后生成 |

最终 `data` 目录应包含：

```text
data/
├── TinyStoriesV2-GPT4-train.txt
├── TinyStoriesV2-GPT4-valid.txt
├── owt_train.txt
└── owt_valid.txt
```

---

## 6. 备选方案：用 `huggingface_hub` 下载

如果 `curl` 反复中断，推荐用 `huggingface_hub`，自带断点续传和重试。

安装：

```cmd
pip install -U huggingface_hub
```

下载 TinyStories：

```cmd
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='roneneldan/TinyStories', filename='TinyStoriesV2-GPT4-train.txt', repo_type='dataset', local_dir='.')"

python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='roneneldan/TinyStories', filename='TinyStoriesV2-GPT4-valid.txt', repo_type='dataset', local_dir='.')"
```

下载 OpenWebText 子集：

```cmd
python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='stanford-cs336/owt-sample', filename='owt_train.txt.gz', repo_type='dataset', local_dir='.')"

python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='stanford-cs336/owt-sample', filename='owt_valid.txt.gz', repo_type='dataset', local_dir='.')"
```

同一个文件重复执行会自动续传。

---

## 7. 常见问题

| 问题                                                   | 解决                                                         |
| ------------------------------------------------------ | ------------------------------------------------------------ |
| `'wget' 不是内部或外部命令`                            | Windows 没有 wget，改用 `curl` 或 `python -m wget`           |
| 下载文件只有 1KB                                       | `curl` 没加 `-L`，删掉假文件，重新用 `curl -L -C - -O`       |
| `curl: (56) Failure when receiving data from the peer` | 网络中断，重复执行 `curl -L -C - ...` 续传                   |
| 下载很慢                                               | 设置 `set HF_ENDPOINT=https://hf-mirror.com` 并使用 `huggingface_hub` |
| 代理不生效                                             | 确认代理端口是 `7897`，或给 curl 加 `-x http://127.0.0.1:7897` |
| 解压失败                                               | 确认 `.gz` 文件已完整下载，再执行 Python 解压命令            |

---

## 8. 一键顺序参考

```cmd
conda activate cs336
set HTTP_PROXY=http://127.0.0.1:7897
set HTTPS_PROXY=http://127.0.0.1:7897

cd C:\Users\Administrator\Desktop\cs336\assignment1-basics
if not exist data mkdir data
cd data

curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl -L -C - --retry 10 --retry-delay 5 --connect-timeout 30 -O https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz

python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('owt_train.txt.gz','rb'), open('owt_train.txt','wb'))"
python -c "import gzip, shutil; shutil.copyfileobj(gzip.open('owt_valid.txt.gz','rb'), open('owt_valid.txt','wb'))"

```

---