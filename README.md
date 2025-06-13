# FastText 数学领域文本分类器

本目录提供了一个利用 FastText 训练"数学领域 (math) / 其他 (other)"二分类模型的最小可复现实现，用于从大规模 Web 语料中筛选数学相关文本。

## 1. 依赖安装
```bash
# 建议使用虚拟环境
conda create -n fasttext_math python=3.10 -y
conda activate fasttext_math

# 安装 Python 依赖
pip install fasttext datasets tqdm langdetect
```

## 2. 快速开始
> 默认采样 10 万条正样本 + 10 万条负样本，首次运行需自动下载数据集。

在低带宽或磁盘空间有限的环境下，可使用 **流式加载** 或子集切片避免下载完整数据集：
```bash
# 只顺序读取前 5000 条正样本 + 5000 条负样本，不落盘大文件
python train.py --n_pos 100000 --n_neg 100000 --streaming

# 或者载入数据集 5% 的切片再随机采样
python train.py --pos_slice 'train[:5%]' --neg_slice 'train[:5%]' --n_pos 100000 --n_neg 100000
```

> ⚠️ 注意：`--streaming` 模式下不支持 `train[:5%]` 这类切片语法，如果同时传入切片参数会被忽略。想限制访问比例请去掉 `--streaming`。

完成后将看到类似输出：
```
Shuffling & splitting…
Split sizes: train 160,000, valid 20,000, test 20,000
Read 1M words
Number of words:  164748
Number of labels: 2
Progress: 100.0% words/sec/thread:  383353 lr:  0.000000 avg.loss:  0.282979 ETA:   0h 0m 0s
Validation set evaluation:
P@1: 0.960, R@1: 0.960, Number of examples: 13679
Test set evaluation:
P@1: 0.960, R@1: 0.960, Number of examples: 6836
Model saved to data_fasttext\math_cls.bin
```

### 常用参数
| 参数 | 说明 | 默认 |
| ---- | ---- | ---- |
| `--n_pos` | open-web-math 采样数量 | 100000 |
| `--n_neg` | fineweb 采样数量 | 100000 |
| `--valid_ratio` | 验证集占比 | 0.1 |
| `--test_ratio` | 测试集占比 | 0.1 |
| `--out_dir` | 输出目录（txt 文件和模型） | `data_fasttext` |


## 3. 开发与实现细节

### 3.1 数据处理
- 对每条文本仅进行了轻量级清洗：`normalize()` 合并连续空白字符并去除首尾空格，不做分词或符号过滤，以保留公式。
- 训练数据格式符合 FastText 要求：`__label__<tag> <text>`（UTF-8，单行）。
- 数据来源：
  - 正样本：`open-web-math/open-web-math`
  - 负样本：`HuggingFaceFW/fineweb`
- 两种下载策略：
  1. **子集切片**：`--pos_slice/--neg_slice` 仅下载指定百分比后再随机采样。
  2. **流式读取**：`--streaming` 边遍历边采样，无需完整下载。

### 3.2 采样与划分
- `sample_indices()` 依据 `--seed` 随机抽取 `n_pos`/`n_neg`。
- 合并后 `random.shuffle` 全局打乱，再按 `--valid_ratio`/`--test_ratio` 划分。

### 3.3 训练参数
使用 `fasttext.train_supervised()`，核心超参：

| 超参 | 值 | 备注 |
| ---- | --- | ---- |
| `lr` | 0.5 | 学习率 |
| `epoch` | 10 | 轮数 |
| `wordNgrams` | 2 | N-gram 范围 |
| `dim` | 200 | 词向量维度 |
| `thread` | `os.cpu_count()` | 并行线程 |

训练输出：
- `math_cls.bin` — FastText 二进制模型


## 4. 结果文件

运行结束后，目录结构示例：
```
data_fasttext/
├── all.txt        # 未划分前的混合数据
├── train.txt
├── valid.txt
├── test.txt
└── math_cls.bin   # FastText 模型
```

## 5. 推理与结果导出

### 5.1 单句预测
```python
import fasttext
model = fasttext.load_model("data_fasttext/math_cls.bin")
text = "Let f(x) be a continuous function defined on the interval ..."
print(model.predict(text))  # (['__label__math'], [0.98])
```

也可以直接使用 `infer.py` 命令行：
```bash
python infer.py --model data_fasttext/math_cls.bin --text "Let f(x) be a continuous function defined on the interval ..."

# 输出示例
# [SINGLE] label=math, prob=0.8274
# text=Let f(x) be a continuous function defined on the interval ...
```

### 5.2 重新打标 `__label__other` 样本
`infer.py` 会扫描训练生成的 `data_fasttext/test.txt`，顺序选取前 **N** 条以 `__label__other` 开头的记录，并使用 `math_cls.bin` 重新预测其标签。结果以 **JSONL** 格式保存，便于后续处理或提交。

```bash
# 重新打标 5 000 条 "other" 样本，并写入 fineweb_relabelled_5000.jsonl
python infer.py \
  --model data_fasttext/math_cls.bin \
  --input data_fasttext/test.txt \
  --output data_fasttext/relabelled_5000.jsonl \
  --n 5000
```

生成的 `data_fasttext/relabelled_5000.jsonl` 中每行都是一个 JSON 对象，例如：

```json
{"text": "Let f(x) be a continuous function defined on the interval ...", "label": "math", "prob": 0.8274}
```

## 6. 参考与致谢
- 数据集: [open-web-math](https://huggingface.co/datasets/open-web-math/open-web-math)、[fineweb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
- 工具: [fastText](https://fasttext.cc/), [🤗 Datasets](https://github.com/huggingface/datasets)
