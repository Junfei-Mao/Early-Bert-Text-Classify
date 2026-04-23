# BERT Text Classification

怀念手撸代码的日子吗？那就来看看这个：
一个基于 `bert-base-chinese` 的中文文本分类项目，适合做工单分派、诉求归类、短文本分类等任务。

这个版本已经做过一轮适合 GitHub 发布的整理，重点包括：

- 去掉了硬编码 GPU 和本地模型路径
- 训练、预测、Flask API 入口都支持配置化使用
- 标签不再写死在代码里，支持从数据自动推断并持久化到模型目录
- 增加了 `.gitignore` / `requirements.txt` / 示例数据，避免把模型权重和缓存直接提交到 GitHub

## 项目结构

```text
bert_text_classify/
├── bert_for_cls.py          # 模型定义（BERT + BiLSTM + attention pooling）
├── text_classify.py         # 训练 / 评估入口
├── predict.py               # 单条预测入口
├── app.py                   # Flask API 示例
├── text_processor.py        # 数据处理、标签加载、特征构造
├── requirements.txt
├── examples/
│   └── sample_train.jsonl
└── start.sh
```

## 环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 数据格式

训练集、验证集、测试集默认放在 `data/` 目录下，文件名分别为：

- `train.tsv`
- `dev.tsv`
- `test.tsv`

虽然扩展名是 `.tsv`，但当前代码实际读取的是“每行一个 JSON 对象”的 `jsonl` 格式，例如：

```json
{"title": "关于小区垃圾清理问题", "content": "市民反映小区门口存在垃圾堆积，希望尽快处理。", "label": "环卫所"}
```

字段说明：

- `title`: 标题，可为空
- `content`: 正文
- `label`: 分类标签

## 训练

最简训练命令：

```bash
python text_classify.py \
  --data_dir ./data \
  --model_name_or_path bert-base-chinese \
  --output_dir ./output \
  --do_train true \
  --do_eval true
```

如果 `output/` 目录已经存在且非空，需要显式覆盖：

```bash
python text_classify.py \
  --data_dir ./data \
  --model_name_or_path bert-base-chinese \
  --output_dir ./output \
  --do_train true \
  --overwrite_output_dir true
```

也可以直接：

```bash
./start.sh --data_dir ./data --model_name_or_path bert-base-chinese --output_dir ./output
```

## 预测

```bash
python predict.py \
  --model_dir ./output \
  --title "关于小区地下室渗水问题" \
  --content "市民来电反映地下室长期渗水，墙体发霉，希望尽快协调处理。"
```

输出示例：

```json
{"label": "嘉禾社区", "label_id": 18, "confidence": 0.973214}
```

交互式预测：

```bash
python predict.py --model_dir ./output --interactive
```

## Flask API

启动：

```bash
python app.py
```

请求示例：

```bash
curl "http://127.0.0.1:5000/api/predict?content=小区门口垃圾很多，希望尽快清理"
```

如果模型不在默认目录，可以这样启动：

```bash
MODEL_DIR=./output DEVICE=cpu python app.py
```

## 标签管理

训练时会优先按下面顺序加载标签：

1. `--labels` 指定的标签文件
2. `output/label_config.json`
3. `model_name_or_path/label_config.json`
4. 从 `data/train.tsv`、`data/dev.tsv`、`data/test.tsv` 自动扫描

训练完成后会把标签保存到：

```text
output/label_config.json
```

这样预测和部署时就不需要再把标签硬编码在代码里。

## 上传到 GitHub 前的建议

- 不要提交 `output/` 下的模型权重，GitHub 对大文件有限制
- 不要提交真实业务数据，当前 `.gitignore` 已默认忽略 `data/`
- 如果你后续想保留模型文件，建议使用 Git LFS 或单独放到云存储
- 如果数据涉及真实用户诉求，建议再做一次脱敏

## 后续可继续优化

- 增加 `train.py / eval.py / infer.py` 的更细分入口
- 增加实验配置文件而不是全部靠命令行参数
- 增加最小单元测试和 smoke test
- 增加 Dockerfile 与部署脚本
