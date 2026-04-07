# Molecular-property-prediction
KNN-based molecular property prediction using text embeddings (Contriever). Predicts stiffness and ion migration activation energy for organic molecules.
基于文本嵌入与 KNN 分类器的有机分子属性预测系统，支持预测刚度（Stiffness）与离子迁移激活能（Ion Migration Activation Energy）。

---

## 项目概述

本工具通过以下流程预测材料属性：

1. 使用预训练文本嵌入模型（Contriever 或 OpenScholar_Reranker）将分子描述转换为向量
2. 从 JSON 训练数据中加载已知高/低属性材料的描述，构建训练集
3. 使用 KNN 分类器对目标分子进行属性概率预测
4. 将结果输出为 Excel 表格，并生成概率分布直方图

---

## 目录结构

```
.
├── predict.py                          # 主预测脚本（运行此文件）
├── merged.json                         # 训练数据（从文献中提取的材料属性描述）
├── Molecule_Information_90.xlsx  # 待预测分子数据（含 Description、Abbreviation 等列）
├── requirements.txt                    # Python 依赖包
├── contriever/                         # Contriever 模型源代码
│   └── src/
│       ├── contriever.py
│       └── ...
└── output_<timestamp>/                 # 自动生成的输出文件夹
    ├── Prediction_Results.xlsx         # 所有分子的预测结果表格
    └── Chart_Distribution.png         # 概率分布柱状图
```

---

## 环境要求

- Python 3.8+
- CUDA（推荐，非必需；默认可使用 CPU）

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 模型准备（二选一）

### 选项 A：Contriever（推荐，默认）

代码将自动从 HuggingFace 下载 `akariasai/pes2o_contriever`，无需手动操作。

若网络受限，可设置国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 选项 B：OpenScholar_Reranker

手动下载模型后，修改脚本中的以下变量：

```python
model_name = 'OpenScholar_Reranker'
model = SentenceTransformer('你的模型本地路径')
```

---

## 数据格式

### 训练数据（merged.json）

```json
{
  "all_description": {
    "high stiffness": ["材料1的描述...", "材料2的描述..."],
    "low stiffness":  ["材料3的描述..."],
    "high Ion migration activation energy of the crystal": ["..."],
    "low Ion migration activation energy of the crystal":  ["..."]
  }
}
```

### 待预测数据（Molecule_Information_90.xlsx）

| 列名 | 说明 |
|------|------|
| `Final_ID` | 分子编号（若不存在则自动生成） |
| `Abbreviation` | 分子缩写名称 |
| `Description` | 分子的文本描述（用于嵌入） |

---

## 使用方法

```bash
python predict.py
```

脚本将按以下步骤自动执行：

1. **加载模型**：初始化 Contriever 编码器
2. **加载数据**：读取训练数据与待预测分子
3. **编码与预测**：对所有分子进行嵌入编码，使用 KNN（k=20~95，步长5）预测属性概率并取均值
4. **保存结果**：输出 Excel 表格与分布图至带时间戳的文件夹

---

## 输出说明

### Prediction_Results.xlsx

| Final_ID | Abbreviation | Stiffness_Prob | Ion_Migration_Prob |
|----------|--------------|---------------|-------------------|
| 1        | CHDA         | 0.59          |0.42               |
| 2        | HDA          | 0.52          |0.34               |
| ...      | ...          | ...           | ...               |

- `Stiffness_Prob`：预测为高刚度的概率（0~1）
- `Ion_Migration_Prob`：预测为高离子迁移激活能的概率（0~1）
- 概率的相对高低作为不同分子参与形成的Bi基卤化物单晶的高刚度或高离子迁移活化能的倾向大小。

### Chart_Distribution.png

并排展示两项属性的预测概率分布直方图，用于快速评估整体分布形态。

---

## 常见问题

**Q1：`ModuleNotFoundError: No module named 'contriever'`**

请确保在项目根目录下运行脚本，且 `contriever/` 文件夹存在。

**Q2：模型下载失败**

设置 HuggingFace 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```
或手动下载模型并修改脚本中的路径。

**Q3：找不到训练数据或待预测数据**

确认以下文件存在于项目根目录：
- `merged.json`
- `Molecule_Information_90.xlsx`

**Q4：CUDA out of memory**

在脚本中将设备改为 CPU：
```python
device = 'cpu'
```

---

## 引用

如使用本工具，请引用以下相关工作：

- Contriever: Izacard et al., *Unsupervised Dense Information Retrieval with Contrastive Learning*, 2021
- OpenScholar: Asai et al., *OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs*, 2024

---
## 联系方式
如有问题请联系[tpd2023@163.com]
