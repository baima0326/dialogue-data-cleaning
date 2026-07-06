# Dialogue Data Cleaning Tool

对话数据集清洗工具，用于从用户-AI对话数据中筛选有效的"一问一答"型对话，并提取人称代词特征。

## 研究背景

本工具服务于研究：**探索提问中的人称代词/文本特征对大模型回复质量的影响**。

由于原始对话数据包含大量翻译请求、代码生成请求、多轮对话等非自然问答内容，需要通过清洗筛选出符合研究需求的有效对话。

## 清洗规则

| 规则 | 说明 | 目的 |
|------|------|------|
| 语言筛选 | 仅保留中文和英文对话 | 聚焦研究目标语言 |
| 长度筛选 | 排除过短或过长的提问/回复 | 确保文本有足够分析价值 |
| 截断检测 | 排除长度恰好为最大值的提问 | 避免使用被截断的数据 |
| 翻译过滤 | 排除"翻译..."、"translate..."等请求 | 翻译任务的人称代词模式不同 |
| 代码过滤 | 排除程序/代码生成请求 | 代码请求缺乏自然语言人称代词特征 |
| 多轮过滤 | 排除引用前文的多轮对话 | 确保每条对话独立可分析 |
| 质量过滤 | 排除BERT分数≈0且信息点=0的对话 | 过滤低质量回复 |
| 问候过滤 | 排除纯问候类提问 | 无研究价值 |

## 新增特征列

清洗后的数据会新增以下人称代词特征列，便于后续研究分析：

- `has_first_person`: 是否包含第一人称（我/我们/I/we...）
- `has_second_person`: 是否包含第二人称（你/您/you...）
- `first_person_count`: 第一人称出现次数
- `second_person_count`: 第二人称出现次数
- `pronoun_type`: 人称代词类型（both / first_only / second_only / none）

## 使用方法

### 命令行运行

```bash
python dialogue_data_cleaning.py your_data.xlsx -o output_directory
```

### Python导入使用

```python
from dialogue_data_cleaning import clean_dialogue_data, CleaningConfig

# 使用默认配置
cleaned_df = clean_dialogue_data('your_data.xlsx', output_dir='./output')

# 自定义配置
config = CleaningConfig()
config.KEEP_LANGUAGES = ['Chinese']  # 仅保留中文
config.MAX_REPLY_LENGTH = 3000       # 调整回复长度上限
```

## 输出文件

- `cleaned_dialogue_data.xlsx` - 清洗后的数据（Excel格式）
- `cleaned_dialogue_data.csv` - 清洗后的数据（CSV格式）
- `cleaning_report.json` - 清洗报告（JSON格式）
- `cleaning_report.txt` - 清洗报告（文本格式）

## 清洗效果示例

| 指标 | 原始数据 | 清洗后 | 保留比例 |
|------|---------|--------|---------|
| 总对话数 | 39,640 | 31,903 | 80.5% |
| 中文对话 | 20,357 | 20,240 | 99.4% |
| 英文对话 | 12,162 | 11,663 | 95.9% |

## Requirements

- Python 3.8+
- pandas
- numpy
- openpyxl

## License

MIT
