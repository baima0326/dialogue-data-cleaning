#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对话数据集清洗脚本
研究目的: 探索提问中的人称代词/文本特征对大模型回复质量的影响
清洗目标: 从原始对话数据中筛选出有效的"一问一答"型对话，排除翻译、代码生成等非自然问答
"""

import pandas as pd
import numpy as np
import re
import os
from datetime import datetime
import argparse
import json


# ============================================================
# 清洗规则配置
# ============================================================

class CleaningConfig:
    """清洗规则配置类 - 可根据研究需要调整参数"""

    # 1. 语言筛选
    KEEP_LANGUAGES = ['Chinese', 'English']  # 保留的语言类型

    # 2. 长度阈值
    MIN_QUESTION_LENGTH = 5       # 提问最小长度（字符）
    MAX_QUESTION_LENGTH = 220     # 提问最大长度（221为可能被截断的阈值）
    MIN_REPLY_LENGTH = 10         # 回复最小长度（字符）
    MAX_REPLY_LENGTH = 5000       # 回复最大长度（排除异常长的回复）

    # 3. 翻译请求关键词（中英文及俄文）
    TRANSLATION_PATTERNS = [
        r'^(请?翻译|translate[\s:]|переведи|перевод|翻成|译成|翻译成|译为|译成)',
        r'^(把.*翻译[成为]|translate\s+this|translate\s+the\s+following)',
    ]

    # 4. 代码生成请求检测模式
    # 策略: 同时包含(编程相关词)和(生成指令词)
    CODE_TECH_KEYWORDS = [
        'python', 'javascript', 'js', 'java ', 'html', 'css', 'sql', 'vue', 'react',
        'angular', 'node.js', 'cpp', 'c++', 'ruby', 'go ', 'rust', 'kotlin', 'swift',
        'php', 'perl', 'r语言', 'matlab', 'scala', 'typescript', 'bash', 'shell',
    ]
    CODE_ACTION_KEYWORDS = [
        'write', 'code', 'program', 'script', 'function', 'class', 'debug', 'implement',
        '写', '编写', '代码', '程序', '脚本', '函数', '帮我写', '给我写', '写个',
        '写一写', '写一段', '写一份', '写一套', '写一个', '生成.*代码', '创建.*程序',
        'give me', 'create a', 'build a', 'develop', '用.*写', '用.*实现',
    ]

    # 5. 多轮对话引用特征词
    MULTI_TURN_KEYWORDS = [
        '上文', '之前说', '刚才的', '前面的', '上面提到', '你刚才', '之前提到的',
        ' aforementioned', 'as mentioned above', 'as discussed earlier',
        'you said earlier', 'you mentioned', 'as you said before',
    ]

    # 6. 低质量阈值
    LOW_QUALITY_BERT_THRESHOLD = 0.01     # BERT分数低于此值视为低质量
    LOW_QUALITY_INFO_THRESHOLD = 0        # 信息点数量为0视为低质量

    # 7. 问候/无意义提问
    GREETING_PATTERNS = [
        r'^(你好|您好|hello|hi|hey|在吗|在么|在不在|你好啊|您好啊)$',
    ]

    # 8. 人称代词列表（用于后续分析，不在清洗中过滤）
    FIRST_PERSON_CN = ['我', '我的', '我们', '我们的', '本人', '本人']  # 第一人称
    SECOND_PERSON_CN = ['你', '你的', '你们', '你们的', '您', '您的']   # 第二人称
    FIRST_PERSON_EN = ['i ', 'my ', 'me ', 'we ', 'our ', 'us ']       # 英文第一人称
    SECOND_PERSON_EN = ['you', 'your', 'yours']                        # 英文第二人称


def load_data(file_path):
    """加载数据文件"""
    if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"不支持的文件格式: {file_path}")

    print(f"原始数据加载完成: {len(df)} 条对话")
    print(f"列名: {df.columns.tolist()}")
    return df


def filter_by_language(df, config):
    """规则1: 语言筛选 - 保留指定语言的对话"""
    mask = df['语言'].isin(config.KEEP_LANGUAGES)
    removed = (~mask).sum()
    df = df[mask].copy()
    print(f"[规则1-语言筛选] 过滤 {removed} 条非目标语言对话, 剩余 {len(df)} 条")
    return df


def filter_by_length(df, config):
    """规则2: 长度筛选 - 排除过短/过长的提问和回复"""
    initial_count = len(df)

    # 提问长度筛选
    q_len_mask = (
        (df['提问长度'] >= config.MIN_QUESTION_LENGTH) &
        (df['提问长度'] <= config.MAX_QUESTION_LENGTH)
    )

    # 回复长度筛选
    r_len_mask = (
        (df['回复长度'] >= config.MIN_REPLY_LENGTH) &
        (df['回复长度'] <= config.MAX_REPLY_LENGTH)
    )

    mask = q_len_mask & r_len_mask
    df = df[mask].copy()
    removed = initial_count - len(df)
    print(f"[规则2-长度筛选] 过滤 {removed} 条长度异常对话, 剩余 {len(df)} 条")
    return df


def filter_truncated_questions(df, config):
    """规则3: 排除被截断的提问"""
    truncated_mask = df['提问长度'] == 221
    removed = truncated_mask.sum()
    df = df[~truncated_mask].copy()
    print(f"[规则3-截断检测] 过滤 {removed} 条可能被截断的提问, 剩余 {len(df)} 条")
    return df


def filter_translation_requests(df, config):
    """规则4: 排除纯翻译请求

    翻译请求的特征:
    - 通常以"翻译"、"translate"等开头
    - 提问中人称代词使用模式与正常问答不同
    - 研究目标是自然问答中人称代词的影响，翻译任务不属于此类
    """
    initial_count = len(df)

    translation_mask = pd.Series(False, index=df.index)
    for pattern in config.TRANSLATION_PATTERNS:
        translation_mask |= df['提问'].str.match(pattern, case=False, na=False)

    extra_translation_mask = df['提问'].str.match(
        r'^(.*翻译.*|.*переведи.*|.*translate\s+(to|into|from).*|译.*)',
        case=False, na=False
    )
    translation_mask |= extra_translation_mask

    df = df[~translation_mask].copy()
    removed = initial_count - len(df)
    print(f"[规则4-翻译过滤] 过滤 {removed} 条翻译请求, 剩余 {len(df)} 条")
    return df


def filter_code_requests(df, config):
    """规则5: 排除纯代码/程序生成请求

    代码生成请求的特征:
    - 通常包含编程语言名 + 生成指令词
    - 此类请求中人称代词使用较少且模式固定（多为"帮我写..."）
    - 与自然问答的文本特征差异较大

    检测策略: 必须同时满足:
    1. 包含编程技术关键词（语言名/技术栈）
    2. 包含生成/编写动作词
    """
    initial_count = len(df)

    tech_pattern = r'(?:\b)' + r'(?:\b)|(?:\b)'.join(
        re.escape(kw.strip()) for kw in config.CODE_TECH_KEYWORDS
    ) + r'(?:\b)'

    action_pattern = r'(?:' + r'|'.join(
        re.escape(kw) for kw in config.CODE_ACTION_KEYWORDS
    ) + r')'

    has_tech = df['提问'].str.contains(tech_pattern, case=False, na=False, regex=True)
    has_action = df['提问'].str.contains(action_pattern, case=False, na=False, regex=True)

    pure_code_mask = df['提问'].str.match(
        r'^(def\s|class\s|import\s|#include|using\s+namespace|const\s|var\s|let\s|function\s)',
        case=False, na=False
    )

    code_mask = (has_tech & has_action) | pure_code_mask
    df = df[~code_mask].copy()
    removed = initial_count - len(df)
    print(f"[规则5-代码过滤] 过滤 {removed} 条代码生成请求, 剩余 {len(df)} 条")
    return df


def filter_multi_turn(df, config):
    """规则6: 排除多轮对话引用

    研究需要确保每条对话是独立的"一问一答"，
    多轮对话中的后续提问可能依赖前文语境，不适用于独立分析
    """
    initial_count = len(df)

    multi_turn_mask = pd.Series(False, index=df.index)
    for keyword in config.MULTI_TURN_KEYWORDS:
        multi_turn_mask |= df['提问'].str.contains(keyword, case=False, na=False)

    df = df[~multi_turn_mask].copy()
    removed = initial_count - len(df)
    print(f"[规则6-多轮过滤] 过滤 {removed} 条多轮对话引用, 剩余 {len(df)} 条")
    return df


def filter_low_quality(df, config):
    """规则7: 排除低质量对话

    低质量标准:
    - BERT分数接近0且信息点数量为0（模型未提供有效回复）
    """
    initial_count = len(df)

    low_quality_mask = (
        (df['BERT分数'] <= config.LOW_QUALITY_BERT_THRESHOLD) &
        (df['信息点数量'] <= config.LOW_QUALITY_INFO_THRESHOLD)
    )

    df = df[~low_quality_mask].copy()
    removed = initial_count - len(df)
    print(f"[规则7-质量过滤] 过滤 {removed} 条低质量对话, 剩余 {len(df)} 条")
    return df


def filter_greetings(df, config):
    """规则8: 排除纯问候/无意义提问"""
    initial_count = len(df)

    greeting_mask = pd.Series(False, index=df.index)
    for pattern in config.GREETING_PATTERNS:
        greeting_mask |= df['提问'].str.match(pattern, case=False, na=False)

    df = df[~greeting_mask].copy()
    removed = initial_count - len(df)
    print(f"[规则8-问候过滤] 过滤 {removed} 条纯问候对话, 剩余 {len(df)} 条")
    return df


def extract_pronoun_features(df, config):
    """提取人称代词特征（用于后续分析）

    此步骤在清洗后执行，为研究添加特征列:
    - has_first_person: 是否包含第一人称
    - has_second_person: 是否包含第二人称
    - first_person_count: 第一人称出现次数
    - second_person_count: 第二人称出现次数
    """
    cn_first_pattern = '|'.join(config.FIRST_PERSON_CN)
    cn_second_pattern = '|'.join(config.SECOND_PERSON_CN)

    en_first_pattern = r'\b(' + '|'.join(config.FIRST_PERSON_EN) + r')'
    en_second_pattern = r'\b(' + '|'.join(config.SECOND_PERSON_EN) + r')'

    df['has_first_person_cn'] = df['提问'].str.contains(cn_first_pattern, case=False, na=False, regex=True)
    df['has_first_person_en'] = df['提问'].str.contains(en_first_pattern, case=False, na=False, regex=True)
    df['has_first_person'] = df['has_first_person_cn'] | df['has_first_person_en']

    df['has_second_person_cn'] = df['提问'].str.contains(cn_second_pattern, case=False, na=False, regex=True)
    df['has_second_person_en'] = df['提问'].str.contains(en_second_pattern, case=False, na=False, regex=True)
    df['has_second_person'] = df['has_second_person_cn'] | df['has_second_person_en']

    df['first_person_count'] = (
        df['提问'].str.count(cn_first_pattern, flags=re.IGNORECASE) +
        df['提问'].str.count(en_first_pattern, flags=re.IGNORECASE)
    )
    df['second_person_count'] = (
        df['提问'].str.count(cn_second_pattern, flags=re.IGNORECASE) +
        df['提问'].str.count(en_second_pattern, flags=re.IGNORECASE)
    )

    conditions = [
        (df['has_first_person'] & df['has_second_person']),
        (df['has_first_person'] & ~df['has_second_person']),
        (~df['has_first_person'] & df['has_second_person']),
    ]
    choices = ['both', 'first_only', 'second_only']
    df['pronoun_type'] = np.select(conditions, choices, default='none')

    print(f"[特征提取] 人称代词特征提取完成")
    print(f"  - 包含第一人称: {df['has_first_person'].sum()} 条")
    print(f"  - 包含第二人称: {df['has_second_person'].sum()} 条")
    print(f"  - 同时包含两种: {(df['pronoun_type'] == 'both').sum()} 条")
    print(f"  - 不包含人称代词: {(df['pronoun_type'] == 'none').sum()} 条")

    return df


def generate_cleaning_report(original_df, cleaned_df, output_dir):
    """生成清洗报告"""
    report = {
        '清洗时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '原始数据': {
            '总对话数': len(original_df),
            '语言分布': original_df['语言'].value_counts().to_dict(),
            '平均提问长度': float(original_df['提问长度'].mean()),
            '平均回复长度': float(original_df['回复长度'].mean()),
        },
        '清洗后数据': {
            '总对话数': len(cleaned_df),
            '保留比例': f'{len(cleaned_df)/len(original_df)*100:.1f}%',
            '语言分布': cleaned_df['语言'].value_counts().to_dict(),
            '平均提问长度': float(cleaned_df['提问长度'].mean()),
            '平均回复长度': float(cleaned_df['回复长度'].mean()),
        },
        '人称代词分布': cleaned_df['pronoun_type'].value_counts().to_dict() if 'pronoun_type' in cleaned_df.columns else {},
    }

    report_path = os.path.join(output_dir, 'cleaning_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    text_report = f"""
========================================
对话数据集清洗报告
========================================
清洗时间: {report['清洗时间']}

【原始数据概况】
- 总对话数: {report['原始数据']['总对话数']}
- 语言分布: {report['原始数据']['语言分布']}
- 平均提问长度: {report['原始数据']['平均提问长度']:.1f} 字符
- 平均回复长度: {report['原始数据']['平均回复长度']:.1f} 字符

【清洗后数据概况】
- 总对话数: {report['清洗后数据']['总对话数']}
- 保留比例: {report['清洗后数据']['保留比例']}
- 语言分布: {report['清洗后数据']['语言分布']}
- 平均提问长度: {report['清洗后数据']['平均提问长度']:.1f} 字符
- 平均回复长度: {report['清洗后数据']['平均回复长度']:.1f} 字符

【人称代词分布】
- 同时包含第一/二人称: {report['人称代词分布'].get('both', 0)} 条
- 仅包含第一人称: {report['人称代词分布'].get('first_only', 0)} 条
- 仅包含第二人称: {report['人称代词分布'].get('second_only', 0)} 条
- 不包含人称代词: {report['人称代词分布'].get('none', 0)} 条

【清洗规则说明】
1. 语言筛选: 仅保留中文和英文对话
2. 长度筛选: 排除过短或过长的提问/回复
3. 截断检测: 排除可能被截断的提问(长度=221)
4. 翻译过滤: 排除纯翻译类请求
5. 代码过滤: 排除程序/代码生成请求
6. 多轮过滤: 排除引用前文的多轮对话
7. 质量过滤: 排除低质量(BERT≈0且信息点=0)对话
8. 问候过滤: 排除纯问候类提问

========================================
"""
    text_report_path = os.path.join(output_dir, 'cleaning_report.txt')
    with open(text_report_path, 'w', encoding='utf-8') as f:
        f.write(text_report)

    print(text_report)
    return report


def clean_dialogue_data(input_path, output_dir=None, save_intermediate=False):
    """主清洗函数

    Args:
        input_path: 输入数据文件路径 (.xlsx 或 .csv)
        output_dir: 输出目录，默认为输入文件所在目录
        save_intermediate: 是否保存中间步骤的数据（用于调试）

    Returns:
        cleaned_df: 清洗后的DataFrame
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(input_path)) or '.'
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("对话数据集清洗开始")
    print("="*60)

    df = load_data(input_path)
    original_df = df.copy()

    config = CleaningConfig()

    print("\n" + "-"*40)
    print("执行清洗规则...")
    print("-"*40)

    df = filter_by_language(df, config)
    df = filter_by_length(df, config)
    df = filter_truncated_questions(df, config)
    df = filter_translation_requests(df, config)
    df = filter_code_requests(df, config)
    df = filter_multi_turn(df, config)
    df = filter_low_quality(df, config)
    df = filter_greetings(df, config)

    print("\n" + "-"*40)
    print("提取研究特征...")
    print("-"*40)
    df = extract_pronoun_features(df, config)

    df = df.reset_index(drop=True)

    output_path = os.path.join(output_dir, 'cleaned_dialogue_data.xlsx')
    df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"\n[保存] 清洗后数据已保存至: {output_path}")

    csv_path = os.path.join(output_dir, 'cleaned_dialogue_data.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"[保存] CSV格式已保存至: {csv_path}")

    print("\n" + "="*60)
    print("生成清洗报告...")
    print("="*60)
    generate_cleaning_report(original_df, df, output_dir)

    print("\n" + "="*60)
    print("清洗完成!")
    print("="*60)

    return df


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='对话数据集清洗工具')
    parser.add_argument('input', help='输入数据文件路径 (.xlsx 或 .csv)')
    parser.add_argument('-o', '--output', help='输出目录', default=None)
    parser.add_argument('--intermediate', action='store_true', help='保存中间结果')

    args = parser.parse_args()

    clean_dialogue_data(
        input_path=args.input,
        output_dir=args.output,
        save_intermediate=args.intermediate
    )


if __name__ == '__main__':
    main()
