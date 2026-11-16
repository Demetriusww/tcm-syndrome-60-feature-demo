"""系统配置与四诊特征目录。"""
import csv
from pathlib import Path

import streamlit as st

# 系统信息
SYSTEM_INFO = {
    'name': '症智明辨',
    'version': '2.0.0',
    'description': '基于机器学习与可解释性 AI 的 2 型糖尿病中医证型智能辅助辨证系统',
    'author': '大数据与人工智能创意赛参赛团队',
}

# 证型详细信息 - 6个证型（基于dataset_final_6class.csv）
SYNDROME_INFO = {
    '气阴两虚': {
        'color': '#FF6B6B',
        'icon': '🌸',
        'keywords': ['口渴', '神疲乏力', '腰膝酸软', '视物模糊', '舌红', '脉细'],
        'treatment': '益气养阴',
        'formula': '生脉散合六味地黄丸加减',
        'description': '气阴两虚证是糖尿病最常见证型，表现为气虚与阴虚并存'
    },
    '痰热互结': {
        'color': '#4ECDC4',
        'icon': '🌿',
        'keywords': ['口渴', '口苦', '苔黄腻', '形体肥胖', '心烦易怒', '大便干结'],
        'treatment': '清热化痰，散结和中',
        'formula': '黄连温胆汤合小陷胸汤加减',
        'description': '痰热互结证多见于肥胖型糖尿病患者，痰热交阻中焦'
    },
    '肝肾阴虚': {
        'color': '#45B7D1',
        'icon': '💧',
        'keywords': ['腰膝酸软', '视物模糊', '舌红', '口渴', '脉细', '形体肥胖'],
        'treatment': '滋补肝肾，养阴清热',
        'formula': '杞菊地黄丸合一贯煎加减',
        'description': '肝肾阴虚证多见于糖尿病病程较长者，肝肾阴液亏虚'
    },
    '热盛伤津': {
        'color': '#FFA07A',
        'icon': '🔥',
        'keywords': ['口渴', '多食易饥', '口苦', '小便色黄', '大便干结', '舌红'],
        'treatment': '清热泻火，养阴生津',
        'formula': '消渴方合白虎加人参汤加减',
        'description': '热盛伤津证多见于糖尿病早中期，燥热炽盛耗伤津液'
    },
    '肝胃郁热': {
        'color': '#9370DB',
        'icon': '⚡',
        'keywords': ['心烦易怒', '口苦', '口渴', '多食易饥', '大便干结', '舌红'],
        'treatment': '清肝泻胃，理气开郁',
        'formula': '大柴胡汤合左金丸加减',
        'description': '肝胃郁热证多见于情绪不畅患者，肝郁化火犯胃'
    },
    '阴阳两虚': {
        'color': '#6B8E23',
        'icon': '☯️',
        'keywords': ['畏寒肢冷', '下肢浮肿', '腰膝酸软', '视物模糊', '神疲乏力', '脉细'],
        'treatment': '温阳滋阴，补肾固摄',
        'formula': '金匮肾气丸合水陆二仙丹加减',
        'description': '阴阳两虚证多见于糖尿病后期，阴损及阳，阴阳俱虚'
    }
}

# 论文最终主模型使用的 60 项部署目录。显示顺序、类别和释义均以 CSV 为准。
FEATURE_CATALOG_PATH = (
    Path(__file__).resolve().parent / "models" / "deployment_feature_catalog_60.csv"
)


def _load_feature_catalog(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("selected_in_final_model", "")).strip().lower() == "true"
        ]
    rows.sort(key=lambda row: int(row["display_order"]))
    names = [row["feature"].strip() for row in rows]
    if len(rows) != 60 or len(set(names)) != 60:
        raise ValueError(
            f"部署特征目录必须包含 60 个唯一特征，实际为 {len(rows)} 行 / {len(set(names))} 项"
        )
    return tuple(rows)


FEATURE_CATALOG = _load_feature_catalog(FEATURE_CATALOG_PATH)
FEATURE_NAMES = [row["feature"].strip() for row in FEATURE_CATALOG]
FEATURE_LABELS = {
    row["feature"].strip(): row["display_label"].strip()
    for row in FEATURE_CATALOG
}
FEATURE_DESCRIPTIONS = {
    row["feature"].strip(): row["description"].strip()
    for row in FEATURE_CATALOG
}
FEATURE_CATEGORIES = {
    row["feature"].strip(): row["category"].strip()
    for row in FEATURE_CATALOG
}

# 四诊分组保留论文目录顺序，供 UI 与 AI 上下文共同使用。
FEATURE_GROUPS = {
    category: [
        row["feature"].strip()
        for row in FEATURE_CATALOG
        if row["category"].strip() == category
    ]
    for category in dict.fromkeys(row["category"].strip() for row in FEATURE_CATALOG)
}
SYMPTOM_FEATURES = FEATURE_GROUPS["症状/全身表现"]
TONGUE_FEATURES = FEATURE_GROUPS["舌象"]
COATING_FEATURES = FEATURE_GROUPS["苔象"]
PULSE_FEATURES = FEATURE_GROUPS["脉象"]
TONGUE_PULSE_FEATURES = TONGUE_FEATURES + COATING_FEATURES + PULSE_FEATURES

# 滑块标签使用所属四诊类别，完整释义仍由 description 字段提供。
FEATURE_ATTRIBUTES = {
    name: category
    for name, category in FEATURE_CATEGORIES.items()
}


def get_syndrome_color(syndrome_name):
    """获取证型颜色"""
    return SYNDROME_INFO.get(syndrome_name, {}).get('color', '#888888')


def get_syndrome_icon(syndrome_name):
    """获取证型图标"""
    return SYNDROME_INFO.get(syndrome_name, {}).get('icon', '⚕️')


def format_probability(prob):
    """格式化概率显示"""
    return f"{prob:.1%}"


def create_feature_tooltip(feature_name):
    """创建特征提示文本（含中医属性与《中医诊断学》释义）"""
    attr = FEATURE_ATTRIBUTES.get(feature_name, '')
    desc = FEATURE_DESCRIPTIONS.get(feature_name, '')
    prefix = f"{feature_name}（{attr}）" if attr else feature_name
    return f"{prefix}：{desc}"


def validate_input(feature_dict):
    """验证输入特征"""
    errors = []
    
    for feature, value in feature_dict.items():
        if not isinstance(value, (int, float)):
            errors.append(f"{feature} 必须是数值")
        elif value < 0 or value > 1:
            errors.append(f"{feature} 必须在 0-1 之间")
    
    return len(errors) == 0, errors


def calculate_risk_score(probabilities):
    """
    计算模型不确定性评级（保留旧接口）

    注意：该函数的等级描述的是“不确定性”高低，等级越高代表把握越低，
    与页面上的“诊断把握度评级”语义相反。新代码请使用 calculate_confidence_level。
    """
    max_prob = max(probabilities.values())
    uncertainty = 1 - max_prob
    
    if uncertainty < 0.1:
        return "低", "模型判断把握很高"
    elif uncertainty < 0.3:
        return "中", "模型判断有一定把握"
    else:
        return "高", "模型判断把握较低，建议结合其他诊断方法"


def calculate_confidence_level(probabilities):
    """
    按主证概率给出“诊断把握度”评级（与页面文案语义一致）。

    阈值：>=60% 高；30%~60% 中（提示可能存在兼证）；<30% 低。
    返回 (等级, 说明文字)。
    """
    max_prob = max(probabilities.values())

    if max_prob >= 0.60:
        return "高", "模型对主证判断把握较高"
    if max_prob >= 0.30:
        return "中", "主证概率中等，提示可能存在兼证"
    return "低", "主证概率偏低，模型判断把握有限"
