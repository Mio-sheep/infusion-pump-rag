"""查询扩展用的中英对照词表。

语料是中英混排的（中文正文里嵌了大量英文术语与标准号），
把中文查询补上对应英文词，能明显提升召回。
"""

from __future__ import annotations

# 中文关键词 -> 需要追加的英文/同义 token
SYNONYMS: dict[str, list[str]] = {
    # 设备与部件
    "输液泵": ["infusion pump", "iv pump"],
    "注射泵": ["syringe pump", "syringe infusion pump"],
    "微量泵": ["syringe pump"],
    "弹性泵": ["elastomeric", "elastomeric pump"],
    "蠕动泵": ["peristaltic", "peristaltic pump"],
    "肠内营养泵": ["enteral", "enteral infusion pump"],
    "胰岛素泵": ["insulin", "insulin infusion pump"],
    "镇痛泵": ["pca", "patient-controlled analgesia", "analgesia"],
    "智能泵": ["smart pump", "drug library", "ders"],
    "多通道": ["multi-channel", "multi channel"],
    "管路": ["tubing", "tube", "line"],
    "输液器": ["infusion set", "administration set", "tubing"],
    "注射器": ["syringe", "plunger", "flange"],
    "泵门": ["door", "latch"],
    "锁扣": ["latch"],
    "传感器": ["sensor", "detector"],
    "电池": ["battery", "battery failure"],
    "电源": ["power", "mains", "electrical"],
    "屏幕": ["screen", "display", "user interface"],
    "按键": ["key", "keypad", "key bounce"],
    "说明书": ["manual", "instructions", "user manual"],
    "固件": ["firmware", "software"],
    # 报警与故障
    "报警": ["alarm", "alert", "warning"],
    "阻塞": ["occlusion", "occluded", "blockage"],
    "气泡": ["air in line", "air-in-line", "air bubble"],
    "空气": ["air"],
    "团注": ["bolus", "bolus after occlusion release"],
    "故障": ["problem", "failure", "fault", "malfunction", "troubleshooting"],
    "排查": ["troubleshooting", "diagnose"],
    "误报": ["nuisance alarm", "false alarm"],
    "连击": ["key bounce"],
    "开裂": ["crack", "cracked", "cracking"],
    "破损": ["damage", "damaged", "broken"],
    "打火": ["spark", "sparks", "fire"],
    "电击": ["shock", "electric shock"],
    "过热": ["overheat", "overheating", "overcharging"],
    "跌落": ["dropped", "drop", "impact"],
    "漏水": ["water", "waterproof", "ingress"],
    # 临床与用药
    "过量": ["over-infusion", "over infusion"],
    "不足": ["under-infusion", "under infusion"],
    "剂量": ["dose", "dosage"],
    "流速": ["flow rate", "rate", "ml/h"],
    "浓度": ["concentration"],
    "体重": ["weight", "kg", "kilograms", "pounds"],
    "单位": ["unit", "units", "measurement"],
    "高危药品": ["high-risk medications", "high alert"],
    "双人核对": ["independent double-check", "double check"],
    "五个正确": ["five rights", "5 rights"],
    "交接班": ["change of shift"],
    "培训": ["education", "training"],
    "患者": ["patient"],
    "医嘱": ["physician order", "prescription"],
    # 风险、法规与维护
    "不良事件": ["adverse event"],
    "召回": ["recall"],
    "上报": ["report", "reporting"],
    "监管": ["regulatory", "fda"],
    "风险": ["risk", "risk reduction"],
    "校准": ["calibration", "calibrate"],
    "检定": ["verification", "calibration"],
    "维护": ["maintenance", "preventive maintenance", "pm"],
    "保养": ["maintenance"],
    "清洁": ["cleaning", "disinfect", "disinfection"],
    "质控": ["quality control"],
    "检测": ["testing", "test", "measurement"],
    "误差": ["error", "accuracy", "inaccuracy"],
    "准确度": ["accuracy"],
    "不确定度": ["uncertainty"],
    "标准": ["standard", "iec", "iso", "yy"],
    "认可": ["approval", "cleared", "pma", "510k"],
}


def expand_tokens(tokens: list[str], query: str) -> list[str]:
    """按查询里出现的中文关键词，追加英文同义 token。"""
    extra: list[str] = []
    seen = set(tokens)
    for key, values in SYNONYMS.items():
        if key in query:
            for value in values:
                for piece in value.split():
                    if piece not in seen:
                        seen.add(piece)
                        extra.append(piece)
    return tokens + extra
