"""查询扩展词表。

语料是中英混排的：正文以中文为主，但大量术语、标准号、报警名称保留英文原文。
这带来一个不对称的问题——

  - 用中文问"阻塞报警"，正文里的 "occlusion" 匹配不上；
  - 用英文问 "occlusion alarm"，正文里的"阻塞"匹配不上。

所以这里按「概念组」组织词表，同组内互相扩展，方向是双向的。

维护建议：只加**确实同义**的说法。把近义但不同概念的词塞进同一组，
会让检索变差而不是变好——比如"阻塞"和"报警"不该合并，
"阻塞报警"这种组合应该交给字符二元组去匹配。
"""
from __future__ import annotations

from .tokenizer import normalize, tokenize

CONCEPTS: dict[str, tuple[str, ...]] = {
    # ---- 设备与部件 -------------------------------------------------- #
    "输液泵": ("infusion pump", "iv pump", "infusion device"),
    "注射泵": ("syringe pump", "syringe infusion pump", "微量泵"),
    "弹性泵": ("elastomeric pump", "elastomeric", "球囊泵", "非电驱动"),
    "蠕动泵": ("peristaltic pump", "peristaltic", "滚轮泵"),
    "肠内营养泵": ("enteral pump", "enteral", "营养泵"),
    "胰岛素泵": ("insulin pump", "insulin infusion pump"),
    "镇痛泵": ("pca pump", "patient-controlled analgesia", "pca", "自控镇痛"),
    "智能泵": ("smart pump", "drug library", "ders", "药物库"),
    "多通道": ("multi-channel", "multichannel", "多通道泵"),
    "输液器": ("administration set", "infusion set", "iv set", "管路", "耗材"),
    "注射器": ("syringe", "plunger", "活塞", "推杆", "法兰"),
    "泵门": ("door", "latch", "锁扣", "门锁", "铰链", "hinge"),
    "传感器": ("sensor", "detector", "探头"),
    "电池": ("battery", "蓄电池"),
    "电源": ("power supply", "mains", "市电", "供电"),
    "显示屏": ("display", "screen", "user interface", "界面", "按键", "keypad"),
    "固件": ("firmware", "software", "软件"),
    "说明书": ("user manual", "instructions", "操作手册"),

    # ---- 报警与故障 -------------------------------------------------- #
    "报警": ("alarm", "alert", "warning", "警报", "提示"),
    "阻塞": ("occlusion", "occluded", "blockage", "obstruction", "堵塞", "堵管"),
    "上游阻塞": ("upstream occlusion", "upstream", "上游"),
    "下游阻塞": ("downstream occlusion", "downstream", "下游"),
    "气泡": ("air in line", "air-in-line", "air bubble", "air", "空气", "进气"),
    "团注": ("bolus", "bolus after occlusion release", "推注"),
    "故障": ("malfunction", "failure", "fault", "troubleshooting", "异常"),
    "排查": ("troubleshooting", "diagnose", "诊断", "检修"),
    "误报": ("nuisance alarm", "false alarm", "假报警", "误报警"),
    "按键连击": ("key bounce", "连击"),
    "开裂": ("crack", "cracked", "cracking", "裂纹"),
    "破损": ("damage", "damaged", "broken", "损坏"),
    "打火": ("spark", "sparks", "fire", "起火", "冒烟", "烧焦"),
    "电击": ("electric shock", "shock", "漏电"),
    "过热": ("overheat", "overheating", "overcharging", "过充"),
    "跌损": ("dropped", "drop", "impact", "跌落", "碰撞"),
    "进水": ("water ingress", "waterproof", "ipx", "防水", "液体渗入"),

    # ---- 临床与用药 -------------------------------------------------- #
    "过量输注": ("over-infusion", "over infusion", "过量"),
    "输注不足": ("under-infusion", "under infusion", "不足"),
    "流速": ("flow rate", "rate", "ml/h", "速度"),
    "待输容量": ("vtbi", "volume to be infused", "预置量"),
    "剂量": ("dose", "dosage", "给药量"),
    "浓度": ("concentration", "配比"),
    "体重": ("weight", "kg", "kilograms", "pounds", "千克", "磅"),
    "高危药品": ("high-alert medications", "high-risk medications", "高危药"),
    "双人核对": ("independent double-check", "double check", "双人查对"),
    "五个正确": ("five rights", "5 rights", "五个对"),
    "交接班": ("change of shift", "handover", "交班"),
    "医嘱": ("physician order", "prescription", "处方"),

    # ---- 风险、法规与维护 -------------------------------------------- #
    "不良事件": ("adverse event", "adverse", "器械不良事件"),
    "召回": ("recall", "市场撤回"),
    "上报": ("reporting", "report", "报告制度", "报告"),
    "风险": ("risk", "hazard", "危害"),
    "校准": ("calibration", "calibrate", "计量", "标定"),
    "维护": ("maintenance", "preventive maintenance", "保养", "预防性维护"),
    "清洁": ("cleaning", "disinfection", "消毒", "擦拭"),
    "质控": ("quality control", "qc", "质量控制"),
    "误差": ("error", "inaccuracy", "偏差", "准确度", "accuracy"),
    "不确定度": ("uncertainty", "测量不确定度"),
    "标准": ("standard", "iec", "iso", "yy", "jjf", "规程", "规范"),
}


def expand_query(tokens: list[str], query: str) -> list[str]:
    """按命中的概念组，把同组的其他说法追加到 token 列表后面。

    返回新列表，不修改入参。
    """
    normalized = normalize(query).lower()
    haystack = " " + " ".join(tokens) + " "

    extra: list[str] = []
    seen = set(tokens)

    for members in CONCEPTS.values():
        if not any(_present(member, normalized, haystack) for member in members):
            continue
        for member in members:
            for token in tokenize(member):
                if token not in seen:
                    seen.add(token)
                    extra.append(token)

    return tokens + extra


def _present(member: str, normalized_query: str, token_haystack: str) -> bool:
    member = member.lower()
    if member.isascii():
        # 英文按 token 边界判断，避免 "air" 命中 "chair" 这类误扩
        return f" {member} " in token_haystack
    # 中文按子串判断：不依赖分词，且术语本身很短
    return member in normalized_query
