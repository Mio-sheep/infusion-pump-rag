"""剂量与流速换算。

这是这个知识库里最"工具性"的一个模块：把医嘱的剂量率换算成泵上要设的 mL/h，
或者反过来，把泵上实际设置的速率换算回患者正在接受的剂量。

为什么值得单独做：这类换算是十倍级差错的高发地 —— 单位看错（μg 当 mg）、
体重单位输错（磅当千克）、浓度算错，任何一处都会让结果差一个数量级，
而且算出来的数字看起来完全正常，屏幕上不会报错。

所以这个模块的设计重点是**把中间步骤摆出来**，而不是只给一个最终数字。
打印出来的推导过程本身就是双人核对时要对的那张纸。

内部统一量纲：质量类一律用微克（ug），生物学效价用 U（unit）。
使用者输入的 mg 会在解析阶段就换算成 ug，后续计算不再关心原始单位。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# 质量单位 -> 微克
_MASS_TO_UG = {"ug": 1.0, "mg": 1000.0, "g": 1_000_000.0}


class CalcError(ValueError):
    """输入本身有问题。消息会直接展示给用户，所以要说人话。"""


@dataclass(frozen=True)
class DoseSpec:
    """一个剂量率表达，例如 0.1 ug/kg/min。value 已换算到内部单位。"""

    value: float  # 每次（每 kg·次）多少 ug（或 U）
    substance: str  # "ug" 或 "U"
    per_kg: bool
    per_minute: bool
    raw: str  # 用户原始写法，用于展示

    def describe(self) -> str:
        return self.raw


@dataclass(frozen=True)
class Concentration:
    """药液浓度，内部统一为「每毫升多少微克（或 U）」。"""

    per_ml: float
    substance: str
    raw: str

    def describe(self) -> str:
        return self.raw


@dataclass
class CalcResult:
    """换算结果，steps 是给人看的推导过程。"""

    rate_ml_h: float
    dose_per_hour: float  # 每小时的药量，单位与 substance 一致
    dose_per_hour_unit: str
    substance: str
    steps: list[str]

    def render(self) -> str:
        return "\n".join(self.steps)


# --------------------------------------------------------------------------- #
# 解析
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """统一写法：去掉空格、全角斜杠，归并希腊字母和 mcg。"""
    text = text.replace("μg", "ug").replace("µg", "ug").replace("mcg", "ug")
    text = text.replace("／", "/").replace(" ", "")
    return text.strip()


def _split_number_unit(text: str) -> tuple[float, str]:
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-zA-Z]+)", text.strip(), re.IGNORECASE)
    if not match:
        raise CalcError(f"看不懂 {text!r}，应写成「数字+单位」，例如 4mg、50mL")
    return float(match.group(1)), match.group(2).lower()


def parse_dose(text: str) -> DoseSpec:
    """解析 "0.1ug/kg/min" / "5mg/h" / "12U/kg/h" 这类剂量率。"""
    raw_input = (text or "").strip()
    raw = _normalize(raw_input)
    if not raw:
        raise CalcError("剂量率为空")

    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([a-zA-Z]+)(/kg)?(/min|/h)?", raw, re.IGNORECASE)
    if not match:
        raise CalcError(f"看不懂剂量率写法：{raw_input!r}。示例：0.1ug/kg/min、5mg/h、12U/kg/h")

    number, unit = float(match.group(1)), match.group(2).lower()
    per_kg, per_minute = bool(match.group(3)), match.group(4) == "/min"

    if unit not in _MASS_TO_UG and unit != "u":
        raise CalcError(f"不认识的药量单位 {unit!r}，支持 ug / μg / mg / g / U")
    if number <= 0:
        raise CalcError("剂量率必须大于 0")

    factor = 1.0 if unit == "u" else _MASS_TO_UG[unit]
    return DoseSpec(
        value=number * factor,
        substance="U" if unit == "u" else "ug",
        per_kg=per_kg,
        per_minute=per_minute,
        raw=raw_input,
    )


def parse_concentration(text: str) -> Concentration:
    """解析药液浓度。两种写法都支持：

    "4mg/50mL"   总药量 / 总体积
    "80ug/mL"    直接给浓度
    """
    raw_input = (text or "").strip()
    raw = _normalize(raw_input)
    if not raw:
        raise CalcError("浓度为空")
    if "/" not in raw:
        raise CalcError(f"浓度写法应为「总药量/总体积」或「浓度/mL」，收到：{raw_input!r}")

    left, right = raw.split("/", 1)
    amount_value, amount_unit = _split_number_unit(left)
    if amount_unit not in _MASS_TO_UG and amount_unit != "u":
        raise CalcError(f"不认识的药量单位 {amount_unit!r}，支持 ug / μg / mg / g / U")

    # 右侧是 "mL"（直接给浓度）或 "50mL"（总体积）
    if right.lower() == "ml":
        volume_ml = 1.0
    else:
        volume_value, volume_unit = _split_number_unit(right)
        if volume_unit != "ml":
            raise CalcError(f"体积单位应为 mL，收到 {volume_unit!r}")
        volume_ml = volume_value

    if amount_value <= 0 or volume_ml <= 0:
        raise CalcError("药量和体积都必须大于 0")

    factor = 1.0 if amount_unit == "u" else _MASS_TO_UG[amount_unit]
    return Concentration(
        per_ml=amount_value * factor / volume_ml,
        substance="U" if amount_unit == "u" else "ug",
        raw=raw_input,
    )


def _parse_rate(text: str) -> float:
    """泵速。写 "4.5"、"4.5mL"、"4.5mL/h" 都认，不带单位时默认 mL/h。"""
    raw = _normalize(text or "")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)(ml(?:/h|/hr|/hour)?)?", raw, re.IGNORECASE)
    if not match:
        raise CalcError(f"速率应写成「数值 mL/h」，例如 4.5 或 4.5mL/h，收到 {text!r}")
    value = float(match.group(1))
    if value <= 0:
        raise CalcError("速率必须大于 0")
    return value


def _require_weight(per_kg: bool, weight_kg: float | None, unit_hint: str) -> float:
    """按体重给的剂量率必须有体重；不按体重给的却给了体重，往往是单位输错的信号。"""
    if per_kg:
        if not weight_kg or weight_kg <= 0:
            raise CalcError(f"{unit_hint} 按体重给（含 /kg），必须提供 --weight，且大于 0")
        return weight_kg
    if weight_kg:
        raise CalcError(f"{unit_hint} 不含 /kg，不需要 --weight —— 请确认剂量率的单位没写错")
    return 1.0


# --------------------------------------------------------------------------- #
# 计算
# --------------------------------------------------------------------------- #
def dose_to_rate(
    dose_text: str, weight_kg: float | None, conc_text: str, *, vtbi_ml: float | None = None
) -> CalcResult:
    """剂量率 → 泵速（mL/h）。"""
    dose = parse_dose(dose_text)
    conc = parse_concentration(conc_text)
    weight = _require_weight(dose.per_kg, weight_kg, f"剂量率 {dose.describe()}")

    if dose.substance != conc.substance:
        raise CalcError(
            f"剂量率用的量纲（{dose.substance}）和浓度的量纲（{conc.substance}）不是同一种，"
            "一个是质量一个是生物学单位，两者之间没有换算关系"
        )

    per_hour = dose.value * weight * (60.0 if dose.per_minute else 1.0)
    rate = per_hour / conc.per_ml

    multiplier = f"{dose.value:g}"
    if dose.per_kg:
        multiplier += f" × {weight:g}"
    if dose.per_minute:
        multiplier += " × 60"

    steps = [
        f"剂量率   {dose.describe()}",
        f"体重     {weight:g} kg" if dose.per_kg else "体重     （此剂量率不按体重计）",
        f"浓度     {conc.describe()} = {conc.per_ml:g} {conc.substance}/mL",
        "",
        f"① 每小时所需药量 = {multiplier} = {per_hour:g} {conc.substance}/h",
        f"② 泵速 = {per_hour:g} ÷ {conc.per_ml:g} = {rate:.4g} mL/h",
    ]
    if vtbi_ml and vtbi_ml > 0:
        hours = vtbi_ml / rate
        steps.append(f"③ {vtbi_ml:g} mL 输完约需 {hours:.3g} 小时（约 {hours * 60:.0f} 分钟）")
    steps.extend(["", "请在泵上设置后，按屏幕显示值与上面第②步的数字逐位核对。"])

    return CalcResult(
        rate_ml_h=rate,
        dose_per_hour=per_hour,
        dose_per_hour_unit=f"{conc.substance}/h",
        substance=conc.substance,
        steps=steps,
    )


def rate_to_dose(
    rate_text: str, weight_kg: float | None, conc_text: str, *, as_unit: str
) -> CalcResult:
    """泵速（mL/h）→ 患者实际接受的剂量率。"""
    rate = _parse_rate(rate_text)
    conc = parse_concentration(conc_text)
    target = parse_dose(f"1{as_unit}")
    weight = _require_weight(target.per_kg, weight_kg, f"目标单位 {as_unit}")

    per_hour = rate * conc.per_ml
    per_target = per_hour / weight / (60.0 if target.per_minute else 1.0)

    steps = [
        f"泵速     {rate:g} mL/h",
        f"浓度     {conc.describe()} = {conc.per_ml:g} {conc.substance}/mL",
        f"体重     {weight:g} kg" if target.per_kg else "体重     （目标单位不按体重计）",
        "",
        f"① 每小时药量 = {rate:g} × {conc.per_ml:g} = {per_hour:g} {conc.substance}/h",
        f"② 换算成 {as_unit} = {per_target:.6g}",
        "",
        "把第②步的结果和医嘱对一遍 —— 差一个数量级说明有一处单位看错了。",
    ]

    return CalcResult(
        rate_ml_h=rate,
        dose_per_hour=per_hour,
        dose_per_hour_unit=f"{conc.substance}/h",
        substance=conc.substance,
        steps=steps,
    )
