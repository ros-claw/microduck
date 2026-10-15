"""English presentation identities and HUD localization; never used by actors."""

import re
from PIL import ImageDraw

CHARACTERS_EN = {
    "lavender": {"name": "Dash", "trait": "Fearless rusher"},
    "cream": {"name": "Waddles", "trait": "Cautious survivor"},
    "sky": {"name": "Sly", "trait": "Cunning interceptor"},
    "graphite": {"name": "Ace", "trait": "Relentless challenger"},
}


def english(text):
    """Translate the legacy HUD at its drawing boundary; reject missing strings."""
    for a, b in {
        "张三": "Dash",
        "二呆": "Waddles",
        "老六": "Sly",
        "卷王": "Ace",
    }.items():
        text = text.replace(a, b)
    patterns = [
        (
            r"剩余(\d)鸭 · 第(\d+)轮争夺(.+) · 抢点不判胜，最后一鸭存活才赢",
            r"\1 alive · Beacon \2: \3 · Last survivor wins",
        ),
        (
            r"剩余(\d)鸭 · (\d+)号地板 ([\d.]+)秒后脱落 · 一次仅一块，留路撤离",
            r"\1 alive · Tile \2 drops in \3s · One tile at a time",
        ),
        (r"(.+)夺下(.+)！目标转移：(.+)", r"\1 captures \2! Next beacon: \3"),
        (
            r"开发预览 · 在线 Jev 决策 \| (.+) \| 比赛 ([\d.]+)秒 \| 同一局连续录像",
            r"Live Jev tactics | \1 | Match \2s | One recorded match",
        ),
        (r"([\d.]+)倍速", lambda m: f"{float(m[1]):.2f}x speed"),
        (r" \| 抢岛(\d+)次", r" | \1 caps"),
        (r"(\d+)号岛", r"Tile \1"),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    phrases = {
        "鸭鸭生存战 · Jev 战术对抗": "DUCK SURVIVAL · JEV TACTICS",
        "唯一幸存者 · ": "SOLE SURVIVOR · ",
        " · 其余三鸭已真实跌落淘汰": " · Three rivals fell out",
        "本局获胜：": "WINNER: ",
        "本局平局 · 不强行指定赢家": "DRAW · No forced winner",
        "Dash视角 · 四鸭争夺中心岛": "Dash POV · Battle for the centre",
        "中心岛": "Centre",
        "西岛": "West",
        "北岛": "North",
        "东岛": "East",
        "南岛": "South",
        "已跌落 · 本局出局": "Fell out · Eliminated",
        "终局定格": "FINAL FREEZE",
        "Jev · 尝试截路": "Jev · Intercept attempt",
        "Jev · 抢点": "Jev · Capture",
        "Jev · 避让": "Jev · Evade",
        "Jev · 撤回中心": "Jev · Retreat",
        "Jev · 暂时站稳": "Jev · Hold",
        "本地 · 起身恢复": "Local · Recovering",
        "本地 · 坐站技能恢复": "Local · Get-up skill",
        "本地 · 撤离危险地板": "Local · Evacuating",
        "等待有效 Jev 指令": "Awaiting Jev",
        "路线失效 · 停步": "Route invalid · Brake",
        "开场站稳": "Settling",
        "本局结束 · 站稳": "Match over · Hold",
        "目标：": "Goal: ",
        "淘汰": "OUT",
        "存活": "ALIVE",
    }
    for a, b in phrases.items():
        text = text.replace(a, b)
    if re.search(r"[\u3400-\u9fff]", text):
        raise ValueError("Untranslated English HUD: " + text)
    return text


class EnglishDraw:
    """Localize both text and measurement, with a final viewport fit check."""

    def __init__(self, image, seen):
        self.draw = ImageDraw.Draw(image)
        self.width = image.width
        self.seen = seen

    def __getattr__(self, key):
        return getattr(self.draw, key)

    def textlength(self, text, **kwargs):
        return self.draw.textlength(english(text), **kwargs)

    def text(self, xy, text, **kwargs):
        text = english(text)
        self.seen.add(text)
        font = kwargs.get("font")
        limit = self.width - xy[0] - 12
        if text.startswith("SOLE SURVIVOR"):
            limit = 290
        while font and self.draw.textlength(text, font=font) > limit and font.size > 14:
            font = font.font_variant(size=font.size - 1)
        if font:
            kwargs["font"] = font
        return self.draw.text(xy, text, **kwargs)
