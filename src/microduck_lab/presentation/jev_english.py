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
        "本地 · 翻身脱困": "Local · Rollover recovery",
        "本地 · 恢复后站稳": "Local · Regain balance",
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


def commentary_text(candidate, audit, decisions):
    """Event-grounded English lines for a fresh match, rather than a fixed dub."""
    evidence = candidate["evidence"]

    def name(body):
        return CHARACTERS_EN[body]["name"]

    if evidence["kind"] == "rules":
        return "Four ducks. One survivor. Jev calls the plays!"
    if evidence["kind"] == "decision":
        row = decisions[evidence["index"]]
        actor = name(row["body_id"])
        return {
            "JEV_CAPTURE": f"Jev sends {actor} after the beacon!",
            "JEV_INTERCEPT": f"{actor} moves to cut off the chase!",
            "JEV_EVADE": f"{actor} looks for breathing room!",
            "JEV_RETREAT_CORE": f"{actor} heads back towards the centre!",
        }[row["intent"]]
    event = audit["events"][evidence["index"]]
    kind = event["state"]
    if kind == "ELIMINATED":
        left = 4 - sum(
            e["state"] == "ELIMINATED" and e["time"] <= event["time"]
            for e in audit["events"]
        )
        suffix = {
            3: "Three remain!",
            2: "The final two!",
            1: "Just one duck left!",
            0: "Nobody left standing!",
        }[left]
        return f"{name(event['body_id'])} is out! {suffix}"
    if kind == "TERMINAL":
        return f"Standing firm! {name(event['winner'])} is the sole survivor!"
    if kind == "RECOVERY_ROLL_STARTED":
        return f"{name(event['body_id'])} tries a rollover!"
    if kind == "RECOVERY_COMPLETED":
        return f"{name(event['body_id'])} is back up! Back in the fight!"
    if kind == "RECOVERY_GRACE":
        return "Extra recovery time! Contact stays live!"
    if kind == "FINAL_WARNING":
        return "The first tile is counting down! Move!"
    if kind == "FINALE_WARNING":
        return "Final warning! Head for the centre!"
    if kind == "FINALE_RING":
        return (
            "The outer ring drops!"
            if event["radius"] == 2
            else "Only the centre remains!"
        )
    if kind == "BEACON_CAPTURE":
        tile = {12: "centre", 2: "west", 14: "north", 22: "east", 10: "south"}.get(
            event["tile"], f"tile {event['tile']}"
        )
        return (
            f"{name(event['body_id'])} has claimed the {tile} beacon!"
            if evidence.get("retrospective")
            else f"{name(event['body_id'])} takes the {tile} beacon!"
        )
    if kind == "BODY_CONTACT":
        contact = max(event["contacts"], key=lambda c: c["force_N"])
        pair = f"{name(contact['a'])} and {name(contact['b'])}"
        return [
            f"{pair} collide!",
            f"{pair}, shoulder to shoulder!",
            f"Look at that contact! {pair}!",
        ][evidence["index"] % 3]
    raise ValueError("No English commentary for " + kind)
