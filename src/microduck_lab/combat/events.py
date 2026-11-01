"""Post-match event attribution; never feeds hidden labels into an actor."""

ATTRIBUTION_SCHEMA = "duckfu.attack-outcomes.v2"


def attack_outcomes(events, episodes, teams=None):
    outcomes = []
    for event in events:
        if event["state"] != "ATTACK_WINDOW_ENDED":
            continue
        attack, actor = event["attack_id"], event["actor"]
        linked = [e for e in episodes if attack in e["attacks"]]
        if teams is not None:
            linked = [
                e
                for e in linked
                if any(teams[n] != teams[actor] for n in e["actor_pair"])
            ]
        reason = event.get("reason", "COMPLETE")
        completed = reason == "COMPLETE"
        outcomes.append(
            dict(
                state=("CONTACTED_ATTACK" if linked else "MISS")
                if completed
                else "ATTACK_INTERRUPTED",
                time_s=event["time_s"],
                actor=actor,
                attack_id=attack,
                target=event.get("target"),
                contact_episode_ids=[e["event_id"] for e in linked],
                end_reason=reason,
                completed=completed,
                contacted=bool(linked),
            )
        )
    return outcomes


def counter_sequences(events):
    """Observed evade + real enemy miss + later contact-qualified counter."""
    launches = [e for e in events if e["state"] == "ATTACK_LAUNCHED"]
    misses = {e["attack_id"]: e for e in events if e["state"] == "MISS"}
    evades = [
        e
        for e in events
        if e["state"] == "MANEUVER"
        and e.get("skill") in ("evade_step", "cut_counter", "sidestep")
    ]
    strikes = [e for e in events if e["state"] == "VALID_STRIKE"]
    used, result = set(), []
    for hit in strikes:
        for evade in reversed(evades):
            if (
                evade["actor"] != hit["actor"]
                or not 0 <= hit["time_s"] - evade["time_s"] <= 5.0
            ):
                continue
            attack = next(
                (
                    a
                    for a in reversed(launches)
                    if a["actor"] == hit["target"]
                    and a.get("target") == hit["actor"]
                    and a["attack_id"] in misses
                    and abs(a["time_s"] - evade["time_s"]) <= 1.0
                ),
                None,
            )
            if attack and attack["attack_id"] not in used:
                used.add(attack["attack_id"])
                result.append(
                    dict(
                        state="COUNTER_SEQUENCE",
                        actor=hit["actor"],
                        target=hit["target"],
                        time_s=max(
                            hit["time_s"], misses[attack["attack_id"]]["time_s"]
                        ),
                        source_attack_id=attack["attack_id"],
                        counter_attack_id=hit["attack_id"],
                        evade_time_s=evade["time_s"],
                        counter_contact_time_s=hit["time_s"],
                    )
                )
                break
    return result
