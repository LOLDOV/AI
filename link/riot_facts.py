"""Player deaths and nearby events from Riot Match-V5 match/timeline JSON. Times are game seconds."""
from __future__ import annotations

SLOT = {0: 'Q', 1: 'W', 2: 'E', 3: 'R', 4: 'D', 5: 'F'}
CONTEXT_TYPES = {'CHAMPION_KILL', 'ELITE_MONSTER_KILL', 'BUILDING_KILL'}


def load(match_id: str, fetch: bool = False, region: str = 'asia'):
    """Cached files under data/riot/, downloaded with RIOT_API_KEY when fetch=True."""
    from RIOT.client import _riot_json
    match_id = match_id.upper()
    match = _riot_json(match_id, 'match', fetch=fetch, region=region)
    timeline = _riot_json(match_id, 'timeline', fetch=fetch, region=region)
    for name, data in (('match', match), ('timeline', timeline)):
        if data.get('metadata', {}).get('matchId', '').upper() != match_id:
            raise RuntimeError(f'{name} JSON matchId does not match {match_id}')
    return match, timeline


def participants(match: dict) -> dict:
    out = {}
    for p in match.get('info', {}).get('participants', []):
        out[p['participantId']] = {
            'participant_id': p['participantId'], 'champion': p.get('championName'),
            'team': 'blue' if p.get('teamId') == 100 else 'red',
            'position': p.get('teamPosition') or p.get('individualPosition') or '',
            'riot_id': f"{p.get('riotIdGameName', '')}#{p.get('riotIdTagline', '')}".strip('#'),
            'puuid': p.get('puuid')}
    return out


def find_player(match: dict, puuid: str | None = None, riot_id: str | None = None,
                champion: str | None = None) -> dict:
    norm = lambda s: (s or '').replace(' ', '').casefold()
    people = participants(match).values()
    for p in people:
        if (puuid and p['puuid'] == puuid) or (riot_id and norm(p['riot_id']) == norm(riot_id)):
            return p
    if champion:
        hits = [p for p in people if norm(p['champion']) == norm(champion)]
        if len(hits) == 1:
            return hits[0]
    raise ValueError(f'player not found in match (puuid={puuid}, riot_id={riot_id}, champion={champion})')


def game_version(match: dict) -> str:
    return str(match.get('info', {}).get('gameVersion') or '')


def _events(timeline: dict):
    for frame in timeline.get('info', {}).get('frames', []):
        yield from frame.get('events', [])


def _champion(pid, people):
    return people[pid]['champion'] if pid in people else None


def damage_received(event: dict, people: dict, top: int = 6) -> list:
    """victimDamageReceived summed by source and skill slot, largest first."""
    totals = {}
    for d in event.get('victimDamageReceived') or []:
        amount = (d.get('physicalDamage') or 0) + (d.get('magicDamage') or 0) + (d.get('trueDamage') or 0)
        pid = d.get('participantId') or 0
        source = people[pid]['champion'] if pid in people else (d.get('name') or d.get('type') or 'other')
        slot = 'AA' if d.get('basic') else SLOT.get(d.get('spellSlot'), '?')
        totals[(source, slot)] = totals.get((source, slot), 0) + amount
    ranked = sorted(totals.items(), key=lambda kv: -kv[1])[:top]
    return [{'source': s, 'slot': slot, 'damage': round(v)} for (s, slot), v in ranked]


def player_state(timeline: dict, pid: int, game_sec: float):
    """Latest per-minute snapshot before game_sec (Riot frames are one per minute)."""
    best = None
    for frame in timeline.get('info', {}).get('frames', []):
        if frame.get('timestamp', 0) / 1000 <= game_sec:
            best = frame
    if not best:
        return None
    pf = (best.get('participantFrames') or {}).get(str(pid)) or {}
    return {'snapshot_game_sec': round(best['timestamp'] / 1000), 'level': pf.get('level'),
            'current_gold': pf.get('currentGold')}


def deaths(match: dict, timeline: dict, pid: int) -> list:
    people = participants(match)
    out = []
    for ev in _events(timeline):
        if ev.get('type') != 'CHAMPION_KILL' or ev.get('victimId') != pid:
            continue
        sec = ev.get('timestamp', 0) / 1000
        out.append({'death_id': f'death_{len(out) + 1:02d}', 'game_sec': round(sec, 1),
                    'killer': _champion(ev.get('killerId'), people),
                    'assists': [people[a]['champion'] for a in ev.get('assistingParticipantIds') or [] if a in people],
                    'damage_received': damage_received(ev, people),
                    'state': player_state(timeline, pid, sec)})
    return out


def events_between(match: dict, timeline: dict, start: float, end: float) -> list:
    """Kills, objectives and structures in a game-time window."""
    people = participants(match)
    out = []
    for ev in _events(timeline):
        sec = ev.get('timestamp', 0) / 1000
        if ev.get('type') not in CONTEXT_TYPES or not start <= sec <= end:
            continue
        row = {'type': ev['type'], 'game_sec': round(sec, 1)}
        if ev['type'] == 'CHAMPION_KILL':
            row.update(killer=_champion(ev.get('killerId'), people), victim=_champion(ev.get('victimId'), people))
        elif ev['type'] == 'ELITE_MONSTER_KILL':
            row.update(monster=ev.get('monsterSubType') or ev.get('monsterType'),
                       team='blue' if ev.get('killerTeamId') == 100 else 'red')
        else:
            row.update(building=ev.get('buildingType'), lane=ev.get('laneType'))
        out.append(row)
    return out
