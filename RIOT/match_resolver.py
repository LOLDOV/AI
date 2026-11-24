"""Riot ID -> recent match cards. Never infer match identity from MP4 alone.

Used by the local prototype, not Riot RSO/OAuth authentication.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

from pipeline.core import ROOT, load
from RIOT.client import _get, _load_env, _riot_json, _pick_player, RIOT_REGIONS

import os

ALLOWED_QUEUES = {400, 420, 430, 440, 490}


def get_puuid(riot_id: str, region: str = 'asia') -> str:
    """Look up an account by its public Riot ID, not proof of account ownership."""
    if region not in RIOT_REGIONS:
        raise ValueError('지원하지 않는 Riot 라우팅 지역')
    if riot_id.count('#') != 1:
        raise ValueError('Riot ID는 게임이름#태그 형식이어야 합니다')
    name, tag = (part.strip() for part in riot_id.rsplit('#', 1))
    if not name or not tag or len(name) > 60 or len(tag) > 15:
        raise ValueError('Riot ID 이름 또는 태그 형식 오류')
    index = ROOT / 'data' / 'riot' / 'account_matches.json'
    if index.is_file():
        cached = load(index)
        row = next((v for k, v in cached.items() if k.casefold() == riot_id.casefold()), None)
        if isinstance(row, dict) and row.get('puuid'):
            return row['puuid']
    _load_env()
    token = os.getenv('RIOT_API_KEY', '').strip()
    if not token:
        raise RuntimeError('RIOT_API_KEY 없음: .env를 확인하세요')
    endpoint = f'https://{region}.api.riotgames.com/riot/account/v1/accounts/by-riot-id/{quote(name, safe="")}/{quote(tag, safe="")}'
    return _get(endpoint, headers={'X-Riot-Token': token})['puuid']


def recent_match_ids(puuid: str, region: str = 'asia', *, days: int = 14, count: int = 20) -> list[str]:
    if region not in RIOT_REGIONS:
        raise ValueError('지원하지 않는 지역')
    if not 1 <= days <= 90 or not 1 <= count <= 100:
        raise ValueError('days는 1~90, count는 1~100이어야 합니다')
    # Use existing index as a fallback when the user's development key expired.
    # Network remains primary so an older, stale index isn't mistaken for an up-to-date list.
    _load_env()
    token = os.getenv('RIOT_API_KEY', '').strip()
    if token:
        end = int(datetime.now(timezone.utc).timestamp())
        start = int((datetime.now(timezone.utc) - timedelta(days=days)).timestamp())
        url = (f'https://{region}.api.riotgames.com/lol/match/v5/matches/'
               f'by-puuid/{quote(puuid, safe="")}/ids?startTime={start}&endTime={end}&start=0&count={count}')
        return _get(url, headers={'X-Riot-Token': token})
    index = ROOT / 'data' / 'riot' / 'account_matches.json'
    if index.is_file():
        for row in load(index).values():
            if isinstance(row, dict) and row.get('puuid') == puuid:
                return list(row.get('match_ids', []))[:count]
    raise RuntimeError('RIOT_API_KEY 만료/미설정이고 계정의 로컬 경기 인덱스가 없습니다')


def game_duration_seconds(info: dict) -> float:
    duration = float(info.get('gameDuration') or 0)
    # Riot Match-v5 gameDuration is seconds in modern matches.
    if not 30 < duration < 14400:
        raise ValueError('경기 길이가 비정상적입니다')
    return duration


def match_card(match: dict, puuid: str) -> dict | None:
    info = match.get('info', {})
    if info.get('mapId') != 11 or info.get('queueId') not in ALLOWED_QUEUES:
        return None
    try:
        player = _pick_player(match, puuid, '')
        duration = game_duration_seconds(info)
    except (ValueError, RuntimeError):
        return None
    return {
        'match_id': match.get('metadata', {}).get('matchId'),
        'champion': player.get('championName'),
        'role': player.get('teamPosition') or 'unknown',
        'duration_sec': round(duration),
        'game_version': info.get('gameVersion'),
        'queue_id': info.get('queueId'),
        'start_timestamp': info.get('gameStartTimestamp') or info.get('gameCreation'),
        'kills': player.get('kills'), 'deaths': player.get('deaths'),
        'assists': player.get('assists'), 'win': player.get('win'),
    }


def list_recent_matches(puuid: str, *, region: str = 'asia', days: int = 14, count: int = 20) -> list[dict]:
    match_ids = recent_match_ids(puuid, region, days=days, count=count)
    cards: list[dict] = []
    for match_id in match_ids:
        try:
            match = _riot_json(match_id, 'match', fetch=True, region=region)
            card = match_card(match, puuid)
            if card:
                cards.append(card)
        except (RuntimeError, ValueError, KeyError):
            # One unavailable match should not erase the remaining match history.
            continue
    return cards


def validate_selected_match(match: dict, puuid: str, match_id: str, video_duration_sec: float) -> dict:
    """Strict identity and duration guard, but this cannot prove what MP4 depicts."""
    if match.get('metadata', {}).get('matchId', '').upper() != match_id.upper():
        raise ValueError('Match ID가 실제 JSON과 일치하지 않습니다')
    card = match_card(match, puuid)
    if not card:
        raise ValueError('해당 계정이 참가한 소환사의 협곡 경기가 아닙니다')
    game_sec = card['duration_sec']
    tolerance = max(90, game_sec * 0.12)
    if abs(video_duration_sec - game_sec) > tolerance:
        raise ValueError(
            f'영상 길이 {video_duration_sec/60:.1f}분과 경기 {game_sec/60:.1f}분이 너무 다릅니다. '
            '잘못 연결된 데이터 또는 편집 영상일 수 있어 분석을 중단했습니다.'
        )
    return card
