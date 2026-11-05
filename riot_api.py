import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')

API_KEY = os.getenv('RIOT_API_KEY', '').strip()
API_HOST = 'https://asia.api.riotgames.com'

DAYS = 14
PAGE_SIZE = 100
QUEUE_IDS = {400, 420, 430, 440, 490}
MIN_INTERVAL = 1.3

ID_FILE = ROOT / 'riot_ids.txt'
DATA_DIR = ROOT / 'data' / 'riot'
MATCH_DIR = DATA_DIR / 'matches'
TIMELINE_DIR = DATA_DIR / 'timelines'
ACCOUNT_INDEX = DATA_DIR / 'account_matches.json'
SKIPPED_INDEX = DATA_DIR / 'excluded_matches.json'

session = requests.Session()
last_request_at = 0.0


def api_get(path, params=None):
    global last_request_at
    url = f'{API_HOST}{path}'

    for attempt in range(8):
        delay = MIN_INTERVAL - (time.monotonic() - last_request_at)
        if delay > 0:
            time.sleep(delay)

        try:
            last_request_at = time.monotonic()
            response = session.get(
                url, params=params, timeout=30
            )
        except requests.RequestException as exc:
            if attempt == 7:
                raise RuntimeError(
                    f'Network error: {exc}'
                ) from exc

            wait = min(2 ** attempt, 30)
            print(f'Network error; retrying in {wait}s')
            time.sleep(wait)
            continue

        if response.status_code == 429:
            try:
                wait = float(
                    response.headers.get('Retry-After', '5')
                )
            except ValueError:
                wait = 5

            wait = max(1, wait)
            print(f'HTTP 429; waiting {wait:g}s')
            time.sleep(wait)
            continue

        if response.status_code in (500, 502, 503, 504):
            wait = min(2 ** attempt, 30)
            print(
                f'HTTP {response.status_code}; '
                f'retrying in {wait}s'
            )
            time.sleep(wait)
            continue

        if not response.ok:
            raise RuntimeError(
                f'HTTP {response.status_code}: '
                f'{response.text[:200]}'
            )

        return response.json()

    raise RuntimeError('Too many retries')


def read_json(path, default=None):
    if not path.is_file():
        return default

    try:
        with path.open(encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')

    with temp_path.open('w', encoding='utf-8') as f:
        json.dump(
            value, f, ensure_ascii=False, indent=2
        )

    temp_path.replace(path)


def get_riot_ids():
    if not ID_FILE.exists():
        ID_FILE.write_text(
            '청 묘#KR2\n', encoding='utf-8'
        )
        print('Created riot_ids.txt. Review it and rerun.')
        return []

    result = []
    seen = set()

    for line in ID_FILE.read_text(
        encoding='utf-8-sig'
    ).splitlines():
        line = line.strip()

        if not line or line.startswith('#'):
            continue

        if '#' not in line:
            print(f'Invalid Riot ID (missing #): {line}')
            continue

        game_name, tag = (
            part.strip()
            for part in line.rsplit('#', 1)
        )

        if not game_name or not tag:
            print(f'Invalid Riot ID: {line}')
            continue

        identifier = f'{game_name}#{tag}'

        if identifier.casefold() not in seen:
            result.append((game_name, tag))
            seen.add(identifier.casefold())

    return result


def list_match_ids(puuid, start_time, end_time):
    all_ids = []
    offset = 0

    while True:
        batch = api_get(
            '/lol/match/v5/matches/by-puuid/'
            f'{quote(puuid, safe="")}/ids',
            {
                'startTime': start_time,
                'endTime': end_time,
                'start': offset,
                'count': PAGE_SIZE,
            },
        )

        all_ids.extend(batch)

        if len(batch) < PAGE_SIZE:
            break

        offset += PAGE_SIZE

    return list(dict.fromkeys(all_ids))


def main():
    if not API_KEY:
        raise SystemExit(
            'RIOT_API_KEY is missing from .env'
        )

    session.headers.update({
        'X-Riot-Token': API_KEY
    })

    riot_ids = get_riot_ids()

    if not riot_ids:
        print('No Riot IDs to process.')
        return

    MATCH_DIR.mkdir(parents=True, exist_ok=True)
    TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

    account_index = read_json(ACCOUNT_INDEX, {}) or {}
    excluded = read_json(SKIPPED_INDEX, {}) or {}

    end_time = int(
        datetime.now(timezone.utc).timestamp()
    )
    start_time = int(
        (
            datetime.now(timezone.utc)
            - timedelta(days=DAYS)
        ).timestamp()
    )

    for number, (game_name, tag) in enumerate(
        riot_ids, 1
    ):
        riot_id = f'{game_name}#{tag}'
        print(
            f'\n[{number}/{len(riot_ids)}] '
            f'Account: {riot_id}'
        )

        try:
            account = api_get(
                '/riot/account/v1/accounts/by-riot-id/'
                f'{quote(game_name, safe="")}/'
                f'{quote(tag, safe="")}'
            )

            match_ids = list_match_ids(
                account['puuid'],
                start_time,
                end_time
            )

        except (RuntimeError, KeyError) as exc:
            print(f'Account failed: {exc}')
            continue

        eligible_ids = []

        for idx, match_id in enumerate(match_ids, 1):
            match_path = MATCH_DIR / f'{match_id}.json'
            timeline_path = (
                TIMELINE_DIR / f'{match_id}.json'
            )

            try:
                if (
                    match_id in excluded
                    and not match_path.exists()
                ):
                    print(
                        f'[{idx}/{len(match_ids)}] '
                        f'excluded (cached): {match_id}'
                    )
                    continue

                match = read_json(match_path)

                if match is None:
                    match = api_get(
                        '/lol/match/v5/matches/'
                        f'{quote(match_id, safe="")}'
                    )

                info = match.get('info', {})

                if (
                    info.get('mapId') != 11
                    or info.get('queueId')
                    not in QUEUE_IDS
                ):
                    excluded[match_id] = {
                        'mapId': info.get('mapId'),
                        'queueId': info.get('queueId')
                    }
                    save_json(SKIPPED_INDEX, excluded)

                    print(
                        f'[{idx}/{len(match_ids)}] '
                        f'excluded: {match_id}'
                    )
                    continue

                if not match_path.exists():
                    save_json(match_path, match)

                eligible_ids.append(match_id)

                if read_json(timeline_path) is None:
                    timeline = api_get(
                        '/lol/match/v5/matches/'
                        f'{quote(match_id, safe="")}'
                        '/timeline'
                    )

                    save_json(timeline_path, timeline)

                    print(
                        f'[{idx}/{len(match_ids)}] '
                        f'downloaded: {match_id}'
                    )
                else:
                    print(
                        f'[{idx}/{len(match_ids)}] '
                        f'already saved: {match_id}'
                    )

            except (
                RuntimeError,
                OSError,
                ValueError,
                TypeError
            ) as exc:
                print(
                    f'[{idx}/{len(match_ids)}] '
                    f'error {match_id}: {exc}'
                )

        account_index[riot_id] = {
            'puuid': account['puuid'],
            'checked_at_utc': (
                datetime.now(timezone.utc).isoformat()
            ),
            'days': DAYS,
            'match_ids': eligible_ids,
        }

        save_json(ACCOUNT_INDEX, account_index)

        print(
            f'Completed: '
            f'{len(eligible_ids)} eligible matches'
        )

    print(
        '\nFinished. JSON saved in '
        'data/riot/matches and data/riot/timelines.'
    )


if __name__ == '__main__':
    main()
