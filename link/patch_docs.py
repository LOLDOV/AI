"""Data Dragon documents for the match patch, built with the RAG/patch_rag helpers.

Besides the player's champion, the kits of champions involved in the death
(killer, assists, damage sources) are added so the coach can explain what hit the player.
"""
from __future__ import annotations

from urllib.parse import quote

from pipeline.core import ROOT
from RAG.patch_rag import DD_BASE, _choose_patch, _docs_from_ddragon, _retrieve
from RIOT.client import _cache_json

CACHE = ROOT / 'data' / 'riot' / 'ddragon_cache'


class PatchDocs:
    def __init__(self, game_version: str, locale: str = 'ko_KR', version: str | None = None):
        """version: force a Data Dragon version (e.g. an older patch for the stale-doc test)."""
        versions = _cache_json(CACHE / 'versions.json', f'{DD_BASE}/api/versions.json')
        self.version = version or _choose_patch(versions, game_version, 'match', None)
        self.locale = locale
        self.base = f'{DD_BASE}/cdn/{self.version}/data/{locale}'
        folder = CACHE / self.version / locale
        self.champions = _cache_json(folder / 'champion.json', self.base + '/champion.json').get('data', {})
        self.items = _cache_json(folder / 'item.json', self.base + '/item.json')
        self._kits = {}

    def key(self, name: str | None):
        """English key ('MonkeyKing') from a key or localized name."""
        if not name:
            return None
        name = name.casefold()
        return next((k for k, row in self.champions.items()
                     if k.casefold() == name or str(row.get('name', '')).casefold() == name), None)

    def local_name(self, name: str | None):
        key = self.key(name)
        return self.champions[key].get('name', key) if key else name

    def names(self) -> dict:
        """{key or localized name: key} for every champion in this patch."""
        out = {}
        for k, row in self.champions.items():
            out[k] = k
            if row.get('name'):
                out[row['name']] = k
        return out

    def kit(self, key: str, player: dict | None = None) -> list:
        if key not in self._kits:
            details = _cache_json(CACHE / self.version / self.locale / 'champions' / f'{key}.json',
                                  self.base + f'/champion/{quote(key)}.json')
            self._kits[key] = _docs_from_ddragon(details, self.items, player, self.version, self.locale, set())
        return self._kits[key]

    def for_death(self, champion: str, death: dict | None, player_raw: dict | None = None, limit: int = 6) -> list:
        """Skills that actually damaged the player first, then the best matches from the involved kits.
        player_raw (match participant) adds the player's final items."""
        names = [champion]
        hits = []
        if death:
            names += [death.get('killer'), *death.get('assists', []), *(d['source'] for d in death.get('damage_received', []))]
            hits = [(self.key(d['source']), d['slot']) for d in death.get('damage_received', []) if d['slot'] in 'QWER']
        keys = list(dict.fromkeys(k for k in map(self.key, names) if k))[:4]
        pool = []
        for i, k in enumerate(keys):
            pool += self.kit(k, player_raw if i == 0 else None)
        wanted = {f'champion:{k}:{slot}' for k, slot in hits if k}
        direct = [{**d, 'retrieval_score_not_confidence': 'damage_record'} for d in pool if d['id'] in wanted]
        query = ' '.join(self.local_name(k) for k in keys)
        rest = _retrieve([d for d in pool if d['id'] not in wanted], query, limit=max(0, limit - len(direct)))
        return (direct + rest)[:limit]
