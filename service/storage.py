"""Minimal local-only account/session/job persistence; not production SSO."""
from __future__ import annotations
import hashlib
import hmac
import os
import secrets
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from pipeline.core import ROOT

DB = ROOT / 'data' / 'service' / 'accounts.sqlite3'
ITERATIONS = 450_000


def _connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB), timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute('PRAGMA foreign_keys = ON')
    conn.execute('PRAGMA busy_timeout = 20000')
    return conn


@contextmanager
def db():
    conn = _connect()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize():
    with db() as conn:
        conn.executescript('''
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          username TEXT UNIQUE NOT NULL,
          password_salt BLOB NOT NULL,
          password_hash BLOB NOT NULL,
          riot_id TEXT,
          puuid TEXT,
          region TEXT DEFAULT 'asia'
        );
        CREATE TABLE IF NOT EXISTS sessions (
          token_hash TEXT PRIMARY KEY,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          expires_at INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS jobs (
          id TEXT PRIMARY KEY,
          user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
          match_id TEXT NOT NULL,
          discovery_id TEXT NOT NULL,
          video_path TEXT NOT NULL,
          status TEXT NOT NULL,
          stage TEXT NOT NULL,
          sync_status TEXT DEFAULT 'pending',
          error TEXT,
          created_at INTEGER NOT NULL,
          updated_at INTEGER NOT NULL
        );
        ''')
        conn.execute("UPDATE jobs SET status='interrupted', stage='restart' WHERE status IN ('queued','running')")
        conn.execute('DELETE FROM sessions WHERE expires_at < ?', (int(time.time()),))


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, ITERATIONS)


def create_user(username: str, password: str) -> int:
    if not 3 <= len(username) <= 40 or not username.replace('_', '').isalnum():
        raise ValueError('아이디는 3~40자의 영문·숫자·밑줄 문자만 사용하세요')
    if not 10 <= len(password) <= 256:
        raise ValueError('비밀번호는 10~256자로 설정하세요')
    salt = os.urandom(16)
    try:
        with db() as conn:
            cur = conn.execute('INSERT INTO users(username,password_salt,password_hash) VALUES (?,?,?)',
                               (username, salt, _hash_password(password,salt)))
            return int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise ValueError('이미 사용 중인 아이디입니다') from exc


def check_password(username: str, password: str) -> int | None:
    with db() as conn:
        row = conn.execute('SELECT * FROM users WHERE username=?', (username,)).fetchone()
    if row is None:
        _hash_password(password, b'\x00'*16)  # constant-ish timing
        return None
    expected = _hash_password(password, row['password_salt'])
    return int(row['id']) if hmac.compare_digest(expected, row['password_hash']) else None


def create_session(user_id: int) -> str:
    value = secrets.token_urlsafe(32)
    with db() as conn:
        conn.execute('INSERT INTO sessions(token_hash,user_id,expires_at) VALUES (?,?,?)',
                     (hashlib.sha256(value.encode()).hexdigest(),user_id,int(time.time())+7*24*3600))
    return value


def logout(value: str | None):
    if value:
        with db() as conn:
            conn.execute('DELETE FROM sessions WHERE token_hash=?', (hashlib.sha256(value.encode()).hexdigest(),))


def get_user(value: str | None) -> dict | None:
    if not value:
        return None
    with db() as conn:
        row = conn.execute('''SELECT u.id, u.username, u.riot_id, u.puuid, u.region
            FROM sessions s JOIN users u ON u.id=s.user_id
            WHERE s.token_hash=? AND s.expires_at>?''',
            (hashlib.sha256(value.encode()).hexdigest(), int(time.time()))).fetchone()
    return dict(row) if row else None


def link_riot(user_id: int, riot_id: str, puuid: str, region: str):
    with db() as conn:
        conn.execute('UPDATE users SET riot_id=?, puuid=?, region=? WHERE id=?',
                     (riot_id, puuid, region, user_id))


def create_job(user_id: int, match_id: str, video_path: Path) -> dict:
    token = secrets.token_hex(8)
    discovery_id = f'AUTO_{token}'
    now = int(time.time())
    with db() as conn:
        conn.execute('''INSERT INTO jobs (id,user_id,match_id,discovery_id,video_path,status,stage,created_at,updated_at)
                        VALUES (?,?,?,?,?,'queued','waiting',?,?)''',
                     (token,user_id,match_id,discovery_id,str(video_path),now,now))
    return {'id':token,'discovery_id':discovery_id}


def update_job(job_id: str, *, status: str | None = None, stage: str | None = None,
               sync_status: str | None = None, error: str | None = None):
    fields = ['updated_at=?']
    values: list = [int(time.time())]
    for col,val in [('status',status),('stage',stage),('sync_status',sync_status),('error',error)]:
        if val is not None:
            fields.append(col+'=?');values.append(val)
    values.append(job_id)
    with db() as conn:
        conn.execute(f"UPDATE jobs SET {','.join(fields)} WHERE id=?",values)


def get_job(user_id: int, job_id: str) -> dict | None:
    with db() as conn:
        row = conn.execute('SELECT * FROM jobs WHERE id=? AND user_id=?', (job_id,user_id)).fetchone()
    return dict(row) if row else None


def list_jobs(user_id: int, limit: int = 20) -> list[dict]:
    with db() as conn:
        rows = conn.execute('SELECT * FROM jobs WHERE user_id=? ORDER BY created_at DESC LIMIT ?', (user_id,limit)).fetchall()
    return [dict(row) for row in rows]
