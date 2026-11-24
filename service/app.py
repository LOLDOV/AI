"""LOCAL PROTOTYPE: service login -> Riot ID link -> match selection -> MP4 -> v4/RAG/coach.

This is not Riot Sign On and does not verify ownership of the Riot ID.
Run uvicorn bound to 127.0.0.1 only; never deploy publicly as-is.
"""
from __future__ import annotations
import os
import re
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request, Response, UploadFile, File, Form, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from pipeline.core import ROOT, video_meta, load
from RIOT.client import _riot_json
from RIOT.match_resolver import get_puuid, list_recent_matches, validate_selected_match
from . import storage
from .workflow import submit_analysis

COOKIE_NAME = 'loldov_local_session'
MAX_UPLOAD_BYTES = 2 * 1024**3
STAGE_LABELS = {'waiting':'대기','discover':'전체 VOD 분석','time_sync':'HUD 시간 동기화',
                'riot_rag_build':'경기 기록 및 패치 지식 조회',
                'coaching':'코칭 생성','done':'완료','failed':'오류','restart':'서버 재시작 중단'}

@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.initialize()
    yield

app = FastAPI(title='LOLDOV local prototype', lifespan=lifespan,
              docs_url=None,redoc_url=None,openapi_url=None)
STATIC = Path(__file__).parent / 'static' / 'index.html'


class Credentials(BaseModel):
    username: str = Field(min_length=3,max_length=40)
    password: str = Field(min_length=10,max_length=256)


class RiotLink(BaseModel):
    riot_id: str = Field(min_length=3,max_length=90)
    region: str = 'asia'


def current_user(request: Request) -> dict:
    user = storage.get_user(request.cookies.get(COOKIE_NAME))
    if not user:
        raise HTTPException(status_code=401, detail='먼저 로그인하세요')
    return user


def safe_error(exc: Exception, code: int = 400):
    msg = str(exc)
    msg = re.sub(r'RGAPI-[A-Za-z0-9-]+','[REDACTED_KEY]',msg)
    # Avoid leaking account PUUID in raw network URL or exceptions.
    msg = re.sub(r'(/by-puuid/)[^/\s?]+', r'\1[REDACTED]', msg)
    raise HTTPException(status_code=code, detail=msg[:300]) from exc


def auth_cookie(response, token: str):
    response.set_cookie(COOKIE_NAME, token, httponly=True, samesite='strict',
                        secure=os.getenv('LOLDOV_HTTPS','0') == '1',
                        max_age=7*24*3600,path='/')
    response.headers['Cache-Control'] = 'no-store'


@app.middleware('http')
async def local_same_origin_protection(request: Request, call_next):
    if request.method in {'POST','PUT','PATCH','DELETE'}:
        origin = request.headers.get('origin')
        host = request.headers.get('host','')
        if origin and origin not in {f'http://{host}',f'https://{host}'}:
            return JSONResponse(status_code=403,content={'detail':'Origin 불일치'})
    response = await call_next(request)
    if request.url.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response


@app.get('/')
def index():
    return FileResponse(STATIC, media_type='text/html',headers={'Cache-Control':'no-store'})


@app.post('/api/auth/register')
def register(data: Credentials, response: Response):
    try:
        uid = storage.create_user(data.username,data.password)
    except ValueError as exc:
        safe_error(exc)
    token=storage.create_session(uid)
    auth_cookie(response,token)
    return {'username':data.username,'riot_id':None,'login_type':'local_service_account'}


@app.post('/api/auth/login')
def login(data: Credentials, response: Response):
    uid = storage.check_password(data.username,data.password)
    if uid is None:
        raise HTTPException(status_code=401,detail='아이디 또는 비밀번호가 올바르지 않습니다')
    token = storage.create_session(uid)
    auth_cookie(response,token)
    return {'username':data.username,'login_type':'local_service_account'}


@app.post('/api/auth/logout')
def logout(request: Request, response: Response):
    storage.logout(request.cookies.get(COOKIE_NAME))
    response.delete_cookie(COOKIE_NAME,path='/')
    return {'ok':True}


@app.get('/api/me')
def me(user:dict=Depends(current_user)):
    return {'username':user['username'],'riot_id':user.get('riot_id'),
            'region':user.get('region'),'login_type':'local_service_account',
            'riot_account_ownership_verified':False}


@app.post('/api/riot/link')
def riot_link(data:RiotLink,user:dict=Depends(current_user)):
    try:
        puuid=get_puuid(data.riot_id,data.region)
        storage.link_riot(user['id'],data.riot_id,puuid,data.region)
    except (ValueError,RuntimeError,KeyError) as exc:
        safe_error(exc)
    return {'riot_id':data.riot_id,'region':data.region,
            'ownership_verified':False,
            'notice':'Riot ID 조회 성공은 계정 소유자 인증(RSO)이 아닙니다.'}


@app.get('/api/riot/matches')
def match_history(days:int=30,count:int=20,user:dict=Depends(current_user)):
    if not user.get('puuid'):
        raise HTTPException(status_code=400, detail='먼저 Riot ID를 등록하세요')
    try:
        cards=list_recent_matches(user['puuid'],region=user['region'],days=days,count=count)
    except (ValueError,RuntimeError,KeyError) as exc:
        safe_error(exc,code=503)
    return {'matches':cards,'account':user['riot_id'],'days':days,
            'note':'경기를 선택해 업로드해야 MP4와 Match ID를 연결합니다. 자동 동일 경기 증명은 아닙니다.'}


@app.post('/api/analyses')
async def create_analysis(match_id:str=Form(...), video:UploadFile=File(...),
                          user:dict=Depends(current_user)):
    if not user.get('puuid'):
        raise HTTPException(status_code=400,detail='먼저 Riot ID를 등록하세요')
    if not re.fullmatch(r'[A-Z0-9]+_\d+',match_id,re.I):
        raise HTTPException(status_code=400,detail='경기 식별자 형식 오류')
    if not (video.filename or '').lower().endswith('.mp4'):
        raise HTTPException(status_code=400,detail='MP4 파일만 지원합니다')
    # Obtain and verify match metadata BEFORE writing the large upload.
    try:
        match = _riot_json(match_id.upper(),'match',fetch=True,region=user['region'])
    except (RuntimeError,ValueError) as exc:
        safe_error(exc,code=503)
    temp_dir = ROOT / 'videos' / 'uploads'
    temp_dir.mkdir(parents=True,exist_ok=True)
    temp_path = temp_dir / (secrets.token_hex(16)+'.mp4')
    total=0
    try:
        with temp_path.open('wb') as out:
            while True:
                data=await video.read(2*1024*1024)
                if not data:
                    break
                total+=len(data)
                if total>MAX_UPLOAD_BYTES:
                    raise ValueError('영상은 최대 2GiB까지만 지원합니다')
                out.write(data)
        await video.close()
        with temp_path.open('rb') as f:
            header=f.read(12)
        if len(header)<12 or header[4:8] != b'ftyp':
            raise ValueError('올바른 MP4 컨테이너가 아닙니다')
        meta=video_meta(temp_path)
        if meta['duration_sec'] < 30 or meta['duration_sec'] > 4*3600:
            raise ValueError('영상 길이는 30초~4시간이어야 합니다')
        card=validate_selected_match(match,user['puuid'],match_id,meta['duration_sec'])
        job=storage.create_job(user['id'],match_id,temp_path)
    except (ValueError,RuntimeError,OSError) as exc:
        temp_path.unlink(missing_ok=True)
        safe_error(exc)
    # The selected match supplies champion & role; no CLI manual input required.
    submit_analysis(job['id'],temp_path,match_id.upper(),user['puuid'],
                    card['champion'],card['role'],job['discovery_id'])
    return {'job_id':job['id'],'discovery_id':job['discovery_id'],'status':'queued',
            'match_id':match_id,'champion':card['champion'],
            'notice':'동일 경기 여부는 선택된 Match와 영상 길이로 1차 확인했습니다.'}


def public_job(row:dict) -> dict:
    return {k:row[k] for k in ('id','match_id','discovery_id','status','stage','sync_status','error',
                               'created_at','updated_at')}


@app.get('/api/analyses')
def jobs(user:dict=Depends(current_user)):
    return {'jobs':[public_job(row) for row in storage.list_jobs(user['id'])]}


@app.get('/api/analyses/{job_id}')
def job_status(job_id:str,user:dict=Depends(current_user)):
    row=storage.get_job(user['id'],job_id)
    if not row:
        raise HTTPException(status_code=404,detail='분석을 찾지 못했습니다')
    item=public_job(row)
    item['stage_label']=STAGE_LABELS.get(item['stage'],item['stage'])
    return item


@app.get('/api/analyses/{job_id}/report')
def report(job_id:str,user:dict=Depends(current_user)):
    row=storage.get_job(user['id'],job_id)
    if not row:
        raise HTTPException(status_code=404,detail='분석을 찾지 못했습니다')
    if row['status']!='complete':
        raise HTTPException(status_code=409,detail='코칭 분석이 아직 완료되지 않았습니다')
    base = ROOT / 'results' / 'discovery' / row['discovery_id'] / 'riot_rag'
    ctx_file=base / 'context_bundle.json'
    bundle=load(ctx_file)
    result_dir=base / 'coaching' / 'riot_rag_ab01'
    scenes=[]
    for scene in bundle.get('scenes',[]):
        file=result_dir / scene['scene_id'] / 'assisted_result.json'
        if not file.exists():
            raise HTTPException(status_code=409,detail='결과 파일 일부가 누락됐습니다')
        result=load(file)
        scenes.append({'scene_id':scene['scene_id'], 'mp4_range':[scene['start_mp4_sec'],scene['end_mp4_sec']],
                       'riot_events':len(scene['riot_event_records']),
                       'coaching':result.get('parsed'),
                       'latency_sec':result.get('latency_sec')})
    return {'discovery_id':row['discovery_id'],'match_id':row['match_id'],
            'patch':bundle['data_dragon_version'],
            'sync_status':row['sync_status'],
            'scene_results':scenes,
            'warning':'Gemini의 코칭은 사람이 검증한 정답이 아닙니다. 시전·회피 등의 주장은 영상 확인 필요.'}
