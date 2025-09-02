from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Optional, List
from utils.supabase_client import supabase
from services.cocoindex_service import CocoIndexService
import httpx
import asyncio
import os

router = APIRouter()

class AnalyzeRepoBody(BaseModel):
    repo_url: str
    user_id: Optional[str] = None
    github_token: Optional[str] = None

class AnalyzeSelectedRepoBody(BaseModel):
    repo_id: int
    user_id: Optional[str] = None
    github_token: Optional[str] = None



@router.post('/analyze-repo')
async def analyze_repo(body: AnalyzeRepoBody):
    try:
        # 1. Indexar con CocoIndex
        database_url = os.getenv('DATABASE_URL')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        
        if not database_url:
            raise HTTPException(status_code=500, detail='DATABASE_URL required')
            
        # 2. Usar sistema completo: CocoIndex + RAG
        cocoindex_service = CocoIndexService(database_url)
        indexing_result = await cocoindex_service.index_repository(
            body.repo_url,
            github_token=body.github_token
        )
        
        if not indexing_result.get('success'):
            raise HTTPException(status_code=500, detail=indexing_result.get('error'))
        
        # 3. Guardar el repositorio analizado en la base de datos
        try:
            # Extraer información del repositorio de la URL
            import re
            repo_match = re.match(r'https://github\.com/([^/]+)/([^/]+)', body.repo_url)
            if repo_match:
                owner, name = repo_match.groups()
                
                # Verificar si ya existe
                existing = supabase.table("repositories").select("*").eq("url", body.repo_url).execute()
                
                if not existing.data:
                    # Insertar nuevo repositorio
                    repo_data = {
                        "url": body.repo_url,
                        "provider": "github",
                        "owner": owner,
                        "name": name,
                        "user_id": body.user_id,
                        "stack": indexing_result.get('analysis', {}),
                        "files": indexing_result.get('indexed_files', 0),
                        "key_files": []
                    }
                    
                    supabase.table("repositories").insert(repo_data).execute()
                    print(f"Repository saved to database: {body.repo_url}")
                else:
                    print(f"Repository already exists in database: {body.repo_url}")
        except Exception as db_error:
            print(f"Error saving repository to database: {db_error}")
            # No fallar el proceso si hay error en la base de datos
        
        # 4. Obtener insights inteligentes si OpenAI está disponible
        context = indexing_result
        if openai_api_key:
            from services.intelligent_ticket_generator import IntelligentTicketGenerator
            generator = IntelligentTicketGenerator(openai_api_key, database_url)
            insights = await generator.get_project_insights(body.repo_url)
            context['insights'] = insights
            context['analysis_method'] = 'cocoindex_plus_rag'
        else:
            context['analysis_method'] = 'cocoindex_only'
            
        return {
            'success': True,
            'analysis': context,
            'repo_url': body.repo_url
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/user-repositories')
async def get_user_repositories(user_id: str = Query(...)):
    """Get repositories analyzed by a specific user"""
    try:
        # Obtener repositorios de la base de datos
        result = supabase.table("repositories").select("*").eq("user_id", user_id).order("created_at", desc=True).execute()
        
        return {
            "success": True,
            "repositories": result.data or []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/repos-with-analysis')
async def repos_with_analysis(github_token: str = Query(...), user_id: str = Query(...), visibility: str | None = Query(default=None)):
    headers = { 'Authorization': f'Bearer {github_token}', 'Accept': 'application/vnd.github+json' }
    params = { 'per_page': 100 }
    if visibility in ('all','public','private'):
        params['visibility'] = visibility
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get('https://api.github.com/user/repos', headers=headers, params=params)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            data = resp.json()
            
            repos_with_analysis = []
            
            async def analyze_and_store_repo(repo_data):
                try:
                    repo_url = repo_data.get('html_url')
                    database_url = os.getenv('DATABASE_URL')
                    if not database_url:
                        raise HTTPException(status_code=500, detail='DATABASE_URL is required')
                    service = CocoIndexService(database_url)
                    result = await service.index_repository(repo_url, github_token=github_token)
                    return {
                        'id': repo_data.get('id'),
                        'name': repo_data.get('name'),
                        'full_name': repo_data.get('full_name'),
                        'private': repo_data.get('private'),
                        'html_url': repo_data.get('html_url'),
                        'description': repo_data.get('description'),
                        'default_branch': repo_data.get('default_branch'),
                        'analysis': result
                    }
                except Exception as e:
                    return {
                        'id': repo_data.get('id'),
                        'name': repo_data.get('name'),
                        'full_name': repo_data.get('full_name'),
                        'private': repo_data.get('private'),
                        'html_url': repo_data.get('html_url'),
                        'description': repo_data.get('description'),
                        'default_branch': repo_data.get('default_branch'),
                        'analysis': None,
                        'error': str(e)
                    }
            
            tasks = [analyze_and_store_repo(repo) for repo in data[:10]]
            repos_with_analysis = await asyncio.gather(*tasks)
            
            return { 'repos': repos_with_analysis }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post('/analyze-selected-repo')
async def analyze_selected_repo(body: AnalyzeSelectedRepoBody):
    headers = { 'Authorization': f'Bearer {body.github_token}', 'Accept': 'application/vnd.github+json' }
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f'https://api.github.com/repositories/{body.repo_id}', headers=headers)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            repo_data = resp.json()
            
            repo_url = repo_data.get('html_url')
            database_url = os.getenv('DATABASE_URL')
            if not database_url:
                raise HTTPException(status_code=500, detail='DATABASE_URL is required')
            service = CocoIndexService(database_url)
            result = await service.index_repository(repo_url, github_token=body.github_token)
            
            if supabase:
                record = {
                    'url': repo_url,
                    'provider': context.get('provider'),
                    'owner': context.get('owner'),
                    'name': context.get('repo'),
                    'stack': context.get('stack'),
                    'files': context.get('files', [])[:5000] if context.get('files') else [],
                    'key_files': context.get('key_files', {}),
                    'user_id': body.user_id
                }
                try:
                    supabase.table('repositories').upsert(record, on_conflict='url').execute()
                except Exception as e:
                    print(f"Error upserting repository: {e}")
            
            return {
                'repo': {
                    'id': repo_data.get('id'),
                    'name': repo_data.get('name'),
                    'full_name': repo_data.get('full_name'),
                    'private': repo_data.get('private'),
                    'html_url': repo_data.get('html_url'),
                    'description': repo_data.get('description'),
                    'default_branch': repo_data.get('default_branch')
                },
                'analysis': result
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/github-repos')
async def github_repos(github_token: str = Query(...), visibility: str | None = Query(default=None)):
    headers = { 'Authorization': f'Bearer {github_token}', 'Accept': 'application/vnd.github+json' }
    params = { 'per_page': 100 }
    if visibility in ('all','public','private'):
        params['visibility'] = visibility
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get('https://api.github.com/user/repos', headers=headers, params=params)
            if resp.status_code != 200:
                raise HTTPException(status_code=resp.status_code, detail=resp.text)
            data = resp.json()
            items = [
                {
                    'id': r.get('id'),
                    'name': r.get('name'),
                    'full_name': r.get('full_name'),
                    'private': r.get('private'),
                    'html_url': r.get('html_url'),
                    'description': r.get('description'),
                    'default_branch': r.get('default_branch')
                } for r in data
            ]
            return { 'repos': items }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


