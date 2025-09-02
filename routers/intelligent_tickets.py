#!/usr/bin/env python3
"""
Intelligent Tickets Router - Enhanced endpoint with CocoIndex + RAG + LLM system
Maintains compatibility with original endpoint names
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi import Body
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
from services.intelligent_ticket_generator import IntelligentTicketGenerator
from utils.supabase_client import supabase

router = APIRouter(tags=["tickets"])

@router.get('/ticket/{transcription_id}')
async def get_ticket(transcription_id: str, github_token: str | None = Query(default=None)):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        tr = supabase.table('transcriptions').select('*').eq('id', transcription_id).single().execute()
        if not tr.data:
            raise HTTPException(status_code=404, detail='Transcription not found')
        transcription = tr.data.get('text') or ''
        repo_url = tr.data.get('repo_url') or ''
        if not repo_url:
            raise HTTPException(status_code=400, detail='repo_url is required in transcription to generate intelligent ticket')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        database_url = os.getenv('DATABASE_URL')
        if not openai_api_key or not database_url:
            raise HTTPException(status_code=500, detail='OPENAI_API_KEY and DATABASE_URL are required')
        generator = IntelligentTicketGenerator(openai_api_key, database_url)
        result = await generator.generate_intelligent_ticket(
            user_request=transcription,
            repo_url=repo_url,
            github_token=github_token
        )
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error') or 'Ticket generation failed')
        ticket_data = {
            'transcription_id': transcription_id,
            'title': result['ticket']['title'],
            'description': result['ticket']['description'],
            'acceptance_criteria': result['ticket']['acceptance_criteria'],
            'files_to_modify': result['ticket']['files_to_modify'],
            'context': result['context'],
            'raw_markdown': result['ticket']['raw_markdown'],
            'generation_method': 'intelligent_rag'
        }
        try:
            existing = supabase.table('tickets').select('*').eq('transcription_id', transcription_id).execute()
            if existing.data:
                updated = supabase.table('tickets').update(ticket_data).eq('transcription_id', transcription_id).execute()
                return {'ticket': updated.data[0] if updated.data else ticket_data}
            created = supabase.table('tickets').insert(ticket_data).execute()
            return {'ticket': created.data[0] if created.data else ticket_data}
        except Exception:
            return {'ticket': ticket_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating ticket: {str(e)}')

@router.get('/ticket-from-session/{session_id}')
async def get_ticket_from_session(session_id: str, github_token: str | None = Query(default=None)):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        session = supabase.table('context_sessions').select('*').eq('id', session_id).single().execute()
        if not session.data:
            raise HTTPException(status_code=404, detail='Session not found')
        transcription = session.data.get('live_transcription') or ''
        repo_url = session.data.get('repo_url') or ''
        if not repo_url:
            raise HTTPException(status_code=400, detail='repo_url is required in session to generate intelligent ticket')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        database_url = os.getenv('DATABASE_URL')
        if not openai_api_key or not database_url:
            raise HTTPException(status_code=500, detail='OPENAI_API_KEY and DATABASE_URL are required')
        generator = IntelligentTicketGenerator(openai_api_key, database_url)
        result = await generator.generate_intelligent_ticket(
            user_request=transcription,
            repo_url=repo_url,
            github_token=github_token
        )
        if not result.get('success'):
            raise HTTPException(status_code=500, detail=result.get('error') or 'Ticket generation failed')
        ticket_data = {
            'session_id': session_id,
            'title': result['ticket']['title'],
            'description': result['ticket']['description'],
            'acceptance_criteria': result['ticket']['acceptance_criteria'],
            'files_to_modify': result['ticket']['files_to_modify'],
            'context': result['context'],
            'raw_markdown': result['ticket']['raw_markdown'],
            'generation_method': 'intelligent_rag'
        }
        saved = None
        try:
            filename = f"ticket-session-{session_id}.md"
            saved = supabase.table('project_metadata').insert({
                'filename': filename,
                'content': ticket_data['raw_markdown'] or ticket_data['description'] or '',
                'file_type': 'ticket',
                'repo_url': repo_url
            }).execute()
        except Exception:
            saved = None
        return {'ticket': ticket_data, 'saved': saved.data[0] if saved and saved.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error generating ticket: {str(e)}')

@router.get('/ticket-from-session/{session_id}/latest')
async def get_latest_ticket_from_session(session_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        filename = f"ticket-session-{session_id}.md"
        records = supabase.table('project_metadata').select('*').eq('filename', filename).order('created_at', desc=True).limit(1).execute()
        if not records.data:
            return {'ticket': None}
        return {'ticket': {'raw_markdown': records.data[0].get('content', '')}}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error fetching ticket: {str(e)}')

@router.post('/ticket-from-session/{session_id}/save')
async def save_ticket_for_session(session_id: str, payload: Dict[str, Any] = Body(...)):
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        session = supabase.table('context_sessions').select('*').eq('id', session_id).single().execute()
        if not session.data:
            raise HTTPException(status_code=404, detail='Session not found')
        repo_url = session.data.get('repo_url') or ''
        raw_markdown = (payload or {}).get('raw_markdown') or ''
        filename = f"ticket-session-{session_id}.md"
        saved = supabase.table('project_metadata').insert({
            'filename': filename,
            'content': raw_markdown,
            'file_type': 'ticket',
            'repo_url': repo_url
        }).execute()
        return {'saved': bool(saved.data), 'record': saved.data[0] if saved.data else None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error saving ticket: {str(e)}')
