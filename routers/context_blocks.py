import os
import tempfile
import uuid
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from services.context_blocks_service import ContextBlocksService
from utils.supabase_client import supabase

router = APIRouter(prefix="/context-blocks", tags=["context-blocks"])

class ResolveItemRequest(BaseModel):
    item_id: str
    resolution_context: str

class SearchItemsRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5

@router.post("/process-meeting")
async def process_meeting(file: UploadFile = File(...), user_id: Optional[str] = Form(None), repo_url: Optional[str] = Form(None)):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    tmp_path = None
    try:
        suffix = os.path.splitext(file.filename or "audio.webm")[1] or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        service = ContextBlocksService()
        result = service.process_meeting(tmp_path, user_id=user_id, repo_url=repo_url)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

# Note: Real-time endpoints removed for simplicity

@router.get("/session/{session_id}")
async def get_session(session_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        session = supabase.table("context_sessions").select("*").eq("id", session_id).single().execute()
        blocks = supabase.table("context_blocks").select("*").eq("session_id", session_id).execute()
        
        blocks_with_items = []
        for block in (blocks.data or []):
            items = supabase.table("context_block_items").select("*").eq("context_block_id", block["id"]).execute()
            block["items"] = items.data or []
            blocks_with_items.append(block)
        
        return {
            "session": session.data,
            "context_blocks": blocks_with_items
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sessions")
async def get_all_sessions():
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        sessions = supabase.table("context_sessions").select("*").order("created_at", desc=True).execute()
        return {"sessions": sessions.data or []}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/items/{item_id}/resolve")
async def resolve_item(item_id: str, request: ResolveItemRequest):
    try:
        service = ContextBlocksService()
        result = service.resolve_item_to_prompt(item_id, request.resolution_context)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/blocks/{block_id}/system-prompt")
async def build_system_prompt(block_id: str):
    try:
        service = ContextBlocksService()
        result = service.build_system_prompt(block_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_similar_items(request: SearchItemsRequest):
    try:
        service = ContextBlocksService()
        results = service.search_similar_items(request.query, request.top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Note: Whisper endpoint removed for simplicity

@router.get("/blocks/{block_id}")
async def get_context_block(block_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        block = supabase.table("context_blocks").select("*").eq("id", block_id).single().execute()
        items = supabase.table("context_block_items").select("*").eq("context_block_id", block_id).execute()
        
        if not block.data:
            raise HTTPException(status_code=404, detail="Context block not found")
        
        block_data = block.data
        block_data["items"] = items.data or []
        
        return block_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/prompts/{prompt_id}")
async def get_system_prompt(prompt_id: str):
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        prompt = supabase.table("system_prompts").select("*").eq("id", prompt_id).single().execute()
        
        if not prompt.data:
            raise HTTPException(status_code=404, detail="System prompt not found")
        
        return prompt.data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/test-database")
async def test_database():
    """Test database connectivity and basic operations"""
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Test session creation
        test_session = {
            "id": str(uuid.uuid4()),
            "user_id": "test-user",
            "repo_url": "https://github.com/test/repo",
            "live_transcription": "Test transcription",
            "status": "completed",
            "session_type": "test"
        }
        
        print("Testing session insertion...")
        session_result = supabase.table("context_sessions").insert(test_session).execute()
        print(f"Session insertion result: {session_result}")
        
        # Test context block creation
        test_block = {
            "id": str(uuid.uuid4()),
            "session_id": test_session["id"],
            "title": "Test Block",
            "description": "Test description",
            "transcript_segment": "Test segment",
            "feature_intent": "Test intent",
            "status": "active"
        }
        
        print("Testing context block insertion...")
        block_result = supabase.table("context_blocks").insert(test_block).execute()
        print(f"Context block insertion result: {block_result}")
        
        # Test context item creation
        test_item = {
            "id": str(uuid.uuid4()),
            "context_block_id": test_block["id"],
            "content": "Test item content",
            "item_type": "recommendation",
            "embedding": None,
            "is_resolved": False
        }
        
        print("Testing context item insertion...")
        item_result = supabase.table("context_block_items").insert(test_item).execute()
        print(f"Context item insertion result: {item_result}")
        
        return {
            "success": True,
            "message": "Database operations successful",
            "session_result": session_result.data,
            "block_result": block_result.data,
            "item_result": item_result.data
        }
        
    except Exception as e:
        print(f"Database test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Database test failed: {str(e)}")

@router.delete("/cleanup-test-data")
async def cleanup_test_data():
    """Clean up test data from database"""
    if supabase is None:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Delete test sessions
        test_sessions = supabase.table("context_sessions").delete().eq("session_type", "test").execute()
        
        return {
            "success": True,
            "message": "Test data cleaned up",
            "deleted_sessions": test_sessions.data
        }
        
    except Exception as e:
        print(f"Cleanup failed: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")
