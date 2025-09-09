import os
import time
from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from utils.supabase_client import supabase
import json

router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer(auto_error=False)

# Simple in-memory session store for extension users
extension_sessions = {}

@router.get("/session")
async def get_session(request: Request, credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Get current user session - works with both cookies and bearer tokens"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Try to get token from Authorization header first
        token = None
        if credentials:
            token = credentials.credentials
        else:
            # Try to get from cookies
            auth_cookie = request.cookies.get("sb-access-token")
            if auth_cookie:
                token = auth_cookie
        
        if not token:
            # No authentication found
            return {"user": None}
        
        # Verify the token with Supabase
        user = supabase.auth.get_user(token)
        
        if not user.user:
            return {"user": None}
        
        user_meta = getattr(user.user, 'user_metadata', {}) or {}
        full_name = user_meta.get('full_name') or user_meta.get('name')
        github_username = user_meta.get('user_name') or user_meta.get('preferred_username')
        avatar_url = user_meta.get('avatar_url')

        return {
            "user": {
                "id": user.user.id,
                "email": user.user.email,
                "created_at": user.user.created_at,
                "name": full_name,
                "user_name": github_username,
                "avatar_url": avatar_url
            }
        }
    except Exception as e:
        print(f"Auth error: {e}")
        return {"user": None}

@router.post("/login")
async def login(request: Request):
    """Login endpoint for frontend"""
    try:
        body = await request.json()
        email = body.get("email")
        password = body.get("password")
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        # Authenticate with Supabase
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            return {
                "success": True,
                "user": {
                    "id": response.user.id,
                    "email": response.user.email,
                    "created_at": response.user.created_at
                },
                "access_token": response.session.access_token if response.session else None
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except Exception as e:
        print(f"Login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.post("/extension-login")
async def extension_login(request: Request):
    """Login endpoint specifically for Chrome extension"""
    try:
        body = await request.json()
        email = body.get("email")
        password = body.get("password")
        
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        
        # Authenticate with Supabase
        response = supabase.auth.sign_in_with_password({
            "email": email,
            "password": password
        })
        
        if response.user:
            # Store session for extension
            session_id = f"ext_{response.user.id}"
            extension_sessions[session_id] = {
                "user_id": response.user.id,
                "email": response.user.email,
                "created_at": response.user.created_at
            }
            
            return {
                "success": True,
                "session_id": session_id,
                "user": {
                    "id": response.user.id,
                    "email": response.user.email,
                    "created_at": response.user.created_at
                }
            }
        else:
            raise HTTPException(status_code=401, detail="Invalid credentials")
            
    except Exception as e:
        print(f"Extension login error: {e}")
        raise HTTPException(status_code=500, detail="Login failed")

@router.get("/extension-session/{session_id}")
async def get_extension_session(session_id: str):
    """Get session for Chrome extension"""
    if session_id in extension_sessions:
        return {
            "user": extension_sessions[session_id]
        }
    else:
        return {"user": None}

@router.post("/extension-logout")
async def extension_logout(request: Request):
    """Logout endpoint for Chrome extension"""
    try:
        body = await request.json()
        session_id = body.get("session_id")
        
        if session_id and session_id in extension_sessions:
            del extension_sessions[session_id]
        
        return {"success": True, "message": "Logged out successfully"}
    except Exception as e:
        print(f"Extension logout error: {e}")
        raise HTTPException(status_code=500, detail="Logout failed")

@router.post("/logout")
async def logout():
    """Logout user"""
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    
    try:
        # Clear the session
        supabase.auth.sign_out()
        return {"message": "Logged out successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Logout failed")

@router.post("/extension-token-bridge")
async def extension_token_bridge(request: Request):
    """Endpoint to bridge token from frontend to extension"""
    try:
        body = await request.json()
        access_token = body.get("access_token")
        
        if not access_token:
            raise HTTPException(status_code=400, detail="Access token required")
        
        # Verify the token with Supabase
        user = supabase.auth.get_user(access_token)
        
        if not user.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Generate a simple session ID for the extension
        session_id = f"ext_{user.user.id}_{int(time.time())}"
        extension_sessions[session_id] = {
            "user_id": user.user.id,
            "email": user.user.email,
            "created_at": user.user.created_at,
            "access_token": access_token
        }
        
        return {
            "success": True,
            "session_id": session_id,
            "user": {
                "id": user.user.id,
                "email": user.user.email,
                "created_at": user.user.created_at
            }
        }
        
    except Exception as e:
        print(f"Token bridge error: {e}")
        raise HTTPException(status_code=500, detail="Token bridge failed")

@router.get("/extension-auth")
async def get_extension_auth():
    """Simple endpoint to check if extension can reach the backend"""
    return {"status": "extension_auth_ready"}
