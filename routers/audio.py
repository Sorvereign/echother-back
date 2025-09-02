from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional
import tempfile
import uuid
import os
import traceback

# Import with error handling
try:
    from utils.supabase_client import supabase
except ImportError as e:
    print(f"Warning: Could not import supabase: {e}")
    supabase = None

try:
    from services.whisper_service import transcribe_audio_to_text
except ImportError as e:
    print(f"Warning: Could not import whisper service: {e}")
    def transcribe_audio_to_text(file_path: str) -> str:
        return "Transcription service not available"

router = APIRouter()

@router.post('/upload-audio')
async def upload_audio(file: UploadFile = File(...), user_id: Optional[str] = Form(None), repo_url: Optional[str] = Form(None)):
    print(f"Received audio upload request: file={file.filename}, user_id={user_id}, repo_url={repo_url}")
    
    if supabase is None:
        print("Error: Supabase not configured")
        raise HTTPException(status_code=500, detail='Supabase not configured')
    
    transcription_id = str(uuid.uuid4())
    suffix = os.path.splitext(file.filename or 'audio.webm')[1] or '.webm'
    object_name = f"{transcription_id}{suffix}"
    tmp_path = None
    
    try:
        # Read and save file
        print(f"Creating temporary file with suffix: {suffix}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            print(f"Read {len(content)} bytes from uploaded file")
            tmp.write(content)
        
        print(f"Temporary file created at: {tmp_path}")
        
        # Upload to Supabase Storage
        print(f"Uploading to Supabase storage as: {object_name}")
        upload_success = False
        
        # First, try to upload directly
        try:
            with open(tmp_path, 'rb') as f:
                result = supabase.storage.from_('audios').upload(
                    object_name, 
                    f, 
                    file_options={
                        "content-type": file.content_type or 'audio/webm'
                    }
                )
                print(f"Upload result: {result}")
                upload_success = True
        except Exception as upload_error:
            print(f"Upload failed: {upload_error}")
            
            # If file already exists, try to update it
            if "already exists" in str(upload_error).lower() or "duplicate" in str(upload_error).lower():
                try:
                    print("File exists, trying to update...")
                    with open(tmp_path, 'rb') as f:
                        result = supabase.storage.from_('audios').update(
                            object_name, 
                            f, 
                            file_options={
                                "content-type": file.content_type or 'audio/webm'
                            }
                        )
                        print(f"Update result: {result}")
                        upload_success = True
                except Exception as update_error:
                    print(f"Update failed: {update_error}")
            
            # If bucket doesn't exist, create it and retry
            if not upload_success:
                try:
                    print("Trying to create audios bucket...")
                    bucket_result = supabase.storage.create_bucket('audios')
                    print(f"Bucket creation result: {bucket_result}")
                    
                    # Retry upload after creating bucket
                    with open(tmp_path, 'rb') as f2:
                        result = supabase.storage.from_('audios').upload(
                            object_name, 
                            f2, 
                            file_options={
                                "content-type": file.content_type or 'audio/webm'
                            }
                        )
                        print(f"Retry upload result: {result}")
                        upload_success = True
                        
                except Exception as bucket_error:
                    print(f"Bucket creation and retry failed: {bucket_error}")
        
        if not upload_success:
            print("Warning: Audio upload to storage failed, but continuing with transcription...")
            # Continue anyway - we can still transcribe without storing in Supabase
        
        # Transcribe audio
        print(f"Starting transcription of file: {tmp_path}")
        try:
            text = transcribe_audio_to_text(tmp_path)
            print(f"Transcription completed. Text length: {len(text)}")
            print(f"Transcription preview: {text[:100]}...")
        except Exception as transcription_error:
            print(f"Transcription failed: {transcription_error}")
            traceback.print_exc()
            text = f"Transcription failed: {str(transcription_error)}"
        
        # Save to database
        data = {
            'id': transcription_id,
            'user_id': user_id,
            'repo_url': repo_url,
            'audio_object': object_name,
            'text': text,
            'status': 'transcribed'
        }
        print(f"Inserting data to database: {data}")
        
        db_result = supabase.table('transcriptions').insert(data).execute()
        print(f"Database insert result: {db_result}")
        
        print(f"Upload completed successfully. Transcription ID: {transcription_id}")
        return {'transcription_id': transcription_id}
        
    except Exception as e:
        print(f"Error in upload_audio: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                print(f"Temporary file removed: {tmp_path}")
            except Exception as cleanup_error:
                print(f"Failed to remove temporary file: {cleanup_error}")

@router.get('/transcriptions')
async def get_all_transcriptions():
    """Get all transcriptions"""
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        transcriptions = supabase.table('transcriptions').select('*').order('created_at', desc=True).execute()
        return transcriptions.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get('/transcriptions/{transcription_id}')
async def get_transcription(transcription_id: str):
    """Get a specific transcription"""
    if supabase is None:
        raise HTTPException(status_code=500, detail='Supabase not configured')
    try:
        transcription = supabase.table('transcriptions').select('*').eq('id', transcription_id).single().execute()
        if not transcription.data:
            raise HTTPException(status_code=404, detail='Transcription not found')
        return transcription.data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
