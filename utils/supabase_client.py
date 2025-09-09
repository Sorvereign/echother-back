from supabase import create_client, Client
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env from multiple likely locations to be robust to different working dirs
# 1) Project backend root (parent of this file's directory)
# 2) Current working directory
backend_root_env = Path(__file__).resolve().parents[1] / '.env'
cwd_env = Path.cwd() / '.env'

for env_path in [backend_root_env, cwd_env]:
    try:
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=False)
    except Exception:
        # Fail quietly; env may already be set by the environment
        pass

url = os.getenv('SUPABASE_URL')
key = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
supabase: Client = create_client(url, key) if url and key else None
