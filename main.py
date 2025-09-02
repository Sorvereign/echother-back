from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers.audio import router as audio_router
from routers.repos import router as repos_router
from routers.intelligent_tickets import router as intelligent_tickets_router
from routers.context_blocks import router as context_blocks_router
from routers.auth import router as auth_router
import os


app = FastAPI()

frontend_origin = os.getenv('FRONTEND_ORIGIN', 'http://localhost:3000')
extension_origin = os.getenv('EXTENSION_ORIGIN', 'chrome-extension://*')

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin, 'http://localhost:3000'],
    allow_origin_regex=r"chrome-extension://.*",
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*']
)

app.include_router(audio_router, prefix='/api')
app.include_router(repos_router, prefix='/api')
app.include_router(intelligent_tickets_router, prefix='/api')
app.include_router(context_blocks_router, prefix='/api')
app.include_router(auth_router, prefix='/api')

@app.get('/api/health')
def health():
    return {'status': 'ok'}

@app.get('/api/extension-test')
def extension_test():
    """Simple endpoint to test Chrome extension connectivity"""
    return {
        'status': 'ok',
        'message': 'Chrome extension can reach the backend',
        'timestamp': '2024-01-01T00:00:00Z'
    }
