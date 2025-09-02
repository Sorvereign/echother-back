import re
import os
import httpx
from typing import Optional
# from .embedding_service import embedding_service

def _parse_github_repo(repo_url: str) -> tuple[str, str]:
    m = re.search(r"github.com/([^/]+)/([^/#?]+)", repo_url)
    if not m:
        return ("", "")
    owner, repo = m.group(1), m.group(2).replace('.git', '')
    return owner, repo

async def analyze_repository(repo_url: str, github_token: Optional[str] = None) -> dict:
    owner, repo = _parse_github_repo(repo_url)
    if not owner or not repo:
        return {"provider": "unknown", "files": [], "stack": [], "key_files": {}}
    api = f"https://api.github.com/repos/{owner}/{repo}"
    headers = {}
    token = github_token or os.getenv('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f"Bearer {token}"
    
    try:
        async with httpx.AsyncClient(timeout=20) as client:
            # Get repository tree
            tree_res = await client.get(f"{api}/git/trees/HEAD?recursive=1", headers=headers)
            files = []
            key_files = {}
            
            if tree_res.status_code == 200:
                try:
                    data = tree_res.json()
                    tree_data = data.get('tree', [])
                    
                    if isinstance(tree_data, list):
                        for node in tree_data:
                            if isinstance(node, dict) and node.get('type') == 'blob':
                                path = node.get('path')
                                if path:
                                    files.append(path)
                    else:
                        return {"provider": "github", "owner": owner, "repo": repo, "files": [], "stack": [], "key_files": {}, "error": f"GitHub tree response format unexpected: {type(tree_data)}"}
                except Exception as e:
                    print(f"Error parsing GitHub tree response: {e}")
                    return {"provider": "github", "owner": owner, "repo": repo, "files": [], "stack": [], "key_files": {}, "error": "Failed to parse repository tree"}
            else:
                print(f"GitHub API error: {tree_res.status_code} - {tree_res.text}")
                return {"provider": "github", "owner": owner, "repo": repo, "files": [], "stack": [], "key_files": {}, "error": f"GitHub API error: {tree_res.status_code}"}
            
            # Expanded list of key files to fetch for better context
            key_file_names = [
                'README.md', 'package.json', 'requirements.txt', 'pyproject.toml', 'Pipfile', 
                'Cargo.toml', 'composer.json', 'pom.xml', 'build.gradle', 'Gemfile',
                'next.config.js', 'next.config.ts', 'tailwind.config.js', 'tailwind.config.ts',
                'tsconfig.json', 'jsconfig.json', '.env.example', 'docker-compose.yml',
                'Dockerfile', 'src/app/layout.tsx', 'src/app/page.tsx', 'src/components/ui/button.tsx',
                'backend/main.py', 'backend/app.py', 'src/main.tsx', 'src/index.js',
                'backend/requirements.txt', 'frontend/package.json', 'app/layout.tsx', 'app/page.tsx'
            ]
            
            # Find relevant files from the repository structure
            relevant_files = []
            if isinstance(files, list) and len(files) > 0:
                for file_path in files:
                    if isinstance(file_path, str) and file_path:
                        try:
                            file_name = file_path.split('/')[-1]
                            # PRIORITY 1: Always include key configuration files
                            if file_name in key_file_names or file_path in key_file_names:
                                relevant_files.append(file_path)
                            # PRIORITY 1.5: Always include root configuration files
                            elif file_path in ['package.json', 'tsconfig.json', 'next.config.ts', 'next.config.js', 'tailwind.config.ts', 'tailwind.config.js']:
                                relevant_files.append(file_path)
                            # PRIORITY 2: Add important source files
                            elif any(file_path.startswith(path) for path in ['src/', 'backend/', 'frontend/', 'app/', 'components/']):
                                if any(file_path.endswith(ext) for ext in ['.tsx', '.ts', '.js', '.jsx', '.py', '.java', '.cs', '.rb']):
                                    relevant_files.append(file_path)
                        except Exception:
                            continue

            # Fetch content for key files (limit to avoid rate limits)
            # Prioritize configuration files first
            config_files = [f for f in relevant_files if any(config in f for config in ['package.json', 'tsconfig.json', 'next.config', 'tailwind.config'])]
            other_files = [f for f in relevant_files if f not in config_files]
            files_to_fetch = config_files + other_files[:12]  # Ensure we get config files + up to 12 others
            
            for file_path in files_to_fetch:
                try:
                    res = await client.get(f"{api}/contents/{file_path}", headers=headers)
                    if res.status_code == 200:
                        j = res.json()
                        if isinstance(j, dict) and j.get('encoding') == 'base64' and j.get('content'):
                            import base64
                            content = base64.b64decode(j['content']).decode('utf-8', errors='ignore')
                            key_files[file_path] = content[:1500]  # Limit content size
                        elif isinstance(j, dict) and j.get('download_url'):
                            raw = await client.get(j['download_url'], headers=headers)
                            if raw.status_code == 200:
                                key_files[file_path] = raw.text[:1500]  # Limit content size
                except Exception:
                    pass
            
            # Enhanced tech stack detection
            stack = []
            
            # Ensure files is a list before iterating
            if isinstance(files, list):
                # Frontend frameworks
                if any('next.config' in f for f in files if isinstance(f, str)) or any('src/app/layout.tsx' in f for f in files if isinstance(f, str)):
                    stack.append('nextjs')
                if any('tailwind.config' in f for f in files if isinstance(f, str)):
                    stack.append('tailwindcss')
                if any(f.endswith('.tsx') or f.endswith('.jsx') for f in files if isinstance(f, str)):
                    stack.append('react')
                if any(f.endswith('.vue') for f in files if isinstance(f, str)):
                    stack.append('vue')
                    
                # Backend frameworks
                if any(f.endswith('main.py') or f.endswith('app.py') for f in files if isinstance(f, str)):
                    stack.append('fastapi/python')
                if any('package.json' in f for f in files if isinstance(f, str)):
                    stack.append('nodejs')
                if any(f.endswith('requirements.txt') or f.endswith('pyproject.toml') for f in files if isinstance(f, str)):
                    stack.append('python')
                if any(f.endswith('Cargo.toml') for f in files if isinstance(f, str)):
                    stack.append('rust')
                if any(f.endswith('.csproj') for f in files if isinstance(f, str)):
                    stack.append('dotnet')
                    
                # Additional tools
                if isinstance(key_files, dict) and key_files:
                    try:
                        if any('supabase' in str(content).lower() for content in key_files.values() if content):
                            stack.append('supabase')
                    except Exception:
                        pass
                if isinstance(files, list) and any('docker' in f.lower() for f in files if isinstance(f, str)):
                    stack.append('docker')
            
            return {
                "provider": "github", 
                "owner": owner, 
                "repo": repo, 
                "files": files, 
                "stack": stack, 
                "key_files": key_files,
                "total_files": len(files) if files else 0,
                "relevant_files_found": len(relevant_files)
            }
    
    except Exception as e:
        print(f"Unexpected error analyzing repository: {e}")
        return {"provider": "github", "owner": owner, "repo": repo, "files": [], "stack": [], "key_files": {}, "error": f"Unexpected error: {str(e)}"}
