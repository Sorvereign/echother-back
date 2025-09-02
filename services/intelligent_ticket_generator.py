#!/usr/bin/env python3
"""
Intelligent Ticket Generator using CocoIndex + RAG + LLM
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from openai import OpenAI
from .cocoindex_service import CocoIndexService
from .rag_service import RAGService, ProjectContext
import json

class IntelligentTicketGenerator:
    def __init__(self, openai_api_key: str, database_url: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.cocoindex_service = CocoIndexService(database_url)
        self.rag_service = RAGService()
        
    async def generate_intelligent_ticket(self, user_request: str, repo_url: str, github_token: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate an intelligent ticket using the complete RAG pipeline
        """
        try:
            print("🚀 Starting intelligent ticket generation...")
            
            # Step 1: Index repository with CocoIndex
            print("📦 Indexing repository with CocoIndex...")
            indexing_result = await self.cocoindex_service.index_repository(repo_url, github_token)
            
            if not indexing_result["success"]:
                return {
                    "success": False,
                    "error": f"Failed to index repository: {indexing_result['error']}",
                    "ticket": None
                }
            
            print(f"✅ Repository indexed successfully: {indexing_result['indexed_files']} files, {indexing_result['embeddings_generated']} embeddings")
            
            # Step 2: Analyze user request
            print("🧠 Analyzing user request...")
            request_analysis = await self.rag_service.analyze_user_request(user_request)
            print(f"✅ Request analyzed: {request_analysis['intent']} ({request_analysis['complexity']} complexity)")
            
            # Step 3: Search for relevant code using RAG
            print("🔍 Searching for relevant code...")
            # For now, we'll use placeholder embeddings until we have the database set up
            placeholder_embeddings = self._get_placeholder_embeddings()
            relevant_code = await self.rag_service.search_relevant_code(user_request, placeholder_embeddings)
            print(f"✅ Found {len(relevant_code)} relevant code chunks")
            
            # Step 4: Build project context
            print("🏗️ Building project context...")
            project_metadata = self._get_placeholder_metadata()
            project_context = await self.rag_service.build_project_context(relevant_code, project_metadata)
            print(f"✅ Project context built: {len(project_context.architectural_patterns)} patterns detected")
            
            # Step 5: Generate contextual prompt
            print("📝 Generating contextual prompt...")
            contextual_prompt = await self.rag_service.generate_contextual_prompt(
                user_request, project_context, request_analysis
            )
            print("✅ Contextual prompt generated")
            
            # Step 6: Generate ticket with LLM
            print("🤖 Generating ticket with LLM...")
            ticket = await self._generate_ticket_with_llm(contextual_prompt, user_request)
            print("✅ Ticket generated successfully")
            
            return {
                "success": True,
                "ticket": ticket,
                "context": {
                    "indexing_result": indexing_result,
                    "request_analysis": request_analysis,
                    "relevant_code_count": len(relevant_code),
                    "project_context": {
                        "languages": list(project_context.technology_stack["languages"].keys()),
                        "frameworks": list(project_context.technology_stack["frameworks"].keys()),
                        "patterns": project_context.architectural_patterns,
                        "best_practices": project_context.best_practices
                    }
                }
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "ticket": None
            }
    
    def _get_placeholder_embeddings(self) -> List[Dict[str, Any]]:
        """Get placeholder embeddings for testing"""
        return [
            {
                "filename": "app/components/LoginForm.tsx",
                "code": "export default function LoginForm() {\n  const [email, setEmail] = useState('');\n  const [password, setPassword] = useState('');\n  \n  const handleSubmit = async (e) => {\n    e.preventDefault();\n    // Login logic here\n  };\n  \n  return (\n    <form onSubmit={handleSubmit}>\n      <input type='email' value={email} onChange={(e) => setEmail(e.target.value)} />\n      <input type='password' value={password} onChange={(e) => setPassword(e.target.value)} />\n      <button type='submit'>Login</button>\n    </form>\n  );\n}",
                "embedding": [0.1] * 384,
                "language": "typescript",
                "metadata": {"has_functions": True, "has_imports": True}
            },
            {
                "filename": "app/api/auth/route.ts",
                "code": "import { NextRequest, NextResponse } from 'next/server';\n\nexport async function POST(request: NextRequest) {\n  try {\n    const { email, password } = await request.json();\n    \n    // Authentication logic here\n    \n    return NextResponse.json({ success: true });\n  } catch (error) {\n    return NextResponse.json({ error: 'Authentication failed' }, { status: 400 });\n  }\n}",
                "embedding": [0.2] * 384,
                "language": "typescript",
                "metadata": {"has_functions": True, "has_imports": True}
            },
            {
                "filename": "components/ui/button.tsx",
                "code": "import * as React from 'react';\nimport { cn } from '@/lib/utils';\n\nexport interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {\n  variant?: 'default' | 'outline' | 'ghost';\n  size?: 'default' | 'sm' | 'lg';\n}\n\nconst Button = React.forwardRef<HTMLButtonElement, ButtonProps>(\n  ({ className, variant = 'default', size = 'default', ...props }, ref) => {\n    return (\n      <button\n        className={cn(\n          'inline-flex items-center justify-center rounded-md text-sm font-medium',\n          'transition-colors focus-visible:outline-none focus-visible:ring-2',\n          className\n        )}\n        ref={ref}\n        {...props}\n      />\n    );\n  }\n);\nButton.displayName = 'Button';\n\nexport { Button };",
                "embedding": [0.3] * 384,
                "language": "typescript",
                "metadata": {"has_functions": True, "has_imports": True}
            }
        ]
    
    def _get_placeholder_metadata(self) -> Dict[str, Any]:
        """Get placeholder project metadata for testing"""
        return {
            "package.json": json.dumps({
                "name": "example-project",
                "dependencies": {
                    "react": "^18.0.0",
                    "next": "^13.0.0",
                    "typescript": "^5.0.0",
                    "@radix-ui/react-dialog": "^1.0.0",
                    "lucide-react": "^0.300.0"
                },
                "devDependencies": {
                    "tailwindcss": "^3.0.0",
                    "eslint": "^8.0.0"
                }
            })
        }
    
    async def _generate_ticket_with_llm(self, contextual_prompt: str, user_request: str) -> Dict[str, Any]:
        """Generate ticket using OpenAI LLM with contextual prompt"""
        try:
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": contextual_prompt
                    },
                    {
                        "role": "user",
                        "content": (
                            "Generate a concise, LLM-ready implementation ticket in Markdown without any placeholder fields or meta headers. "
                            "Do NOT include sections like 'Example Code Snippet', 'Assigned To', 'Due Date', 'Tags', 'Ticket ID', 'Project', 'Component', 'Priority', 'Complexity', 'Status'. "
                            "Focus on: Title, Summary, Intent, Scope, Files to Modify (list real paths if known or leave empty), Considerations, Acceptance Criteria as checklist. "
                            f"User request: {user_request}"
                        )
                    }
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            markdown_content = (response.choices[0].message.content or "").strip()
            markdown_content = self._sanitize_ticket(markdown_content)
            
            # Parse the generated ticket
            ticket = self._parse_generated_ticket(markdown_content)
            
            return ticket
            
        except Exception as e:
            print(f"Error generating ticket with LLM: {e}")
            return {
                "title": "Error generating ticket",
                "description": f"Failed to generate ticket: {str(e)}",
                "acceptance_criteria": [],
                "files_to_modify": []
            }
    
    def _parse_generated_ticket(self, markdown_content: str) -> Dict[str, Any]:
        """Parse the generated markdown ticket"""
        lines = markdown_content.split('\n')
        
        # Extract title
        title = "Generated Ticket"
        for line in lines:
            if line.startswith('# '):
                title = line[2:].strip()
                break
        
        # Extract description
        description = markdown_content
        
        # Extract acceptance criteria
        acceptance_criteria = []
        in_criteria_section = False
        for line in lines:
            if line.strip() == "## Acceptance Criteria":
                in_criteria_section = True
                continue
            elif line.startswith('## ') and in_criteria_section:
                in_criteria_section = False
                break
            elif in_criteria_section and line.strip().startswith('- [ ]'):
                criteria = line.strip()[5:].strip()
                if criteria:
                    acceptance_criteria.append(criteria)
        
        # Extract files to modify
        files_to_modify = []
        in_files_section = False
        for line in lines:
            if line.strip() == "## Files to Modify":
                in_files_section = True
                continue
            elif line.startswith('## ') and in_files_section:
                in_files_section = False
                break
            elif in_files_section and '`' in line:
                import re
                matches = re.findall(r'`([^`]+)`', line)
                if matches:
                    files_to_modify.append(matches[0])
        
        return {
            "title": title,
            "description": description,
            "acceptance_criteria": acceptance_criteria,
            "files_to_modify": files_to_modify,
            "raw_markdown": markdown_content
        }

    def _sanitize_ticket(self, md: str) -> str:
        disallowed_patterns = [
            r"\*\*Assigned To:\*\*.*\n?",
            r"\*\*Due Date:\*\*.*\n?",
            r"\*\*Tags:\*\*.*\n?",
            r"\*\*Ticket ID:\*\*.*\n?",
            r"\*\*Project:\*\*.*\n?",
            r"\*\*Component:\*\*.*\n?",
            r"\*\*Priority:\*\*.*\n?",
            r"\*\*Complexity:\*\*.*\n?",
            r"\*\*Status:\*\*.*\n?",
            r"(?si)##\s*Example Code Snippet.*?(?:\n##|\Z)",
        ]
        import re
        cleaned = md
        for pat in disallowed_patterns:
            cleaned = re.sub(pat, "", cleaned)
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned
    
    async def search_code_semantically(self, query: str, repo_url: str) -> List[Dict[str, Any]]:
        """Search for code semantically using RAG"""
        try:
            # This would use the actual indexed embeddings
            # For now, return placeholder results
            return [
                {
                    "filename": "app/components/Example.tsx",
                    "code": "// Example code that matches the query",
                    "score": 0.95,
                    "language": "typescript"
                }
            ]
        except Exception as e:
            return [{"error": str(e)}]
    
    async def get_project_insights(self, repo_url: str) -> Dict[str, Any]:
        """Get comprehensive project insights"""
        try:
            # This would analyze the indexed repository
            # For now, return placeholder insights
            return {
                "technology_stack": {
                    "languages": ["typescript", "javascript"],
                    "frameworks": ["next.js", "react"],
                    "libraries": ["@radix-ui/react-dialog", "lucide-react"]
                },
                "architectural_patterns": ["Component-based", "App Router"],
                "coding_conventions": {
                    "naming": "camelCase",
                    "structure": "Feature-based"
                },
                "best_practices": ["TypeScript", "Component Composition", "Error Handling"]
            }
        except Exception as e:
            return {"error": str(e)}
