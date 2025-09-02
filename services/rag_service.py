#!/usr/bin/env python3
"""
RAG Service for intelligent code search and context retrieval
"""

import os
import asyncio
from typing import Dict, List, Optional, Any, Tuple
import json
import re
# from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

@dataclass
class CodeChunk:
    filename: str
    code: str
    language: str
    score: float
    metadata: Dict[str, Any]
    location: str

@dataclass
class ProjectContext:
    technology_stack: Dict[str, Any]
    architectural_patterns: List[str]
    coding_conventions: Dict[str, Any]
    similar_implementations: List[CodeChunk]
    best_practices: List[str]
    dependencies: Dict[str, List[str]]

class RAGService:
    def __init__(self, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_model = None
        self.embedding_dimension = 384  # Dimension of all-MiniLM-L6-v2
        
    async def analyze_user_request(self, user_request: str) -> Dict[str, Any]:
        """Analyze user request to understand intent and requirements"""
        analysis = {
            "intent": self._classify_intent(user_request),
            "complexity": self._estimate_complexity(user_request),
            "scope": self._determine_scope(user_request),
            "keywords": self._extract_keywords(user_request),
            "technology_hints": self._extract_technology_hints(user_request)
        }
        return analysis
    
    def _classify_intent(self, request: str) -> str:
        """Classify the intent of the user request"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["add", "create", "implement", "build"]):
            return "feature_implementation"
        elif any(word in request_lower for word in ["fix", "bug", "error", "issue"]):
            return "bug_fix"
        elif any(word in request_lower for word in ["refactor", "improve", "optimize"]):
            return "refactoring"
        elif any(word in request_lower for word in ["security", "vulnerability", "auth"]):
            return "security_update"
        elif any(word in request_lower for word in ["performance", "speed", "efficiency"]):
            return "performance_optimization"
        else:
            return "general_implementation"
    
    def _estimate_complexity(self, request: str) -> str:
        """Estimate the complexity of the request"""
        words = request.split()
        if len(words) < 10:
            return "simple"
        elif len(words) < 30:
            return "medium"
        else:
            return "complex"
    
    def _determine_scope(self, request: str) -> str:
        """Determine the scope of the request"""
        request_lower = request.lower()
        
        if any(word in request_lower for word in ["component", "function", "class"]):
            return "component"
        elif any(word in request_lower for word in ["module", "service", "api"]):
            return "module"
        elif any(word in request_lower for word in ["system", "architecture", "database"]):
            return "system"
        else:
            return "component"
    
    def _extract_keywords(self, request: str) -> List[str]:
        """Extract important keywords from the request"""
        # Remove common words and extract meaningful keywords
        common_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "this", "that", "these", "those"}
        
        words = re.findall(r'\b\w+\b', request.lower())
        keywords = [word for word in words if word not in common_words and len(word) > 2]
        
        return list(set(keywords))
    
    def _extract_technology_hints(self, request: str) -> List[str]:
        """Extract technology hints from the request"""
        technology_keywords = {
            "react": ["react", "jsx", "component", "hook"],
            "vue": ["vue", "template", "composition"],
            "angular": ["angular", "service", "directive"],
            "flutter": ["flutter", "widget", "dart"],
            "python": ["python", "django", "flask", "fastapi"],
            "java": ["java", "spring", "maven"],
            "typescript": ["typescript", "ts", "interface", "type"],
            "javascript": ["javascript", "js", "node", "express"],
            "database": ["database", "sql", "mongodb", "postgres"],
            "api": ["api", "rest", "graphql", "endpoint"],
            "auth": ["authentication", "auth", "login", "jwt", "oauth"],
            "testing": ["test", "spec", "unit", "integration", "e2e"]
        }
        
        hints = []
        request_lower = request.lower()
        
        for tech, keywords in technology_keywords.items():
            if any(keyword in request_lower for keyword in keywords):
                hints.append(tech)
        
        return hints
    
    async def search_relevant_code(self, query: str, code_embeddings: List[Dict[str, Any]], top_k: int = 15) -> List[CodeChunk]:
        """Search for relevant code using semantic similarity"""
        try:
            similarities = []
            if self.embedding_model is not None:
                query_embedding = self.embedding_model.encode([query])[0]
                for chunk in code_embeddings:
                    if "embedding" in chunk:
                        chunk_embedding = np.array(chunk["embedding"])
                        similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                        similarities.append((similarity, chunk))
            else:
                q = query.lower()
                q_terms = set(re.findall(r"[a-z0-9_]+", q))
                for chunk in code_embeddings:
                    text = f"{chunk.get('filename','')}\n{chunk.get('code','')}".lower()
                    matches = sum(1 for t in q_terms if t and t in text)
                    length_penalty = min(len(text) / 10000.0, 1.0)
                    score = (matches / (len(q_terms) or 1)) * 0.7 + 0.3 * length_penalty
                    similarities.append((score, chunk))
            
            # Sort by similarity and get top results
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_results = similarities[:top_k]
            
            # Convert to CodeChunk objects
            code_chunks = []
            for score, chunk in top_results:
                if score > 0.3:  # Minimum similarity threshold
                    code_chunks.append(CodeChunk(
                        filename=chunk.get("filename", ""),
                        code=chunk.get("code", ""),
                        language=chunk.get("language", ""),
                        score=float(score),
                        metadata=chunk.get("metadata", {}),
                        location=chunk.get("location", "")
                    ))
            
            return code_chunks
            
        except Exception as e:
            print(f"Error in search_relevant_code: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    async def build_project_context(self, code_chunks: List[CodeChunk], project_metadata: Dict[str, Any]) -> ProjectContext:
        """Build comprehensive project context from code chunks and metadata"""
        
        # Analyze technology stack
        technology_stack = self._analyze_technology_stack(code_chunks, project_metadata)
        
        # Detect architectural patterns
        architectural_patterns = self._detect_architectural_patterns(code_chunks)
        
        # Analyze coding conventions
        coding_conventions = self._analyze_coding_conventions(code_chunks)
        
        # Extract similar implementations
        similar_implementations = self._extract_similar_implementations(code_chunks)
        
        # Identify best practices
        best_practices = self._identify_best_practices(code_chunks)
        
        # Analyze dependencies
        dependencies = self._analyze_dependencies(code_chunks, project_metadata)
        
        return ProjectContext(
            technology_stack=technology_stack,
            architectural_patterns=architectural_patterns,
            coding_conventions=coding_conventions,
            similar_implementations=similar_implementations,
            best_practices=best_practices,
            dependencies=dependencies
        )
    
    def _analyze_technology_stack(self, code_chunks: List[CodeChunk], project_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the technology stack from code chunks"""
        stack = {
            "languages": {},
            "frameworks": {},
            "libraries": {},
            "tools": {},
            "databases": {},
            "platforms": {}
        }
        
        # Count languages
        for chunk in code_chunks:
            lang = chunk.language
            if lang not in stack["languages"]:
                stack["languages"][lang] = 0
            stack["languages"][lang] += 1
        
        # Analyze code content for frameworks and libraries
        for chunk in code_chunks:
            code_lower = chunk.code.lower()
            
            # Detect frameworks
            if "react" in code_lower or "jsx" in code_lower:
                stack["frameworks"]["react"] = stack["frameworks"].get("react", 0) + 1
            if "vue" in code_lower:
                stack["frameworks"]["vue"] = stack["frameworks"].get("vue", 0) + 1
            if "angular" in code_lower:
                stack["frameworks"]["angular"] = stack["frameworks"].get("angular", 0) + 1
            if "flutter" in code_lower:
                stack["frameworks"]["flutter"] = stack["frameworks"].get("flutter", 0) + 1
            if "django" in code_lower:
                stack["frameworks"]["django"] = stack["frameworks"].get("django", 0) + 1
            if "fastapi" in code_lower:
                stack["frameworks"]["fastapi"] = stack["frameworks"].get("fastapi", 0) + 1
            if "spring" in code_lower:
                stack["frameworks"]["spring"] = stack["frameworks"].get("spring", 0) + 1
            
            # Detect libraries
            if "axios" in code_lower:
                stack["libraries"]["axios"] = stack["libraries"].get("axios", 0) + 1
            if "lodash" in code_lower:
                stack["libraries"]["lodash"] = stack["libraries"].get("lodash", 0) + 1
            if "moment" in code_lower:
                stack["libraries"]["moment"] = stack["libraries"].get("moment", 0) + 1
            if "pandas" in code_lower:
                stack["libraries"]["pandas"] = stack["libraries"].get("pandas", 0) + 1
        
        return stack
    
    def _detect_architectural_patterns(self, code_chunks: List[CodeChunk]) -> List[str]:
        """Detect architectural patterns from code"""
        patterns = []
        
        # Analyze file structure and naming
        filenames = [chunk.filename for chunk in code_chunks]
        
        # MVC Pattern
        if any("model" in f.lower() for f in filenames) and any("view" in f.lower() for f in filenames) and any("controller" in f.lower() for f in filenames):
            patterns.append("MVC")
        
        # Component-based architecture
        if any("component" in f.lower() for f in filenames) or any("components" in f.lower() for f in filenames):
            patterns.append("Component-based")
        
        # Service layer pattern
        if any("service" in f.lower() for f in filenames):
            patterns.append("Service Layer")
        
        # Repository pattern
        if any("repository" in f.lower() for f in filenames):
            patterns.append("Repository Pattern")
        
        # Clean Architecture
        if any("domain" in f.lower() for f in filenames) and any("application" in f.lower() for f in filenames):
            patterns.append("Clean Architecture")
        
        return patterns
    
    def _analyze_coding_conventions(self, code_chunks: List[CodeChunk]) -> Dict[str, Any]:
        """Analyze coding conventions from code"""
        conventions = {
            "naming": {},
            "structure": {},
            "style": {}
        }
        
        # Analyze naming conventions
        for chunk in code_chunks:
            code = chunk.code
            
            # Function naming
            function_patterns = [
                r'def\s+([a-z_][a-z0-9_]*)',  # Python
                r'function\s+([a-z][a-zA-Z0-9]*)',  # JavaScript
                r'const\s+([a-z][a-zA-Z0-9]*)\s*=',  # JavaScript/TypeScript
            ]
            
            for pattern in function_patterns:
                matches = re.findall(pattern, code)
                for match in matches:
                    if match[0].isupper():
                        conventions["naming"]["PascalCase"] = conventions["naming"].get("PascalCase", 0) + 1
                    elif "_" in match:
                        conventions["naming"]["snake_case"] = conventions["naming"].get("snake_case", 0) + 1
                    else:
                        conventions["naming"]["camelCase"] = conventions["naming"].get("camelCase", 0) + 1
        
        return conventions
    
    def _extract_similar_implementations(self, code_chunks: List[CodeChunk]) -> List[CodeChunk]:
        """Extract similar implementations for reference"""
        # Group by functionality and return the best examples
        similar_impls = []
        
        # Group by file type and functionality
        grouped = {}
        for chunk in code_chunks:
            if chunk.score > 0.7:  # High relevance threshold
                key = f"{chunk.language}_{chunk.filename.split('/')[-1]}"
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(chunk)
        
        # Take the best example from each group
        for group_key, chunks in grouped.items():
            best_chunk = max(chunks, key=lambda x: x.score)
            similar_impls.append(best_chunk)
        
        return similar_impls[:5]  # Limit to top 5 examples
    
    def _identify_best_practices(self, code_chunks: List[CodeChunk]) -> List[str]:
        """Identify best practices from the codebase"""
        practices = []
        
        for chunk in code_chunks:
            code = chunk.code
            
            # Error handling
            if "try:" in code or "catch" in code or "except" in code:
                practices.append("Error Handling")
            
            # Type annotations
            if ":" in code and ("def " in code or "function" in code):
                practices.append("Type Annotations")
            
            # Documentation
            if '"""' in code or "'''" in code or "//" in code:
                practices.append("Code Documentation")
            
            # Testing
            if "test" in code.lower() or "spec" in code.lower():
                practices.append("Testing")
        
        return list(set(practices))
    
    def _analyze_dependencies(self, code_chunks: List[CodeChunk], project_metadata: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze project dependencies"""
        dependencies = {
            "runtime": [],
            "development": [],
            "external": []
        }
        
        # Extract from project metadata if available
        if "package.json" in project_metadata:
            try:
                package_data = json.loads(project_metadata["package.json"])
                dependencies["runtime"] = list(package_data.get("dependencies", {}).keys())
                dependencies["development"] = list(package_data.get("devDependencies", {}).keys())
            except:
                pass
        
        # Extract from code imports
        for chunk in code_chunks:
            code = chunk.code
            
            # Python imports
            import_matches = re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            from_matches = re.findall(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)', code)
            
            # JavaScript/TypeScript imports
            require_matches = re.findall(r'require\([\'"]([^\'"]+)[\'"]\)', code)
            import_js_matches = re.findall(r'import\s+.*?from\s+[\'"]([^\'"]+)[\'"]', code)
            
            all_imports = import_matches + from_matches + require_matches + import_js_matches
            
            for imp in all_imports:
                if imp not in dependencies["external"]:
                    dependencies["external"].append(imp)
        
        return dependencies
    
    async def generate_contextual_prompt(self, user_request: str, project_context: ProjectContext, request_analysis: Dict[str, Any]) -> str:
        """Generate a contextual prompt for ticket generation"""
        
        # Build technology-specific context
        tech_context = self._build_technology_context(project_context.technology_stack)
        
        # Build architectural context
        arch_context = self._build_architectural_context(project_context.architectural_patterns)
        
        # Build coding conventions context
        conventions_context = self._build_conventions_context(project_context.coding_conventions)
        
        # Build similar implementations context
        examples_context = self._build_examples_context(project_context.similar_implementations)
        
        # Build the complete prompt
        prompt = f"""
You are a senior technical lead generating implementation tickets for a project with the following characteristics:

{tech_context}

{arch_context}

{conventions_context}

USER REQUEST ANALYSIS:
- Intent: {request_analysis['intent']}
- Complexity: {request_analysis['complexity']}
- Scope: {request_analysis['scope']}
- Keywords: {', '.join(request_analysis['keywords'])}
- Technology Hints: {', '.join(request_analysis['technology_hints'])}

{examples_context}

USER REQUEST: "{user_request}"

CRITICAL INSTRUCTIONS:
1. Follow the EXACT patterns and conventions shown in the examples above
2. Use the SAME technology stack and libraries detected in the project
3. Maintain consistency with the architectural patterns identified
4. Reference ACTUAL files and structures from the project
5. Include specific implementation details based on the project's coding style
6. Consider the complexity level and scope identified in the analysis

Generate a comprehensive implementation ticket that is perfectly aligned with this project's specific characteristics and patterns.
"""
        
        return prompt
    
    def _build_technology_context(self, technology_stack: Dict[str, Any]) -> str:
        """Build technology-specific context"""
        context = "TECHNOLOGY STACK:\n"
        
        if technology_stack["languages"]:
            main_languages = sorted(technology_stack["languages"].items(), key=lambda x: x[1], reverse=True)[:3]
            context += f"- Primary Languages: {', '.join([lang for lang, _ in main_languages])}\n"
        
        if technology_stack["frameworks"]:
            main_frameworks = sorted(technology_stack["frameworks"].items(), key=lambda x: x[1], reverse=True)[:3]
            context += f"- Frameworks: {', '.join([fw for fw, _ in main_frameworks])}\n"
        
        if technology_stack["libraries"]:
            main_libraries = sorted(technology_stack["libraries"].items(), key=lambda x: x[1], reverse=True)[:5]
            context += f"- Key Libraries: {', '.join([lib for lib, _ in main_libraries])}\n"
        
        return context
    
    def _build_architectural_context(self, architectural_patterns: List[str]) -> str:
        """Build architectural context"""
        if not architectural_patterns:
            return "ARCHITECTURE: Standard application architecture\n"
        
        context = "ARCHITECTURAL PATTERNS:\n"
        for pattern in architectural_patterns:
            context += f"- {pattern}\n"
        
        return context
    
    def _build_conventions_context(self, coding_conventions: Dict[str, Any]) -> str:
        """Build coding conventions context"""
        context = "CODING CONVENTIONS:\n"
        
        if coding_conventions["naming"]:
            naming_style = max(coding_conventions["naming"].items(), key=lambda x: x[1])[0]
            context += f"- Naming Convention: {naming_style}\n"
        
        return context
    
    def _build_examples_context(self, similar_implementations: List[CodeChunk]) -> str:
        """Build examples context from similar implementations"""
        if not similar_implementations:
            return "EXAMPLES: No specific examples available\n"
        
        context = "SIMILAR IMPLEMENTATIONS (for reference):\n"
        for i, impl in enumerate(similar_implementations[:3], 1):
            context += f"{i}. File: {impl.filename}\n"
            context += f"   Language: {impl.language}\n"
            context += f"   Code: {impl.code[:200]}...\n\n"
        
        return context
