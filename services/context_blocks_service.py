import os
import uuid
import json
from typing import Any, Dict, List, Optional
from openai import OpenAI
# from sentence_transformers import SentenceTransformer
from .whisper_service import transcribe_audio_to_text
from utils.supabase_client import supabase


class ContextBlocksService:
    def __init__(self, openai_api_key: Optional[str] = None):
        api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        self.client = OpenAI(api_key=api_key)
        # self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.embedding_model = None

    def process_meeting(self, audio_file_path: str, user_id: Optional[str] = None, repo_url: Optional[str] = None) -> Dict[str, Any]:
        print(f"Starting process_meeting with user_id: {user_id}, repo_url: {repo_url}")
        session_id = str(uuid.uuid4())
        print(f"Generated session_id: {session_id}")
        
        transcription = transcribe_audio_to_text(audio_file_path)
        print(f"Transcription length: {len(transcription)} characters")
        
        # Try to persist original audio in Supabase Storage (public bucket 'recordings')
        audio_public_url: Optional[str] = None
        if supabase is not None:
            try:
                storage = supabase.storage
                storage_bucket = storage.from_("recordings")
                storage_path = f"{session_id}.webm"
                # Upload the local file to storage
                print(f"Uploading audio to storage at path: {storage_path}")
                storage_bucket.upload(file=audio_file_path, path=storage_path)
                # Build public URL (assumes 'recordings' bucket is public)
                try:
                    public_url_resp = storage_bucket.get_public_url(storage_path)
                    # supabase-py may return dict or object; handle both
                    if isinstance(public_url_resp, dict):
                        audio_public_url = public_url_resp.get("publicUrl") or public_url_resp.get("data", {}).get("publicUrl")
                    else:
                        audio_public_url = getattr(public_url_resp, "public_url", None) or getattr(public_url_resp, "publicUrl", None)
                    print(f"Audio public URL: {audio_public_url}")
                except Exception as e:
                    print(f"Failed to get public URL: {e}")
            except Exception as e:
                print(f"Audio upload to storage failed: {e}")
        
        if supabase is not None:
            session_row = {
                "id": session_id,
                "user_id": user_id,
                "repo_url": repo_url,
                "live_transcription": transcription,
                "status": "completed",
                "session_type": "audio_upload"
            }
            try:
                print(f"Inserting session into database: {session_row}")
                result = supabase.table("context_sessions").insert(session_row).execute()
                print(f"Session inserted successfully: {result}")
            except Exception as e:
                print(f"Failed to create session: {e}")
        else:
            print("Supabase is None, skipping session creation")
        
        context_blocks = self.analyze_and_generate_context_blocks(session_id, transcription)
        print(f"Generated {len(context_blocks)} context blocks")
        
        return {
            "session_id": session_id,
            "status": "completed", 
            "transcription": transcription,
            "context_blocks": context_blocks,
            "audio_url": audio_public_url
        }

    def analyze_and_generate_context_blocks(self, session_id: str, transcription: str) -> List[Dict[str, Any]]:
        prompt = """
        Analyze this development conversation and identify specific features or tasks being discussed.
        For each feature/task, provide:
        1. A clear title (max 50 chars)
        2. Brief description (max 200 chars)
        3. The specific transcript segment discussing it
        4. The main intent/goal

        Return JSON with array 'blocks', each containing: title, description, transcript_segment, feature_intent
        Focus on actionable development tasks and features.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return only valid JSON. Focus on development features and tasks."},
                    {"role": "user", "content": f"{prompt}\n\nTranscription:\n{transcription}"}
                ],
                temperature=0.2,
            )
            
            content = response.choices[0].message.content or "{}"
            data = json.loads(content)
            blocks = data.get("blocks", [])
            
            created_blocks = []
            for block in blocks:
                created_block = self.create_context_block(session_id, block)
                if created_block:
                    created_blocks.append(created_block)
            
            return created_blocks
        except Exception as e:
            print(f"Error analyzing context: {e}")
            return []

    def create_context_block(self, session_id: str, block_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print(f"Creating context block for session {session_id}")
        if not supabase:
            print("Supabase is None, cannot create context block")
            return None
        
        try:
            block_id = str(uuid.uuid4())
            print(f"Generated block_id: {block_id}")
            
            context_block = {
                "id": block_id,
                "session_id": session_id,
                "title": block_data.get("title", "")[:50],
                "description": block_data.get("description", "")[:200],
                "transcript_segment": block_data.get("transcript_segment", ""),
                "feature_intent": block_data.get("feature_intent", ""),
                "status": "active"
            }
            
            print(f"Inserting context block: {context_block}")
            result = supabase.table("context_blocks").insert(context_block).execute()
            print(f"Context block inserted successfully: {result}")
            
            items = self.generate_context_items(block_id, context_block)
            print(f"Generated {len(items)} items for block {block_id}")
            
            return {
                "id": block_id,
                "title": context_block["title"],
                "description": context_block["description"],
                "transcript_segment": context_block["transcript_segment"],
                "feature_intent": context_block["feature_intent"],
                "items": items
            }
        except Exception as e:
            print(f"Error creating context block: {e}")
            return None

    def generate_context_items(self, context_block_id: str, context_block: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt = f"""
        Based on this development feature/task, generate 3-5 specific actionable items:
        
        Feature: {context_block['title']}
        Description: {context_block['description']}
        Intent: {context_block['feature_intent']}
        
        Generate items that help developers:
        - Implementation recommendations
        - Questions to clarify requirements  
        - Technical considerations
        - Next steps

        Return JSON with array 'items', each containing: content (max 150 chars), item_type (recommendation/question/consideration/step)
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that returns only valid JSON. Always return a JSON object with an 'items' array containing development items."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
            )
            
            content = response.choices[0].message.content or "{}"
            print(f"OpenAI response content: {content}")
            
            # Handle empty or malformed responses
            if not content or content.strip() == "":
                print("Empty response from OpenAI, using fallback items")
                return self.create_fallback_items(context_block_id, context_block)
            
            # Try to parse JSON
            try:
                data = json.loads(content)
            except json.JSONDecodeError as json_error:
                print(f"JSON decode error: {json_error}")
                print(f"Raw content: {content}")
                return self.create_fallback_items(context_block_id, context_block)
            
            items = data.get("items", [])
            if not items:
                print("No items found in response, using fallback items")
                return self.create_fallback_items(context_block_id, context_block)
            
            created_items = []
            for item in items:
                created_item = self.create_context_item(context_block_id, item)
                if created_item:
                    created_items.append(created_item)
            
            return created_items
        except Exception as e:
            print(f"Error generating context items: {e}")
            return self.create_fallback_items(context_block_id, context_block)

    def create_fallback_items(self, context_block_id: str, context_block: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback items when OpenAI fails"""
        fallback_items = [
            {
                "content": f"Review requirements for {context_block['title']}",
                "item_type": "consideration"
            },
            {
                "content": f"Plan implementation approach for {context_block['title']}",
                "item_type": "recommendation"
            },
            {
                "content": f"Identify potential challenges in {context_block['title']}",
                "item_type": "consideration"
            }
        ]
        
        created_items = []
        for item in fallback_items:
            created_item = self.create_context_item(context_block_id, item)
            if created_item:
                created_items.append(created_item)
        
        return created_items

    def create_context_item(self, context_block_id: str, item_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        print(f"Creating context item for block {context_block_id}")
        if not supabase:
            print("Supabase is None, cannot create context item")
            return None
        
        try:
            item_content = item_data.get("content", "")[:150]
            item_type = item_data.get("item_type", "recommendation")
            
            # Temporarily disable embeddings due to dependency issues
            embedding = None  # self.embedding_model.encode(item_content).tolist() if self.embedding_model else None
            
            item = {
                "id": str(uuid.uuid4()),
                "context_block_id": context_block_id,
                "content": item_content,
                "item_type": item_type,
                "embedding": embedding,
                "is_resolved": False
            }
            
            print(f"Inserting context item: {item}")
            result = supabase.table("context_block_items").insert(item).execute()
            print(f"Context item inserted successfully: {result}")
            
            return {
                "id": item["id"],
                "content": item_content,
                "item_type": item_type,
                "is_resolved": False
            }
        except Exception as e:
            print(f"Error creating context item: {e}")
            return None

    def resolve_item_to_prompt(self, item_id: str, resolution_context: str) -> Dict[str, Any]:
        if not supabase:
            return {"success": False, "error": "Database not available"}
        
        try:
            item = supabase.table("context_block_items").select("*").eq("id", item_id).single().execute()
            if not item.data:
                return {"success": False, "error": "Item not found"}
            
            item_data = item.data
            
            prompt_text = self.generate_specific_prompt(item_data["content"], item_data["item_type"], resolution_context)
            
            supabase.table("context_block_items").update({
                "is_resolved": True,
                "generated_prompt": prompt_text,
                "task_context": {"resolution": resolution_context}
            }).eq("id", item_id).execute()
            
            return {
                "success": True,
                "prompt": prompt_text,
                "item_id": item_id
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_specific_prompt(self, item_content: str, item_type: str, context: str) -> str:
        prompt = f"""
        Convert this development item into a specific, actionable prompt:
        
        Item: {item_content}
        Type: {item_type}
        Context: {context}
        
        Create a clear, specific prompt for a coding assistant.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Create specific, actionable prompts for development tasks."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
            )
            
            return response.choices[0].message.content or ""
        except Exception as e:
            print(f"Error generating prompt: {e}")
            return f"Help me with: {item_content}"

    def build_system_prompt(self, context_block_id: str) -> Dict[str, Any]:
        if not supabase:
            return {"success": False, "error": "Database not available"}
        
        try:
            block = supabase.table("context_blocks").select("*").eq("id", context_block_id).single().execute()
            items = supabase.table("context_block_items").select("*").eq("context_block_id", context_block_id).execute()
            
            if not block.data:
                return {"success": False, "error": "Context block not found"}
            
            block_data = block.data
            items_data = items.data or []
            
            resolved_prompts = [item["generated_prompt"] for item in items_data if item["is_resolved"] and item["generated_prompt"]]
            
            system_prompt = self.create_comprehensive_system_prompt(block_data, items_data, resolved_prompts)
            
            prompt_record = {
                "id": str(uuid.uuid4()),
                "context_block_id": context_block_id,
                "prompt_text": system_prompt,
                "planning_context": {
                    "total_items": len(items_data),
                    "resolved_items": len(resolved_prompts),
                    "feature_intent": block_data.get("feature_intent")
                },
                "referenced_items": [item["id"] for item in items_data],
                "is_active": True
            }
            
            supabase.table("system_prompts").insert(prompt_record).execute()
            
            return {
                "success": True,
                "system_prompt": system_prompt,
                "prompt_id": prompt_record["id"]
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def create_comprehensive_system_prompt(self, block_data: Dict[str, Any], items_data: List[Dict[str, Any]], resolved_prompts: List[str]) -> str:
        feature_title = block_data.get("title", "")
        feature_description = block_data.get("description", "")
        feature_intent = block_data.get("feature_intent", "")
        
        system_prompt = f"""You are an expert software development assistant working on the following feature:

## Feature: {feature_title}

**Description:** {feature_description}

**Intent:** {feature_intent}

**Context Items:**
"""
        
        for item in items_data:
            status = "✅ RESOLVED" if item["is_resolved"] else "⏳ PENDING"
            system_prompt += f"- [{status}] {item['content']} ({item['item_type']})\n"
        
        if resolved_prompts:
            system_prompt += f"\n**Resolved Action Items:**\n"
            for i, prompt in enumerate(resolved_prompts, 1):
                system_prompt += f"{i}. {prompt}\n"
        
        system_prompt += """
**Your Role:**
- Provide specific, actionable guidance for this feature
- Reference the context items when relevant
- Help break down complex tasks into manageable steps
- Suggest best practices and implementation approaches
- Ask clarifying questions when requirements are unclear

Focus on helping the developer successfully implement this feature based on the analyzed context and resolved action items.
"""
        
        return system_prompt

    def search_similar_items(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if not supabase:
            return []
        
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            results = supabase.rpc("search_context_items", {
                "query_embedding": query_embedding,
                "similarity_threshold": 0.7,
                "match_count": top_k
            }).execute()
            
            return results.data or []
        except Exception as e:
            print(f"Error searching similar items: {e}")
            return []


