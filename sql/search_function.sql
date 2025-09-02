-- Function for vector similarity search on context items
-- Execute this in your Supabase SQL editor

CREATE OR REPLACE FUNCTION search_context_items(
    query_embedding vector(384),
    similarity_threshold float DEFAULT 0.7,
    match_count int DEFAULT 10
)
RETURNS TABLE (
    id uuid,
    content text,
    item_type text,
    similarity float,
    context_block_id uuid,
    is_resolved boolean
)
LANGUAGE sql STABLE
AS $$
    SELECT 
        context_block_items.id,
        context_block_items.content,
        context_block_items.item_type,
        1 - (context_block_items.embedding <=> query_embedding) as similarity,
        context_block_items.context_block_id,
        context_block_items.is_resolved
    FROM context_block_items
    WHERE 1 - (context_block_items.embedding <=> query_embedding) > similarity_threshold
    ORDER BY context_block_items.embedding <=> query_embedding
    LIMIT match_count;
$$;
