#!/usr/bin/env python3
import os
import psycopg2
from psycopg2.extras import execute_values


def run_sql(conn, sql):
    cur = conn.cursor()
    cur.execute(sql)
    cur.close()


def setup():
    database_url = os.getenv('DATABASE_URL')
    if not database_url:
        print('DATABASE_URL not set')
        return False

    conn = psycopg2.connect(database_url)
    conn.autocommit = True

    try:
        run_sql(conn, "CREATE EXTENSION IF NOT EXISTS vector;")
        run_sql(conn, "CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        run_sql(conn, """
        CREATE TABLE IF NOT EXISTS repository_embeddings (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            location TEXT,
            code TEXT NOT NULL,
            embedding vector(384),
            language TEXT,
            file_type TEXT,
            metadata JSONB,
            repo_url TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        run_sql(conn, """
        CREATE INDEX IF NOT EXISTS repository_embeddings_embedding_idx
        ON repository_embeddings USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
        """)

        run_sql(conn, """
        CREATE TABLE IF NOT EXISTS project_metadata (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            content TEXT NOT NULL,
            file_type TEXT,
            repo_url TEXT NOT NULL,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            UNIQUE(filename, repo_url)
        );
        """)

        run_sql(conn, """
        ALTER TABLE tickets
        ADD COLUMN IF NOT EXISTS generation_method TEXT DEFAULT 'standard',
        ADD COLUMN IF NOT EXISTS context JSONB,
        ADD COLUMN IF NOT EXISTS raw_markdown TEXT;
        """)

        run_sql(conn, "CREATE INDEX IF NOT EXISTS idx_repository_embeddings_repo_url ON repository_embeddings(repo_url);")
        run_sql(conn, "CREATE INDEX IF NOT EXISTS idx_repository_embeddings_language ON repository_embeddings(language);")
        run_sql(conn, "CREATE INDEX IF NOT EXISTS idx_project_metadata_repo_url ON project_metadata(repo_url);")
        run_sql(conn, "CREATE INDEX IF NOT EXISTS idx_tickets_generation_method ON tickets(generation_method);")

        run_sql(conn, """
        CREATE TABLE IF NOT EXISTS context_block_sessions (
            id UUID PRIMARY KEY,
            user_id UUID,
            audio_file_path TEXT,
            full_transcription TEXT,
            status TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        run_sql(conn, """
        CREATE TABLE IF NOT EXISTS context_blocks (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            session_id UUID REFERENCES context_block_sessions(id),
            feature_name TEXT,
            transcript_segment TEXT,
            specflow_intent JSONB,
            specflow_roadmap JSONB,
            specflow_tasks JSONB,
            generated_ticket TEXT,
            repository_context TEXT,
            created_at TIMESTAMPTZ DEFAULT NOW()
        );
        """)

        print('Setup completed')
        return True
    finally:
        conn.close()


if __name__ == '__main__':
    setup()


