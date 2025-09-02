# 🧠 **SISTEMA INTELIGENTE DE TICKETS - COCOINDEX + RAG + LLM**

## 📋 **RESUMEN EJECUTIVO**

Este sistema implementa una arquitectura completa de **Retrieval-Augmented Generation (RAG)** para la generación inteligente de tickets de implementación técnica. Combina:

- **CocoIndex**: Para indexación inteligente de repositorios
- **Sentence Transformers**: Para embeddings semánticos
- **pgvector**: Para búsqueda vectorial en PostgreSQL
- **OpenAI GPT-4**: Para generación de tickets contextuales

## 🏗️ **ARQUITECTURA DEL SISTEMA**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Request  │───▶│  RAG Service    │───▶│  OpenAI LLM     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ CocoIndex       │◀───│ Vector Database │◀───│  Embeddings     │
│ Repository      │    │ (pgvector)      │    │ (Sentence       │
│ Indexing        │    │                 │    │  Transformers)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 🚀 **CARACTERÍSTICAS PRINCIPALES**

### 1. **Indexación Inteligente con CocoIndex**
- Clonación automática de repositorios
- Análisis de estructura de archivos
- Detección de lenguajes y frameworks
- Chunking semántico con Tree-sitter
- Generación de embeddings vectoriales

### 2. **Búsqueda Semántica con RAG**
- Análisis de intención del usuario
- Búsqueda vectorial de código relevante
- Extracción de contexto del proyecto
- Identificación de patrones arquitectónicos

### 3. **Generación Contextual con LLM**
- Prompts dinámicos basados en contexto
- Análisis de stack tecnológico
- Detección de convenciones de código
- Generación de tickets específicos del proyecto

## 📦 **INSTALACIÓN Y CONFIGURACIÓN**

### 1. **Instalar Dependencias**
```bash
pip install -r requirements.txt
```

### 2. **Configurar Variables de Entorno**
```bash
# .env
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
DATABASE_URL=your_database_url
```

### 3. **Configurar Base de Datos**
```bash
python setup_intelligent_database.py
```

### 4. **Ejecutar Tests**
```bash
python test_intelligent_system.py
```

## 🔧 **COMPONENTES DEL SISTEMA**

### 1. **CocoIndexService** (`services/cocoindex_service.py`)
```python
class CocoIndexService:
    async def index_repository(self, repo_url: str, github_token: Optional[str] = None)
    async def search_code(self, query: str, top_k: int = 10)
```

**Funcionalidades:**
- Clonación de repositorios
- Indexación con CocoIndex
- Generación de embeddings
- Almacenamiento en PostgreSQL con pgvector

### 2. **RAGService** (`services/rag_service.py`)
```python
class RAGService:
    async def analyze_user_request(self, user_request: str)
    async def search_relevant_code(self, query: str, code_embeddings: List[Dict])
    async def build_project_context(self, code_chunks: List[CodeChunk], project_metadata: Dict)
```

**Funcionalidades:**
- Análisis de intención del usuario
- Búsqueda semántica de código
- Construcción de contexto del proyecto
- Generación de prompts contextuales

### 3. **IntelligentTicketGenerator** (`services/intelligent_ticket_generator.py`)
```python
class IntelligentTicketGenerator:
    async def generate_intelligent_ticket(self, user_request: str, repo_url: str, github_token: Optional[str] = None)
    async def search_code_semantically(self, query: str, repo_url: str)
    async def get_project_insights(self, repo_url: str)
```

**Funcionalidades:**
- Orquestación del pipeline completo
- Generación de tickets con OpenAI
- Búsqueda semántica de código
- Análisis de insights del proyecto

## 🌐 **ENDPOINTS API**

### 1. **Generar Ticket Inteligente**
```http
POST /api/intelligent/generate-ticket
Content-Type: application/json

{
  "user_request": "Add user authentication with JWT",
  "repo_url": "https://github.com/user/repo",
  "github_token": "optional_github_token"
}
```

**Respuesta:**
```json
{
  "success": true,
  "ticket": {
    "title": "Implement JWT Authentication System",
    "description": "...",
    "acceptance_criteria": [...],
    "files_to_modify": [...],
    "raw_markdown": "..."
  },
  "context": {
    "indexing_result": {...},
    "request_analysis": {...},
    "project_context": {...}
  }
}
```

### 2. **Búsqueda Semántica de Código**
```http
POST /api/intelligent/search-code
Content-Type: application/json

{
  "query": "authentication login form",
  "repo_url": "https://github.com/user/repo",
  "top_k": 10
}
```

### 3. **Obtener Insights del Proyecto**
```http
POST /api/intelligent/project-insights
Content-Type: application/json

{
  "repo_url": "https://github.com/user/repo"
}
```

### 4. **Health Check**
```http
GET /api/intelligent/health
```

## 🗄️ **ESQUEMA DE BASE DE DATOS**

### 1. **repository_embeddings**
```sql
CREATE TABLE repository_embeddings (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    location TEXT,
    code TEXT NOT NULL,
    embedding vector(384),
    language TEXT,
    file_type TEXT,
    metadata JSONB,
    repo_url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 2. **project_metadata**
```sql
CREATE TABLE project_metadata (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL,
    content TEXT NOT NULL,
    file_type TEXT,
    repo_url TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(filename, repo_url)
);
```

### 3. **tickets** (actualizado)
```sql
ALTER TABLE tickets 
ADD COLUMN generation_method TEXT DEFAULT 'standard',
ADD COLUMN context JSONB,
ADD COLUMN raw_markdown TEXT;
```

## 🔍 **FUNCIÓN DE BÚSQUEDA VECTORIAL**

```sql
CREATE OR REPLACE FUNCTION search_code_embeddings(
    query_embedding vector(384),
    repo_url_filter TEXT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.3,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id INT,
    filename TEXT,
    location TEXT,
    code TEXT,
    language TEXT,
    file_type TEXT,
    metadata JSONB,
    similarity FLOAT
)
```

## 🧪 **TESTING**

### Ejecutar Tests Completos
```bash
python test_intelligent_system.py
```

### Tests Individuales
```python
# Test de generación de tickets
await test_intelligent_system()

# Test de búsqueda semántica
await test_code_search()

# Test de insights del proyecto
await test_project_insights()
```

## 📊 **MÉTRICAS Y MONITOREO**

### 1. **Métricas de Rendimiento**
- Tiempo de indexación de repositorios
- Tiempo de búsqueda semántica
- Tiempo de generación de tickets
- Precisión de embeddings

### 2. **Métricas de Calidad**
- Relevancia de código encontrado
- Calidad de tickets generados
- Satisfacción del usuario
- Tasa de aceptación de tickets

## 🔧 **CONFIGURACIÓN AVANZADA**

### 1. **Modelos de Embeddings**
```python
# Cambiar modelo de embeddings
embedding_model = "sentence-transformers/all-MiniLM-L6-v2"  # Default
embedding_model = "sentence-transformers/all-mpnet-base-v2"  # Más preciso
embedding_model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multilingüe
```

### 2. **Configuración de Chunking**
```python
# Tamaño de chunks para embeddings
chunk_size = 1500  # Caracteres por chunk
chunk_overlap = 200  # Overlap entre chunks
```

### 3. **Umbrales de Similitud**
```python
# Umbral mínimo para considerar código relevante
similarity_threshold = 0.3  # 0.0 a 1.0
```

## 🚨 **TROUBLESHOOTING**

### 1. **Error: pgvector extension not enabled**
```bash
# En Supabase SQL Editor
CREATE EXTENSION IF NOT EXISTS vector;
```

### 2. **Error: OpenAI API key not configured**
```bash
# Verificar variable de entorno
echo $OPENAI_API_KEY
```

### 3. **Error: Repository cloning failed**
```bash
# Verificar token de GitHub
# Verificar permisos del repositorio
```

### 4. **Error: Memory issues during indexing**
```python
# Reducir chunk_size
chunk_size = 1000  # En lugar de 1500
```

## 🔮 **ROADMAP FUTURO**

### Fase 1: Optimizaciones
- [ ] Caché de embeddings
- [ ] Indexación incremental
- [ ] Compresión de embeddings

### Fase 2: Funcionalidades Avanzadas
- [ ] Análisis de dependencias
- [ ] Detección de vulnerabilidades
- [ ] Sugerencias de refactoring

### Fase 3: Integración
- [ ] Webhook para repositorios
- [ ] Integración con GitHub Actions
- [ ] Dashboard de analytics

## 📞 **SOPORTE**

Para soporte técnico o preguntas sobre el sistema inteligente:

1. Revisar la documentación
2. Ejecutar tests de diagnóstico
3. Verificar logs del sistema
4. Contactar al equipo de desarrollo

---

**🎉 ¡El sistema inteligente está listo para revolucionar la generación de tickets!**
