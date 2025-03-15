# GitHub Repository QnA System

A system that creates a Q&A interface for GitHub repositories using lightweight LLMs with Ollama.

## Features

- Clone and process GitHub repositories
- Create knowledge graphs in Neo4j
- Query repositories using natural language questions
- Use local LLMs through Ollama (llama3.2)

## Requirements

- Python 3.8+
- Node.js 14+
- Neo4j database
- Ollama with llama3.2 model installed

## Setup Instructions

### 1. Set up Neo4j

- Install Neo4j (Desktop or Docker)
- Create a new database with credentials
- Update `.env` file with your Neo4j credentials

#### Neo4j Docker setup

```bash
docker run --name neo4j-apoc -e NEO4J_AUTH=neo4j/password -p 7474:7474 -p 7687:7687 -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4J_PLUGINS=\[\"apoc\"\] -e NEO4J_dbms_security_procedures_unrestricted=apoc.* -e NEO4J_dbms_security_procedures_allowlist=apoc.* -v $PWD/neo4j/data:/data -v $PWD/neo4j/logs:/logs -v $PWD/neo4j/import:/import neo4j:latest
```
Inside Docker shell do

```bash
cp  /var/lib/neo4j/labs/apoc-2025.02.0-core.jar /var/lib/neo4j/plugins/
```

### 2. Install Ollama and Models

```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull the model
ollama pull llama3.2
```

### 3. Set up the Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### 4. Set up the Frontend

```bash
cd frontend
npm install
npm start
```


## Usage

1. Visit http://localhost:3000 in your browser
2. Enter a GitHub repository URL and click "Process Repository"
3. Select the processed repository from the dropdown
4. Enter your question and click "Ask Question"

## Architecture

- **Backend**: FastAPI, LangChain, LangGraph, Neo4j
- **Frontend**: React, Tailwind CSS
- **LLM**: Ollama with llama3.2
- **Vector Storage**: Neo4j Graph Database