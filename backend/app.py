# app.py
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import shutil
import git
import glob
from git.exc import GitCommandError
from langchain.vectorstores import Neo4jVector
from langchain.embeddings import HuggingFaceEmbeddings  
from langchain_neo4j import Neo4jGraph
from langchain_ollama.llms import OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
#from langchain.chains import GraphQAChain
#from langchain.graphs.neo4j_graph import Neo4j
from langgraph.graph import END, StateGraph
from typing import Dict, TypedDict, List, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="GitHub Repo QnA System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Neo4j configuration - update these values as needed
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USERNAME = "neo4j"
NEO4J_PASSWORD = "password"
REPOS_DIR = "./repos"

# Create repos directory if it doesn't exist
os.makedirs(REPOS_DIR, exist_ok=True)

# Initialize models
class RepoRequest(BaseModel):
    repo_url: str
    
class QuestionRequest(BaseModel):
    repo_name: str
    question: str

# Initialize Ollama model
def get_llm():
    return OllamaLLM(model="llama3.2")

# Initialize embeddings model
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Clone and process repository
def clone_repository(repo_url):
    try:
        # Extract repo name from URL
        repo_name = repo_url.split("/")[-1].replace(".git", "")
        repo_path = f"{REPOS_DIR}/{repo_name}"
        
        # Remove existing repo if it exists
        if os.path.exists(repo_path):
            shutil.rmtree(repo_path)
        
        # Clone the repository
        logger.info(f"Cloning repository: {repo_url}")
        git.Repo.clone_from(repo_url, repo_path)
        
        return repo_name, repo_path
    except GitCommandError as e:
        logger.error(f"Git error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Failed to clone repository: {str(e)}")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

# Load files from repo and create documents
def load_documents(repo_path):
    files = []
    # Find all code files
    extensions = [
        "*.py", "*.js", "*.jsx", "*.ts", "*.tsx", "*.html", "*.css", "*.md", 
        "*.json", "*.yaml", "*.yml", "*.txt", "*.sh", "*.java", "*.c", "*.cpp", 
        "*.h", "*.go", "*.rb", "*.php"
    ]
    
    for ext in extensions:
        files.extend(glob.glob(f"{repo_path}/**/{ext}", recursive=True))
    
    # Convert files to documents
    documents = []
    for file_path in files:
        try:
            relative_path = os.path.relpath(file_path, repo_path)
            if '.git/' not in relative_path:  # Skip git files
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    documents.append(Document(
                        page_content=content,
                        metadata={"source": relative_path}
                    ))
        except Exception as e:
            logger.warning(f"Error reading file {file_path}: {str(e)}")
    
    return documents

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    return text_splitter.split_documents(documents)

# Create knowledge graph in Neo4j
def create_knowledge_graph(repo_name, docs):
    try:
        # Initialize Neo4j connection
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD
        )
        
        # Clear existing data for this repo
        graph.query(f"MATCH (n:Document {{repo: '{repo_name}'}}) DETACH DELETE n")
        
        # Create vector index if it doesn't exist
        try:
            graph.query(
                "CALL db.index.vector.createNodeIndex($index_name, $node_label, $property_key, $dimension, $similarity_metric)",
                {"index_name": f"{repo_name}_vector_index", 
                 "node_label": "Document", 
                 "property_key": "embedding", 
                 "dimension": 384,  # All-MiniLM-L6-v2 dimension
                 "similarity_metric": "cosine"}
            )
        except Exception as e:
            if "already exists" not in str(e):
                raise e
        
        # Store documents in Neo4j
        embeddings = get_embeddings()
        
        # Create nodes for documents
        for i, doc in enumerate(docs):
            # Get embedding
            embedding = embeddings.embed_query(doc.page_content)
            
            # Create node with document content
            graph.query(
                """
                CREATE (d:Document {
                    id: $id,
                    repo: $repo,
                    content: $content,
                    source: $source,
                    embedding: $embedding
                })
                """,
                {
                    "id": f"{repo_name}_{i}",
                    "repo": repo_name,
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "unknown"),
                    "embedding": embedding
                }
            )
        
        # Create relationships between documents from the same file
        graph.query(
            """
            MATCH (d1:Document {repo: $repo}), (d2:Document {repo: $repo})
            WHERE d1.source = d2.source AND d1.id <> d2.id
            CREATE (d1)-[:SAME_FILE]->(d2)
            """,
            {"repo": repo_name}
        )
        
        logger.info(f"Knowledge graph created for {repo_name} with {len(docs)} document chunks")
        return True
    except Exception as e:
        logger.error(f"Error creating knowledge graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create knowledge graph: {str(e)}")

# LangGraph implementation for querying the graph
class GraphState(TypedDict):
    question: str
    repo_name: str
    context: List[str]
    answer: str

# Query Neo4j for relevant documents
def retrieve(state: GraphState) -> GraphState:
    question = state["question"]
    repo_name = state["repo_name"]
    
    embeddings = get_embeddings()
    question_embedding = embeddings.embed_query(question)
    
    graph = Neo4jGraph(
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD
    )
    
    # Vector search for relevant documents
    result = graph.query(
        """
        MATCH (d:Document {repo: $repo})
WHERE d.embedding IS NOT NULL

WITH d, 
     reduce(s = 0.0, i IN range(0, size(d.embedding) - 1) | s + (d.embedding[i] * $embedding[i])) AS dotProduct,
     sqrt(reduce(s = 0.0, i IN range(0, size(d.embedding) - 1) | s + (d.embedding[i] * d.embedding[i]))) AS magnitudeA,
     sqrt(reduce(s = 0.0, i IN range(0, size($embedding) - 1) | s + ($embedding[i] * $embedding[i]))) AS magnitudeB

WITH d, dotProduct / (magnitudeA * magnitudeB) AS score
ORDER BY score DESC
LIMIT 5

RETURN d.content AS content, d.source AS source, score

        """,
        {"repo": repo_name, "embedding": question_embedding}
    )
    
    # Extract contexts with their sources
    contexts = []
    for row in result:
        contexts.append(f"File: {row['source']}\n{row['content']}")
    
    return {"question": question, "repo_name": repo_name, "context": contexts, "answer": ""}

# Generate answer based on retrieved contexts
def generate_answer(state: GraphState) -> GraphState:
    question = state["question"]
    contexts = state["context"]
    
    if not contexts:
        return {"question": question, "repo_name": state["repo_name"], "context": contexts, 
                "answer": "I couldn't find any relevant information in the repository to answer your question."}
    
    # Join contexts
    context_text = "\n\n".join(contexts)
    
    # Prompt for the LLM
    prompt = f"""
    You are an expert developer assistant that understands code repositories.
    
    Based on the following context from a GitHub repository, please answer this question:
    
    QUESTION: {question}
    
    CONTEXT:
    {context_text}
    
    Provide a concise and accurate answer based only on the information in the context.
    """
    
    # Get answer from LLM
    llm = get_llm()
    answer = llm.predict(prompt)
    
    return {"question": question, "repo_name": state["repo_name"], "context": contexts, "answer": answer}

# Create the workflow graph
def build_graph():
    # Define the workflow
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate_answer)
    
    # Add edges
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    return workflow.compile()

# Graph query function
def query_graph(repo_name, question):
    try:
        graph_app = build_graph()
        
        # Execute the graph
        result = graph_app.invoke({
            "question": question,
            "repo_name": repo_name,
            "context": [],
            "answer": ""
        })
        
        return result["answer"]
    except Exception as e:
        logger.error(f"Error querying graph: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to query knowledge graph: {str(e)}")

# API Endpoints
@app.post("/process-repo")
async def process_repository(repo_request: RepoRequest):
    try:
        repo_name, repo_path = clone_repository(repo_request.repo_url)
        documents = load_documents(repo_path)
        chunks = split_documents(documents)
        create_knowledge_graph(repo_name, chunks)
        
        return {"status": "success", "repo_name": repo_name, "message": f"Repository processed successfully with {len(chunks)} document chunks"}
    except Exception as e:
        logger.error(f"Error in process_repository: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(question_request: QuestionRequest):
    try:
        answer = query_graph(question_request.repo_name, question_request.question)
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error in ask_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/repos")
async def list_repos():
    repos = []
    for repo_dir in os.listdir(REPOS_DIR):
        if os.path.isdir(os.path.join(REPOS_DIR, repo_dir)):
            repos.append(repo_dir)
    return {"repos": repos}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)