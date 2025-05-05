import streamlit as st
import nest_asyncio
from io import BytesIO
from agno.agent import Agent
from agno.document.reader.pdf_reader import PDFReader
from agno.models.openai import OpenAIChat
from agno.knowledge.pdf_url import PDFUrlKnowledgeBase
from agno.knowledge.text import TextKnowledgeBase
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.embedder.openai import OpenAIEmbedder
from agno.vectordb.pgvector import PgVector
from agno.storage.agent.postgres import PostgresAgentStorage
from agno.knowledge.combined import CombinedKnowledgeBase
from pathlib import Path


# Apply nest_asyncio to allow nested event loops, required for running async functions in Streamlit
nest_asyncio.apply()

# Database connection string for PostgreSQL
DB_URL = "postgresql+psycopg://ai:ai@db:5432/ai"

# Function to set up the Assistant, utilizing caching for resource efficiency
@st.cache_resource
def setup_assistant(api_key: str) -> Agent:
    """Initializes and returns an AI Assistant agent with caching for efficiency.

    This function sets up an AI Assistant agent using the OpenAI GPT-4o-mini model 
    and configures it with a knowledge base, storage, and web search tools. The 
    assistant is designed to first search its knowledge base before querying the 
    internet, providing clear and concise answers.

    Args:
        api_key (str): The API key required to access the OpenAI services.

    Returns:
        Agent: An initialized Assistant agent configured with a language model, 
        knowledge base, storage, and additional tools for enhanced functionality."""
    

    pdf_kb = PDFUrlKnowledgeBase(
        vector_db=PgVector(
            table_name="auto_rag_docs",  
            db_url=DB_URL,  
            embedder=OpenAIEmbedder(id="text-embedding-ada-002", dimensions=1536, api_key=api_key),
        ),
        num_documents=3,
    )

    text_kb = TextKnowledgeBase(
        path=Path("data/text_docs"),  # Directory where .txt files will be stored
        vector_db=PgVector(
            table_name="auto_rag_text",  # Separate table for text knowledge
            db_url=DB_URL,
            embedder=OpenAIEmbedder(id="text-embedding-ada-002", dimensions=1536, api_key=api_key),
        ),
        num_documents=3,
    )

    # Combine knowledge bases
    knowledge_base = CombinedKnowledgeBase(
        sources=[
            pdf_kb,
            text_kb
        ],
        vector_db=PgVector(
            table_name="combined_documents",
            db_url=DB_URL,
            embedder=OpenAIEmbedder(id="text-embedding-ada-002", dimensions=1536, api_key=api_key),
        ),
    )
    # Load the knowledge base
    knowledge_base.load(recreate=False)

    llm = OpenAIChat(id="gpt-4o-mini", api_key=api_key)

    # Set up the Assistant with storage, knowledge base, and tools
    return Agent(
        name="auto_rag_agent",  # Name of the Assistant
        model=llm,  # Language model to be used
        storage=PostgresAgentStorage(table_name="auto_rag_storage", db_url=DB_URL),  
        knowledge=knowledge_base,
        tools=[DuckDuckGoTools()],  # Additional tool for web search via DuckDuckGo
        instructions=[
            "Search your knowledge base first.",  
            "If not found, search the internet.",  
            "Provide clear and concise answers.",  
        ],
        show_tool_calls=True,  
        search_knowledge=True,  
        markdown=True,  
        debug_mode=True,  
    )

# Function to add a PDF document to the knowledge base
def add_document(agent: Agent, file: BytesIO):
    """Add a PDF document to the agent's knowledge base.

    This function reads a PDF document from a file-like object and adds its contents to the specified agent's knowledge base. If the document is successfully read, the contents are loaded into the knowledge base with the option to upsert existing data.

    Args:
        agent (Agent): The agent whose knowledge base will be updated.
        file (BytesIO): A file-like object containing the PDF document to be added.

    Returns:
        None: The function does not return a value but provides feedback on whether the operation was successful."""
    reader = PDFReader()
    docs = reader.read(file)
    if docs:
        agent.knowledge.load_documents(docs, upsert=True)
        st.success("Document added to the knowledge base.")
    else:
        st.error("Failed to read the document.")

def add_text_document(file):
    """Reads a TXT file and adds its content to the text knowledge base."""
    target_dir = Path("data/text_docs")
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / file.name
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(file.read().decode("utf-8"))

    st.success("Text file added to the knowledge base. Reload page to use it.")

# Function to query the Assistant and return a response
def query_assistant(agent: Agent, question: str) -> str:
    """Queries the Assistant and returns a response.

    Args:
        agent (Agent): An instance of the Agent class used to process the query.
        question (str): The question to be asked to the Assistant.

    Returns:
        str: The response generated by the Assistant for the given question."""
    return agent.run(question).content

# Main function to handle Streamlit app layout and interactions
def main():
    """Main function to handle the layout and interactions for the Streamlit app.

    This function sets up the Streamlit app configuration, handles user inputs such
    as OpenAI API key, PDF uploads, and user questions, and interacts with an
    autonomous retrieval-augmented generation (RAG) assistant based on GPT-4o.
    
    The app allows users to upload PDF documents to enhance the knowledge base and
    submit questions to receive generated responses. 

    Side Effects:
        - Configures Streamlit page and title.
        - Prompts users to input an OpenAI API key and a question.
        - Allows users to upload PDF documents.
        - Displays responses generated by querying an assistant.

    Raises:
        StreamlitWarning: If the OpenAI API key is not provided."""
    st.set_page_config(page_title="AutoRAG", layout="wide")
    st.title("🤖 Auto-RAG: Autonomous RAG with GPT-4o")

    api_key = st.sidebar.text_input("Enter your OpenAI API Key 🔑", type="password")
    
    if not api_key:
        st.sidebar.warning("Enter your OpenAI API Key to proceed.")
        st.stop()

    assistant = setup_assistant(api_key)
    
    uploaded_file = st.sidebar.file_uploader("📄 Upload PDF or TXT", type=["pdf", "txt"])

    if uploaded_file and st.sidebar.button("🛠️ Add to Knowledge Base"):
        file_type = uploaded_file.name.split(".")[-1].lower()
        if file_type == "pdf":
            add_document(assistant, BytesIO(uploaded_file.read()))
        elif file_type == "txt":
            add_text_document(uploaded_file)
        else:
            st.error("Unsupported file type.")

    question = st.text_input("💬 Ask Your Question:")
    
    # When the user submits a question, query the assistant for an answer
    if st.button("🔍 Get Answer"):
        # Ensure the question is not empty
        if question.strip():
            with st.spinner("🤔 Thinking..."):
                # Query the assistant and display the response
                answer = query_assistant(assistant, question)
                st.write("📝 **Response:**", answer)
        else:
            # Show an error if the question input is empty
            st.error("Please enter a question.")

# Entry point of the application
if __name__ == "__main__":
    main()
