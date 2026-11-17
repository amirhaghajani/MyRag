from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.chat_engine.types import BaseChatEngine, ChatMode
from llama_index.llms.openai_like import OpenAILike
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))   # explicit path: works from any CWD
_api_key = os.getenv("OPENROUTER_API_KEY")

# --- 1. Which models to use (do this BEFORE building the index) -------------
# Without this, LlamaIndex defaults to OpenAI and throws
# "Could not load OpenAI model ... please check your OPENAI_API_KEY".
# Settings.llm = Ollama(
#     model="llama3.2:3b", 
#     request_timeout=120.0,
#     base_url="http://localhost:11434"
#     )
Settings.llm = OpenAILike(
    model="baidu/ernie-5.5-flash-preview:free",
    api_base="https://openrouter.ai/api/v1",
    api_key=_api_key,
    is_chat_model=True,
    is_function_calling_model=False,
    context_window=32768,
    timeout=120,
    max_retries=2,
)

Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# --- 2. Read the files -----------------------------------------------------
docs = SimpleDirectoryReader(
    #input_dir="Samples",          # path relative to where you run the script
    #required_exts=[".txt"],       # ignore anything that is not .txt
    input_files=["Samples\\01.txt",]
).load_data()

print(f"Loaded {len(docs)} documents from disk.")

# --- 3. Build the index ----------------------------------------------------
# Documents -> nodes (chunks) -> embeddings -> in-memory vector store.
# Nothing is written to disk; the index lives only while the script runs.
index = VectorStoreIndex.from_documents(docs)


# --- 4. Turn the index into a chat engine ---------------------------------
# Default chat_mode is "best", which in 0.14.25 = CondensePlusContextChatEngine:
# it rewrites follow-up questions using the chat history, then retrieves context.
# 4. Chat engine
chat_engine = index.as_chat_engine(
    chat_mode= ChatMode.CONTEXT,
    system_prompt=(
        "You are a helpful assistant. Answer questions about the documents "
        "in the knowledge base, and chat normally about anything else."
    ),
)

# 5. Loop until the user types exit
EXIT_WORDS = {"exit", "quit", "bye", "q"}
print('Chat with the documents. Type "exit" to quit.\n')

while True:
    try:
        message = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break

    if not message:
        continue

    if message.lower() in EXIT_WORDS:
        print("Bye!")
        break

    response = chat_engine.chat(message)
    print(f"Bot: {response}\n")

