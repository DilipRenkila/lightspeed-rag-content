"""Utility script to generate embeddings."""

import argparse
import json
import os
import time
from typing import Callable, Dict, List
from multiprocessing import Pool, cpu_count

import faiss
import requests
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.llms.utils import resolve_llm
from llama_index.readers.file.flat.base import FlatReader
from llama_index.core.schema import TextNode
from llama_index.core.storage.storage_context import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore

OCP_DOCS_ROOT_URL = "https://docs.openshift.com/container-platform/"
OCP_DOCS_VERSION = "4.15"
UNREACHABLE_DOCS: int = 0
RUNBOOKS_ROOT_URL = "https://github.com/openshift/runbooks/blob/master/alerts"

def ping_url(url: str) -> bool:
    """Check if the URL parameter is live."""
    try:
        response = requests.get(url, timeout=30)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def get_file_title(file_path: str) -> str:
    """Extract title from the plaintext doc file."""
    title = ""
    try:
        with open(file_path, "r") as file:
            title = file.readline().rstrip("\n").lstrip("# ")
    except Exception:  # noqa: S110
        pass
    return title

def file_metadata_func(file_path: str, docs_url_func: Callable[[str], str]) -> Dict:
    """Populate the docs_url and title metadata elements with docs URL and the page's title."""
    docs_url = docs_url_func(file_path)
    title = get_file_title(file_path)
    msg = f"file_path: {file_path}, title: {title}, docs_url: {docs_url}"
    if not ping_url(docs_url):
        global UNREACHABLE_DOCS
        UNREACHABLE_DOCS += 1
        msg += ", UNREACHABLE"
    print(msg)
    return {"docs_url": docs_url, "title": title}

def ocp_file_metadata_func(file_path: str) -> Dict:
    """Populate metadata for an OCP docs page."""
    docs_url = lambda file_path: (
        OCP_DOCS_ROOT_URL
        + OCP_DOCS_VERSION
        + file_path.removeprefix(EMBEDDINGS_ROOT_DIR).removesuffix("txt")
        + "html"
    )
    return file_metadata_func(file_path, docs_url)

def runbook_file_metadata_func(file_path: str) -> Dict:
    """Populate metadata for a runbook page."""
    docs_url = lambda file_path: (
        RUNBOOKS_ROOT_URL + file_path.removeprefix(RUNBOOKS_ROOT_DIR)
    )
    return file_metadata_func(file_path, docs_url)

def got_whitespace(text: str) -> bool:
    """Indicate if the parameter string contains whitespace."""
    return any(c.isspace() for c in text)

def process_batch(batch: List[Dict]) -> List[TextNode]:
    """Process a batch of documents."""
    nodes = Settings.text_splitter.get_nodes_from_documents(batch)
    return [node for node in nodes if isinstance(node, TextNode) and got_whitespace(node.text)]

if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="embedding cli for task execution")
    parser.add_argument("-f", "--folder", help="Plain text folder path")
    parser.add_argument("-r", "--runbooks", help="Runbooks folder path")
    parser.add_argument("-md", "--model-dir", default="embeddings_model", help="Directory containing the embedding model")
    parser.add_argument("-mn", "--model-name", help="HF repo id of the embedding model")
    parser.add_argument("-c", "--chunk", type=int, default=380, help="Chunk size for embedding")
    parser.add_argument("-l", "--overlap", type=int, default=0, help="Chunk overlap for embedding")
    parser.add_argument("-em", "--exclude-metadata", nargs="+", default=None, help="Metadata to be excluded during embedding")
    parser.add_argument("-o", "--output", help="Vector DB output folder")
    parser.add_argument("-i", "--index", help="Product index")
    parser.add_argument("-v", "--ocp-version", help="OCP version")
    args = parser.parse_args()
    print(f"Arguments used: {args}")

    PERSIST_FOLDER = os.path.normpath("/" + args.output).lstrip("/")
    if PERSIST_FOLDER == "":
        PERSIST_FOLDER = "."

    EMBEDDINGS_ROOT_DIR = os.path.abspath(args.folder)
    if EMBEDDINGS_ROOT_DIR.endswith("/"):
        EMBEDDINGS_ROOT_DIR = EMBEDDINGS_ROOT_DIR[:-1]
    RUNBOOKS_ROOT_DIR = os.path.abspath(args.runbooks)
    if RUNBOOKS_ROOT_DIR.endswith("/"):
        RUNBOOKS_ROOT_DIR = RUNBOOKS_ROOT_DIR[:-1]

    OCP_DOCS_VERSION = args.ocp_version

    os.environ["HF_HOME"] = args.model_dir
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    Settings.chunk_size = args.chunk
    Settings.chunk_overlap = args.overlap
    Settings.embed_model = HuggingFaceEmbedding(model_name=args.model_dir)
    Settings.llm = resolve_llm(None)

    embedding_dimension = len(Settings.embed_model.get_text_embedding("random text"))
    faiss_index = faiss.IndexFlatIP(embedding_dimension)
    vector_store = FaissVectorStore(faiss_index=faiss_index)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Load all documents
    all_documents = SimpleDirectoryReader(
        args.folder,
        recursive=True,
        file_metadata=ocp_file_metadata_func
    ).load_data()

    # Implement manual batching
    batch_size = 100  # Adjust this value based on your system's capabilities
    document_batches = [all_documents[i:i + batch_size] for i in range(0, len(all_documents), batch_size)]

    # Use multiprocessing to process batches
    with Pool(processes=cpu_count()) as pool:
        all_nodes = pool.map(process_batch, document_batches)

    # Flatten the list of nodes
    good_nodes = [node for batch in all_nodes for node in batch]

    runbook_documents = SimpleDirectoryReader(
        args.runbooks,
        recursive=True,
        required_exts=[".md"],
        file_extractor={".md": FlatReader()},
        file_metadata=runbook_file_metadata_func,
    ).load_data()
    runbook_nodes = Settings.text_splitter.get_nodes_from_documents(runbook_documents)

    good_nodes.extend(runbook_nodes)

    # Create & save Index
    index = VectorStoreIndex(
        good_nodes,
        storage_context=storage_context,
    )
    index.set_index_id(args.index)
    index.storage_context.persist(persist_dir=PERSIST_FOLDER)

    metadata: dict = {
        "execution-time": time.time() - start_time,
        "llm": "None",
        "embedding-model": args.model_name,
        "index-id": args.index,
        "vector-db": "faiss.IndexFlatIP",
        "embedding-dimension": embedding_dimension,
        "chunk": args.chunk,
        "overlap": args.overlap,
        "total-embedded-files": len(all_documents)
    }

    with open(os.path.join(PERSIST_FOLDER, "metadata.json"), "w") as file:
        json.dump(metadata, file)

    if UNREACHABLE_DOCS > 0:
        print(f"WARNING: There were documents with {UNREACHABLE_DOCS} unreachable URLs, "
              "grep the log for UNREACHABLE. Please update the plain text.")
