import os
import uuid  # Use uuid for generating IDs if not provided
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Import the real embedding function
from llm_utils import generate_embeddings

# Load environment variables
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# --- Configuration ---
# Dimension for text-embedding-004 is 768
EMBEDDING_DIMENSION = 768
COLLECTIONS = {
    "episodic_memory": VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    "semantic_memory": VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    "tool_memory": VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
    "goal_memory": VectorParams(size=EMBEDDING_DIMENSION, distance=Distance.COSINE),
}

# --- Client Initialization ---


def get_qdrant_client() -> QdrantClient:
    """Initializes and returns the Qdrant client."""
    if QDRANT_API_KEY:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    else:
        # Assumes local instance without auth
        client = QdrantClient(url=QDRANT_URL)
    print("Qdrant client initialized.")
    return client


def initialize_qdrant_collections(client: QdrantClient):
    """Creates Qdrant collections if they don't exist."""
    existing_collections = [col.name for col in client.get_collections().collections]
    print(f"Existing collections: {existing_collections}")
    for name, vector_params in COLLECTIONS.items():
        if name not in existing_collections:
            print(f"Creating collection: {name}")
            client.create_collection(
                collection_name=name,
                vectors_config=vector_params,
            )
        else:
            print(f"Collection '{name}' already exists.")


# --- Memory Operations ---


def add_memory(
    client: QdrantClient,
    collection_name: str,
    documents: List[Dict[str, Any]],
    texts: List[str],
):
    """Adds documents with their embeddings to a Qdrant collection."""
    if collection_name not in COLLECTIONS:
        print(f"Error: Collection '{collection_name}' is not defined.")
        return

    if not documents or not texts or len(documents) != len(texts):
        print("Error: Mismatch between documents and texts or empty lists.")
        return

    # Generate embeddings using the imported function
    # Use RETRIEVAL_DOCUMENT task type for storing memories
    embeddings = generate_embeddings(texts, task_type="RETRIEVAL_DOCUMENT")

    if embeddings is None or len(embeddings) != len(documents):
        print(
            f"Error: Failed to generate embeddings or mismatch in count for collection '{collection_name}'."
        )
        return

    points = []
    for i, doc in enumerate(documents):
        # Ensure documents have unique IDs or generate them
        doc_id = doc.get("id", str(uuid.uuid4()))
        points.append(
            models.PointStruct(
                id=doc_id,
                vector=embeddings[i],
                payload=doc,  # Store the original document content
            )
        )

    print(f"Adding {len(points)} points to collection '{collection_name}'...")
    try:
        # Use upsert for idempotency (add or update)
        response = client.upsert(
            collection_name=collection_name, points=points, wait=True
        )
        print(f"Upsert response: {response}")
    except Exception as e:
        print(f"Error adding points to Qdrant collection '{collection_name}': {e}")


def search_memory(
    client: QdrantClient, collection_name: str, query_text: str, limit: int = 5
) -> List[Dict[str, Any]]:
    """Searches a Qdrant collection based on query text."""
    if collection_name not in COLLECTIONS:
        print(f"Error: Collection '{collection_name}' is not defined.")
        return []

    # Generate embedding for the query text
    # Use RETRIEVAL_QUERY task type for searching
    query_embedding_list = generate_embeddings(
        [query_text], task_type="RETRIEVAL_QUERY"
    )

    if query_embedding_list is None or not query_embedding_list:
        print(f"Error: Failed to generate embedding for query: '{query_text}'")
        return []

    query_vector = query_embedding_list[0]

    print(f"Searching collection '{collection_name}' for: '{query_text}'")
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
            with_payload=True,  # Include the stored payload in results
        )
        print(f"Found {len(search_result)} results in '{collection_name}'.")
        # Convert ScoredPoint objects to dictionaries
        results = [
            {"id": hit.id, "score": hit.score, "payload": hit.payload}
            for hit in search_result
        ]
        return results
    except Exception as e:
        print(f"Error searching Qdrant collection '{collection_name}': {e}")
        return []


# --- Example Usage (Optional) ---
if __name__ == "__main__":
    qdrant_client = get_qdrant_client()
    initialize_qdrant_collections(qdrant_client)

    # Example: Add a dummy memory using real embeddings
    print("\n--- Testing Add Memory ---")
    dummy_doc = {
        "id": "ep1",
        "content": "User asked about the weather.",
        "timestamp": "2024-01-01T10:00:00Z",
        "type": "user_interaction",
    }
    dummy_text = "User asked about the weather."
    add_memory(qdrant_client, "episodic_memory", [dummy_doc], [dummy_text])

    dummy_doc_2 = {
        "id": "sem1",
        "insight": "Weather questions often require accessing an external API.",
        "source_action": "reflection",
    }
    dummy_text_2 = "Insight: Weather questions often require accessing an external API."
    add_memory(qdrant_client, "semantic_memory", [dummy_doc_2], [dummy_text_2])

    # Example: Search memory using real embeddings
    print("\n--- Testing Search Memory ---")
    search_results_ep = search_memory(
        qdrant_client, "episodic_memory", "questions about weather"
    )
    print("\nEpisodic Search Results ('questions about weather'):")
    for result in search_results_ep:
        print(result)

    search_results_sem = search_memory(
        qdrant_client, "semantic_memory", "how to handle weather queries"
    )
    print("\nSemantic Search Results ('how to handle weather queries'):")
    for result in search_results_sem:
        print(result)
