import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore

def initialize_chroma_client(chroma_db_path):
    """Initialize a Chroma client with the given database path.
    Args:
        chroma_db_path: The directory path where ChromaDB client gets initialised.
    
    Returns:
        Initialise ChromaDB in the specified path.
        """
    chroma_client = chromadb.PersistentClient(path=chroma_db_path)
    return chroma_client


def create_index(chroma_client, docs, embed_model, collection_name="note_excerpts"):
    """
    Creates a new index with the given documents in ChromaDB. If the collection already exists, it deletes it first.
    
    Args:
        chroma_client: The ChromaDB client instance.
        docs: The documents to be indexed.
        embed_model: The embedding model to use.
        collection_name: The name of the collection (default is "note_excerpts").
    
    Returns:
        The created VectorStoreIndex instance.
    """
    # Delete existing collection if it exists
    try:
        existing_collections = chroma_client.list_collections()

        if "note_excerpts" in existing_collections:
            chroma_client.delete_collection("note_excerpts")
    
    except Exception as e:
        print(f"Warning: Could not delete collection {collection_name}: {e}")
    
    # Create a new collection
    chroma_collection = chroma_client.get_or_create_collection(collection_name)
    
    # Setup vector store and storage context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    # Create and return the index
    return VectorStoreIndex.from_documents(docs, storage_context=storage_context, embed_model=embed_model)
