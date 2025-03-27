import Stemmer
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import VectorIndexRetriever, AutoMergingRetriever


# Function to build a retriever for a specific case
def build_retriever(index, nodes, retriever_type, top_k=5):
    """
    Build a retriever for a specific retriever type.

    Args:
    - index (VectorStoreIndex): The VectorStoreIndex instance of the documents or nodes.
    - nodes (list): The list of nodes to use for the retriever.
    - retriever_type (str): The type of retriever to use -- base, bm25, or auto_merger.
    - top_k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
    - retriever (Retriever): The built retriever of the specific type.
    """
    if retriever_type == "base":
        return index.as_retriever(similarity_top_k=top_k)

    elif retriever_type == "auto_merger":
        base_retriever = index.as_retriever(similarity_top_k=top_k)
        return AutoMergingRetriever(base_retriever, storage_context=index.storage_context, verbose=True)

    elif retriever_type == "bm25":
        return BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=top_k,
            stemmer= Stemmer.Stemmer("english"),
            language="english",
        )

    else:
        raise ValueError("Invalid retriever_type. Choose from: 'base', 'auto_merger', 'bm25'.")


# Function to retrieve nodes for a specific case
def get_case_retrieved_nodes(retriever, question):
    """
    Get the retrieved nodes for a given question based on the retriever.

    Args:
    - retriever (Retriever): The retriever instance to use for the retrieval.
    - question (str): The question to use for the retrieval.

    Returns:
    - retrieved_nodes (list): The list of retrieved nodes.
    """
    results = retriever.retrieve(question)
    return results