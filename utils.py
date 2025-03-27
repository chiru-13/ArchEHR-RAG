from llama_index.core.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from config import embed_models
from llama_index.llms.ollama import Ollama
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.langchain import LangchainEmbedding

def load_embed_model(embed_model_name):
    """
    Embedding model function to be used in the index.
    """
    if embed_model_name in embed_models.keys():
        lc_embed_model = HuggingFaceEmbeddings(
            model_name=embed_models[embed_model_name]
        )
        return LangchainEmbedding(lc_embed_model)
    else:
        raise ValueError(f"Embedding model {embed_model_name} not found in embed models list. Please check and retry again.")
    
def create_docs_n_nodes(note_excerpts):
    """
    Create a list of documents and nodes from a dictionary of notes.
    
    Args:
    note_excerpts (dict): A dictionary where keys are node names and values are text notes.

    Returns:
    list: A list of documents and nodes.
    """
    docs = [Document(text=sentence, metadata={"key": key}) for key, sentence in note_excerpts.items()]
    node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=0)
    nodes = node_parser.get_nodes_from_documents(docs)
    return docs, nodes

def initialise_llm(llm_model):
    """"
    Initialises the LLM to be used for response generation."
    """
    return Ollama(model=llm_model, request_timeout=120.0)

