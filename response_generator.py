from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core import get_response_synthesizer
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import PromptTemplate
from config import summary_prompt


def create_response_synthesizer(llm):
    """Create a response synthesizer based on the given LLaMA model and response type.
    
    Args:
    - llm: The llm model to be used for response synthesis.

    Returns:
    - A response synthesizer object.
    """
    summary_tmpl = PromptTemplate(summary_prompt)

    response_synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.REFINE, llm=llm, text_qa_template=summary_tmpl, simple_template=summary_tmpl, refine_template=summary_tmpl
    )
    return response_synthesizer

def build_query_engine(retriever, response_synthesizer):
    """Build a query engine based on the given retriever and response synthesizer.
    
    Args:
    - retriever: The retriever instance to be used for query execution.
    - response_synthesizer: The response synthesizer instance to be used for response synthesis.

    Returns:
    - A query engine object.
    """
    query_engine = RetrieverQueryEngine(retriever = retriever, response_synthesizer=response_synthesizer)
    return query_engine


