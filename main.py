from config import *
from utils import *
from response_generator import *
from retriever import *
from vector_db import *
from typing import TypedDict, Dict, Any, List
from llama_index.core import VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from langgraph.graph import StateGraph, START, END
import json
import colorama
from colorama import Fore, Style
from rich.console import Console
from rich.panel import Panel
from rich.json import JSON
from rich import print as rich_print


# Initialize colorama for cross-platform color support
colorama.init()

# Initialize Rich console
console = Console()

ChromaDB_DIR = "chromadb"

def print_step_header(step_name: str, step_number: int):
    """Print a formatted header for each step"""
    console.print(
        Panel(
            f"[bold white]STEP {step_number}: {step_name}[/bold white]",
            border_style="blue",
            expand=False,
            padding=(1, 2)
        )
    )

def print_agent_output(agent_name: str, output: Dict[str, Any], success: bool = True):
    """Print agent output in a structured format"""
    status_color = "green" if success else "red"
    status_text = "SUCCESS" if success else "FAILURE"
    
    console.print(
        Panel(
            f"[bold white]AGENT: {agent_name} - [bold {status_color}]{status_text}[/bold {status_color}][/bold white]",
            border_style=status_color,
            expand=False,
            padding=(1, 2)
        )
    )
    
    # Print the output as formatted JSON
    if output:
        console.print(JSON(json.dumps(output, indent=2)))

# Define Graph State
class QueryState(TypedDict):
    input: Dict[str, Any] | str
    docs: List
    nodes: List
    error: bool
    index: Any
    retriever_type: str
    retriever: Any
    response: str
    note_texts: str

# Initialize Chroma Client, Embedding Model and LLM
console.print("[bold blue]Initializing Chroma Client, Embedding Model and LLM...[/bold blue]")
chroma_client = initialize_chroma_client(ChromaDB_DIR)
embed_model = load_embed_model("BAAI_bge")
llm = initialise_llm(llm_model=llm_model)
console.print("[green]Initialization Complete![/green]")

def load_documents(state):
    # Simulating document loading (replace with actual source)
    try:
        print_step_header("Document Loader", 1)
        note_excerpts = state['input']['note_excerpts']
        docs, nodes = create_docs_n_nodes(note_excerpts)
        index = create_index(chroma_client, docs, embed_model)


        state['docs'] = docs
        state['nodes'] = nodes
        state['index'] = index
        print_agent_output("Document Loader", {"Loaded Docs": len(docs), "Nodes Created": len(nodes), "Index Created": "Sucessfully"})
        return state
    except Exception as e:
        print_agent_output("Document Loader", {"error": str(e)}, False)
        print(f"Error loading documents: {e}")
        state['error'] = True
        return state

def retrieve(state):
    try:
        print_step_header("Retriever", 2)
        index = state["index"]
        nodes = state["nodes"]
        note_excerpt = state['input']['note_excerpts'] 
        retriever_type = state['retriever_type']
        retriever = build_retriever(index, nodes, retriever_type, top_k =max(1, (len(note_excerpt)*2) // 3))

        print_agent_output("Retriever", {"status": "Sucessfully created retriever", "retriever_type": retriever_type})
        state['retriever'] = retriever
        return state
    except Exception as e:
        print_agent_output("Retriever", {"error": str(e)}, False)
        print(f"Error retrieving: {e}")
        state['error'] = True
        return state
    
def generate_response(state):
    try:
        print_step_header("Response Generator", 3)
        retriever = state["retriever"]
        response_synthesizer = create_response_synthesizer(llm)
        query_engine = build_query_engine(retriever, response_synthesizer)
        patient_question_dict = state['input']['patient_question']
        clinical_question_text = state['input']['clinical_question']
        combined_query_text = " ".join(patient_question_dict.values()) + " " + clinical_question_text
        response = query_engine.query(combined_query_text)

        note_texts = []
        nodes = []
        for node in response.source_nodes:
            note_id = node.node.metadata.get("key", "Unknown")
            note_text = node.node.text
            note_texts.append(f"({note_id}): {note_text}")
            nodes.append(note_id)
        
        print_agent_output("Response Generator", {"status": f"Response generated, retrieved nodes: {nodes}"})
    
        state['response'] = response.response
        state['note_texts'] = "\n".join(note_texts)
        return state
    except Exception as e:
        state['error'] = True
        print(f"Error generating response: {e}")
        return state

# Define the Graph
workflow = StateGraph(QueryState)
workflow.add_node("Document Loader", load_documents)
workflow.add_node("Retriever", retrieve)
workflow.add_node("Response Generator", generate_response)

workflow.add_conditional_edges(
    "Document Loader",
    lambda state: "Retriever" if not state['error'] else END
)
workflow.add_conditional_edges(
    "Retriever",
    lambda state: "Response Generator" if not state['error'] else END
)

workflow.add_edge(START, "Document Loader")
workflow.add_edge("Response Generator", END)

console.print("[bold blue]Compiling Workflow...[/bold blue]")
ehr_workflow = workflow.compile()
console.print("[green]Workflow Compilation Complete![/green]")

def process_query(input) -> Dict[str, Any]:
    """
    Process a query through the entire graph workflow
    
    Args:
        input: Input dictionary containing the query and any other relevant information
    
    Returns:
        Dict containing final query results and workflow details
    """
    initial_state = {
        "input": input,
        "docs": [],
        "nodes": [],
        "index": "",
        "retriever_type": "base",
        "error": False,
        "retriever": "",
        "response": "",
        "note_texts": ""
    }

    result = ehr_workflow.invoke(initial_state)
    return result

# Main Execution
if __name__ == "__main__":
    console.print("[bold yellow] Starting EHR Workflow...[/bold yellow]")
    user_input = {
      "note_excerpts": {
            "0": "Medical Assessment:",
            "1": "The patient has stage 2 hypertension with elevated blood pressure readings.", 
            "2": "Given intolerance to amlodipine, alternative treatment options were discussed.", 
            "3": "Lifestyle modifications such as reduced sodium intake and increased physical activity were recommended.",
            "4": "A follow-up in two weeks was scheduled to reassess blood pressure response to the new treatment.",
        },
      "patient_question": {
            "0": "What are some antihypertensive medications that don’t cause leg swelling, and how do they work differently?",
            "1": "If I improve my diet and exercise, how much can I expect my blood pressure to drop without medication?"
        },
      "clinical_question": "What are the potential alternative medications for this patient’s hypertension, considering his history of leg swelling with amlodipine, and how do they compare in terms of efficacy and side effects?",
    }
    response = process_query(user_input)
    if not response['error']:
      print("Response:", response['response'] + "\n" + response['note_texts'])
      console.print("[bold green]Workflow Execution Complete![/bold green]")
    else:
      console.print("[bold red]Workflow Stopped!![/bold red]")
