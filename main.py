from config import *
from utils import *
from response_generator import *
from retriever import *
from vector_db import *
from question_relevance import * 
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
    relevance: str
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

def relevance_node(state):
    try:
        print_step_header("Question Relevance", 1)
        patient_narr = state['input']['patient_narrative']
        patient_ques_dict = state['input']['patient_question']
        notes_dict = state['input']['note_excerpts']
        rel_response = check_question_relevance(patient_ques_dict, patient_narr, notes_dict)
        if rel_response.strip() == "Yes":
          state['relevance'] = rel_response
          print_agent_output("Question Relevance", {"message": f"The given inputs are relevant."})
        else:
          state['relevance'] = "No"
          print_agent_output("Question Relevance", {"message": f"The given inputs are not relevant. Please try again..."}, False)
        return state
    except Exception as e:
        print_agent_output("Question Relevance", {"error": str(e)}, False)
        state['relevance'] = "No"
        print(f"Error finding relevance: {str(e)}") 
        return state
    
def load_documents(state):
    try:
        print_step_header("Document Loader", 2)
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
        print_step_header("Retriever", 3)
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
        print_step_header("Response Generator", 4)
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
workflow.add_node("Question Relevance", relevance_node)
workflow.add_node("Document Loader", load_documents)
workflow.add_node("Retriever", retrieve)
workflow.add_node("Response Generator", generate_response)

workflow.add_conditional_edges(
    "Question Relevance",
    lambda state: "Document Loader" if state['relevance']=="Yes" else END
)

workflow.add_conditional_edges(
    "Document Loader",
    lambda state: "Retriever" if not state['error'] else END
)
workflow.add_conditional_edges(
    "Retriever",
    lambda state: "Response Generator" if not state['error'] else END
)

workflow.add_edge(START, "Question Relevance")
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
            "0": "Could you explain what causes ringing in the ears and how it can be treated?",
            "1": "Is it normal to experience mild hand tremors when feeling anxious or stressed?"
        },
      "clinical_question": "What are the potential alternative medications for this patient’s hypertension, considering his history of leg swelling with amlodipine, and how do they compare in terms of efficacy and side effects?",
      "patient_narrative": "Lately, I’ve been having a lot of trouble sleeping due to frequent nightmares. I also noticed some mild skin rashes after using a new laundry detergent. I haven’t had any major health issues recently, but I’m thinking about adopting a cat soon, so I wanted to make sure I’m not allergic. Additionally, I’ve been feeling a bit more anxious at work due to increased deadlines",
      }
    response = process_query(user_input)
    if not response['error'] and response['response']:
      print("Response:", response['response'] + "\n" + response['note_texts'])
      console.print("[bold green]Workflow Execution Complete![/bold green]")
    else:
      console.print("[bold red]Workflow Stopped!![/bold red]")
