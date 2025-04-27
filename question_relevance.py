from llama_index.llms.openrouter import OpenRouter
from config import *

# Initialize the LLM
llm = OpenRouter(
    api_key=OPENROUTER_API,  
    max_tokens=4096,
    context_window=4096,
    # model="openai/gpt-3.5-turbo",
    model="deepseek/deepseek-r1:free",
)

def check_question_relevance(patient_question_dict, patient_narr, notes_dict):
    # Get patient question
    ques_dict = patient_question_dict
    if ques_dict is None:
        return "Error: patient_question is None"
    ques_text = "\n".join(ques_dict.values()) if isinstance(ques_dict, dict) else str(ques_dict)

    # Get note excerpt
    if notes_dict is None:
        return "Error: note_excerpt_sentences is None"
    notes_text = "\n".join(notes_dict.values()) if isinstance(notes_dict, dict) else str(notes_dict)

    prompt = f"""
You are an expert clinical assistant.

# Task
Determine whether the **patient's question** is relevant to the **clinical notes** that is you should understand the clinical history using the clinical notes and then give response provided. Also take help of the patient narrative as well. 

# Types of questions can be asked:
  - It can be a follow up question.
  - Some conseqence or complications of the problem/issues faced.
  - It can be a treatment or medication queries.
  - It can be a diagnostic clarification or symptom concerns about the problem/issue faced
  - A question directly related with the clinical notes.

# Inputs
## Patient Narrative
{patient_narr}

## Patient Question
{ques_text}

## Clinical Notes
{notes_text}

# Output Format (Strictly follow the output format)
Respond with one of the following:
- Yes -- if they are relevant
- No -- if they are not relevant

# Instructions
Use clinical reasoning to assess whether the information in the clinical notes can help answer or provide insight into the patient's question.
Do not explain your answer. Just say "Yes" or "No". NOTHING ELSE.
    """
    try:
        response = llm.complete(prompt)
        if response is None or not hasattr(response, 'text'):
            return "No"
        return response.text.strip()
    except Exception as e:
        print(f"Error : {e}")
        return "Error: Exception occurred"