# ArchEHR-RAG

## Overview
This project focuses on automatically generating responses to patients' health-related questions using evidence from their clinical notes. The dataset comprises realistic patient inquiries submitted through a patient portal and their corresponding clinician-rewritten versions, designed to aid in formulating accurate responses. The primary objective is to construct well-grounded, coherent answers that leverage the provided clinical note excerpts to ensure reliable and contextually appropriate information. The pipeline consists of data preprocessing, embedding storage, retrieval, and response generation using a large language model (LLM).

## Project Structure

```
ArchEHR-RAG/
│── data_processing/   
│   └── preprocess.ipynb  # Converts dataset into a pandas DataFrame for accessibility
│
│── experiments/
│   └── model_experiments.ipynb  # Tests embedding models and retrievers
│
│── main.py  # Main execution script
│── config.py  # Configuration file containing model settings
│── utils.py  # Utility functions for embedding models and LLM initialization
│── vector_db.py  # Manages vector storage using ChromaDB
│── retriever.py  # Implements different retrieval mechanisms
│── response_generator.py  # Handles LLM-based response synthesis
│── requirements.txt  # Dependencies
```

## Technologies and Models Used
- **Embedding Models**: `mpnet`, `minilm`, `distilroberta`, `MedEmbed`, `BAAI_bge`, `MiniLM`, `GTE_base` (defined in `config.py`)
- **Retrievers**: `Base`, `AutoMerging`, `BM25`
- **Database**: `ChromaDB`
- **LLM**: `ahmgam/medllama3-v20` (Ollama model)
- **Libraries**: `llama-index`, `langchain`, `langgraph`, `chromadb`, `fastembed`

## Input Format (Example)
Input query format for the system:
```json
{
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

```

## How to Run the Project
### 1. Clone the repository:
```sh
git clone https://github.com/chiru-13/ArchEHR-RAG.git
```
```sh
cd ArchEHR-RAG
```

### 2. Create a Virtual Environment
- For Ubuntu
    ```sh
    python -m venv venv
    source venv/bin/activate
    ```
- For Windows
    ```sh
    python -m venv venv
    venv\Scripts\activate
    ```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Download Ollama Model
#### Ensure that `ollama` is installed. If not run the below commands in the terminal:
- For Ubuntu
    ```sh
    curl https://ollama.ai/install.sh | sh
    ```
    ```sh
    ollama serve &
    ```
    
- For Windows
    * Download the Ollama desktop app from [https://ollama.com/download/windows](https://ollama.com/download/windows)

#### Then pull the required model:
```sh
ollama pull ahmgam/medllama3-v20
```

### 5. Run the Main Script
```sh
python main.py
```

## Workflow Visualization
### The workflow of the project can be visualized as follows:

<p align='center'>
<img src="workflow.png" width="400">
</p>

## File Descriptions

### `main.py`
The main entry point of the project, orchestrating data loading, retrieval, and response generation. It initializes components such as the retriever, embedding models, vector database, and LLM, then processes user queries to generate responses.

### `config.py`
Contains configuration settings, including the selected LLM model (`ahmgam/medllama3-v20`), prompt templates, and embedding models used in the project.

### `response_generator.py`
Defines functions for generating responses using the selected LLM. It creates a response synthesizer using `llama_index` and constructs a query engine by integrating a retriever and a response synthesizer.

### `retriever.py`
Implements different retrieval mechanisms to fetch relevant information:
- **Base retriever**: Retrieves documents based on vector similarity.
- **AutoMerging retriever**: Enhances retrieval by merging relevant document chunks.
- **BM25 retriever**: Uses a traditional term-based ranking algorithm for retrieval.

### `utils.py`
Contains utility functions:
- **Embedding model loader**: Loads embedding models based on the configuration.
- **Document and node creation**: Converts input text into structured nodes for indexing.
- **LLM initialization**: Loads and configures the LLM model (Ollama) for response generation.

### `vector_db.py`
Handles vector storage using ChromaDB. It initializes a persistent ChromaDB client, creates vector indexes for documents, and manages retrieval operations.

### `data_processing/preprocess.ipynb`
Prepares the dataset by converting it into a Pandas DataFrame for easy accessibility and readability before embedding and retrieval.

### `experiments/model_experiments.ipynb`
Used to test different embedding models and retrievers before finalizing the workflow. This notebook ensures the best-performing configurations are selected.


## Citation for ArchEHR-QA:
Soni, S., & Demner-Fushman, D. (2025). ArchEHR-QA: BioNLP at ACL 2025 Shared Task on Grounded Electronic Health Record Question Answering (version 1.1). PhysioNet. https://doi.org/10.13026/f9xr-rr81.

## Citation for PhysioNet:
Goldberger, A. L., Amaral, L. A. N., Glass, L., Hausdorff, J. M., Ivanov, P. C., Mark, R. G., Mietus, J. E., Moody, G. B., Peng, C.-K., & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215.
