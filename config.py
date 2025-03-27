llm_model= "ahmgam/medllama3-v20"

summary_prompt = (
        "You are a clinical bot designed to answer queries based strictly on the provided context.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and **not prior knowledge**, provide a **concise and to-the-point answer** in a few sentences.\n"
        "**DO NOT include unnecessary details or elaborate explanations**—only state the key information needed to answer the question accurately.\n"
        "CRITICAL: While answering the question, **consider only the relevant context**. Ignore any irrelevant data within the context that does not pertain to the question.\n"
        "STRICTLY answer the query based on the given context and not on prior knowledge.\n"
        "Limit your response to a maximum of **two to three sentences**.\n"
        "Query: {query_str}\n"
        "Answer: "
    )


embed_models = {
    "mpnet": "sentence-transformers/all-mpnet-base-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "distilroberta": "sentence-transformers/all-distilroberta-v1",
    "MedEmbed": "abhinand/MedEmbed-base-v0.1",
    "BAAI_bge": "BAAI/bge-base-en-v1.5",
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "GTE_base": "thenlper/gte-base",
}