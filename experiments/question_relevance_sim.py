import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the embedding model
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Function to compute similarity scores
def compute_similarity(df):
    results = []

    for _, row in df.iterrows():
        # Extract patient question from dictionary
        patient_question = " ".join(row["patient_question"].values())  
        
        # Extract note sentences from dictionary
        note_sentences = list(row["note_excerpt_sentences"].values())

        # Encode patient question
        question_embedding = model.encode(patient_question, convert_to_tensor=True)

        # Compute similarity for each note sentence
        similarity_scores = [
            util.pytorch_cos_sim(question_embedding, model.encode(sent, convert_to_tensor=True)).item()
            for sent in note_sentences
        ]

        # Get similarity scores as lists for each category
        supplementary_sim = [similarity_scores[i] for i in row["supplementary"]] if row["supplementary"] else []
        relevant_sim = [similarity_scores[i] for i in row["essential"]] if row["essential"] else []
        not_relevant_sim = [similarity_scores[i] for i in row["not_relevant"]] if row["not_relevant"] else []

        results.append({
            "case_id": row["case_id"],
            "patient_question": patient_question,
            "supplementary_similarity": supplementary_sim,
            "relevant_similarity": relevant_sim,
            "not_relevant_similarity": not_relevant_sim
        })
    
    return pd.DataFrame(results)

# Example DataFrame with dictionary format
df = pd.DataFrame([
    {
        "case_id": 1,
        "patient_question": {"0": "What are some antihypertensive medications I can take?"},
        "note_excerpt_sentences": {
            "0": "Medical Assessment:",
            "1": "The patient has stage 2 hypertension.",
            "2": "Prescribed Losartan for treatment.",
            "3": "Alternative options include ACE inhibitors.",
            "4": "The patient has a history of allergic reactions to beta-blockers."
        },
        "supplementary": [3],
        "essential": [1, 2],
        "not_relevant": [0, 4]
    }
])

# Compute similarity scores
result_df = compute_similarity(df)

# Display results
print(result_df)