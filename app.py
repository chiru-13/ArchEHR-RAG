import streamlit as st
import os
import warnings
from main import (
    process_query,
    create_docs_n_nodes,
    create_index,
    load_embed_model,
    initialize_chroma_client
)

# Fix PyTorch/Streamlit watcher issue
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
warnings.filterwarnings("ignore", message=".*__path__.*")

st.set_page_config(page_title="MedicalBot Assistant", layout="wide")
st.title("🩺 MedicalBot: Clinical Decision Assistant")

# Initialize session state
if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.nodes = None
    st.session_state.note_excerpts = None
    st.session_state.chroma_client = initialize_chroma_client("chromadb")
    st.session_state.embed_model = load_embed_model("BAAI_bge")
    st.session_state.chat_history = []
    st.session_state.patient_narrative = ""

# ---------- Sidebar: Note Entry ----------
with st.sidebar:
    st.header("📋 Patient Notes Setup")

    note_input = st.text_area("Paste EHR Note Excerpts (one per line)", height=250)
    patient_narr = st.text_area("🧠 Patient Narrative", placeholder="E.g., I’ve been having headaches and trouble sleeping...")

    if st.button("📚 Load Notes"):
        try:
            note_excerpts = {
                str(i): val.strip()
                for i, val in enumerate(note_input.strip().splitlines())
                if val.strip()
            }
            docs, nodes = create_docs_n_nodes(note_excerpts)
            index = create_index(st.session_state.chroma_client, docs, st.session_state.embed_model)

            st.session_state.index = index
            st.session_state.nodes = nodes
            st.session_state.note_excerpts = note_excerpts
            st.session_state.patient_narrative = patient_narr.strip()

            st.success("✅ Notes indexed and patient narrative saved.")
        except Exception as e:
            st.error(f"❌ Failed to load notes: {e}")

    if st.button("🔄 Reset All"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ---------- Chat Section ----------
if st.session_state.index and st.session_state.patient_narrative:
    st.subheader("💬 Ask Your Questions")

    # Display chat history
    for sender, message in st.session_state.chat_history:
        st.chat_message(sender).markdown(message)

    # Two-question form
    with st.form("query_form"):
        patient_q_input = st.text_area("🧑‍⚕️ Patient Question", placeholder="e.g., What causes leg swelling?")
        clinical_q_input = st.text_area("👨‍🔬 Clinical Question", placeholder="e.g., Alternatives to amlodipine?")
        submitted = st.form_submit_button("🧠 Get Response")

    if submitted:
        if not patient_q_input.strip() and not clinical_q_input.strip():
            st.warning("Please enter at least one question.")
        else:
            with st.spinner("Evaluating relevance and generating response..."):
                input_payload = {
                    "note_excerpts": st.session_state.note_excerpts,
                    "patient_question": {"0": patient_q_input.strip()},
                    "clinical_question": clinical_q_input.strip(),
                    "patient_narrative": st.session_state.patient_narrative
                }

                response = process_query(input_payload)

                if not response["error"] and response.get("response"):
                    answer = response["response"]
                    sources = response["note_texts"]

                    st.chat_message("user").markdown(
                        f"**Patient Q:** {patient_q_input}\n**Clinical Q:** {clinical_q_input}"
                    )
                    st.chat_message("assistant").markdown(
                        answer + "\n\n📄 **Supporting Notes**:\n" + f"```\n{sources}\n```"
                    )

                    # Update history
                    st.session_state.chat_history.append((
                        "user", f"**Patient Q:** {patient_q_input}\n**Clinical Q:** {clinical_q_input}"
                    ))
                    st.session_state.chat_history.append((
                        "assistant", answer + "\n\n📄 **Supporting Notes**:\n" + f"```\n{sources}\n```"
                    ))
                else:
                    st.error("🚫 Input deemed irrelevant or an error occurred.")
else:
    st.info("⬅️ Please load patient notes and narrative from the sidebar to begin.")
