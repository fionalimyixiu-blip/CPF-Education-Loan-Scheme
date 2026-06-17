import os
import time
from openai import OpenAI
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# 1. Initialize OpenAI Client
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = OpenAI(api_key=OPENAI_API_KEY)

# 2. Load and Cache the FAQ Excel file (keeps the app fast!)
@st.cache_data
def load_faq_data():
    return pd.read_excel('EDN FAQ_Trial _AIBC.xlsx')

df = load_faq_data()

# 3. Helper functions for RAG
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

# Cache embeddings generation so it only happens once at startup
@st.cache_data
def precompute_embeddings(_df):
    _df['embedding'] = _df['Question'].apply(get_embedding)
    return _df

df = precompute_embeddings(df)

def find_relevant_faqs(user_question, df, top_k=3):
    question_embedding = get_embedding(user_question)
    faq_embeddings = np.array(df['embedding'].tolist())

    similarities = cosine_similarity([question_embedding], faq_embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    relevant_faqs = df.iloc[top_indices][['Category', 'Question', 'Answers']].copy()
    relevant_faqs['similarity_score'] = similarities[top_indices]
    return relevant_faqs

def generate_response(user_question, relevant_faqs):
    # Build context from retrieved FAQs
    context = ""
    for _, row in relevant_faqs.iterrows():
        context += f"Q: {row['Question']}\nA: {row['Answers']}\n\n"

    # Enhanced, highly consultative prompt template
    prompt_text = f"""You are an expert personal advisor for the CPF Education Loan Scheme. 
Your goal is to guide students and parents smoothly through their options with clear, structured, and actionable advice.

Instructions:
1. Base your answer strictly on the FAQ information provided below.
2. Maintain a friendly, supportive, reassuring, and highly professional tone.
3. Structure your response cleanly using bullet points or steps if explaining a timeline, eligibility rule, or process.
4. If the exact answer is missing from the provided FAQs, gently let the user know you don't have that specific data on hand, and provide practical next steps by asking them to contact the CPF Board at 1800-227-1188 or visit cpf.gov.sg for official case-by-case evaluation.

FAQ Information:
{context}

User Question: {user_question}

Please provide an advisory, structured, and warm response:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a professional, empathetic, and clear advisor specializing in the CPF Education Loan Scheme."},
            {"role": "user", "content": prompt_text}
        ],
        temperature=0.4
    )
    return response.choices[0].message.content

def cpf_faq_chatbot(user_question):
    relevant_faqs = find_relevant_faqs(user_question, df, top_k=3)
    response = generate_response(user_question, relevant_faqs)
    return response

# 4. Streamlit User Interface Setup
st.set_page_config(page_title="CPF Education Loan Assistant", page_icon="🎓")
st.title("🎓 CPF Education Loan Scheme Advisor")
st.caption("Welcome! Ask me anything about application criteria, repayment timelines, or withdrawal limits.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'm here to help guide you through the CPF Education Loan Scheme. Whether you're planning your studies or figuring out repayment details, let me know what questions you have!"}
    ]

# Display existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept live user input
if prompt := st.chat_input("Ask a question (e.g., When does repayment start?, can I use my sibling's CPF?)"):
    # Add user message to history and show it
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response dynamically
    with st.chat_message("assistant"):
        with st.spinner("Reviewing guidelines..."):
            ai_response = cpf_faq_chatbot(prompt)
        st.markdown(ai_response)
        
    # Store assistant response in history
    st.session_state.messages.append({"role": "assistant", "content": ai_response})
