# AI Learner Pro

AI Learner Pro is a Streamlit-based GenAI learning assistant that helps users understand any topic through personalized explanations, PDF-based context-aware answers, flashcards, and quizzes.

## Features

- Generate AI-powered explanations for any topic
- Choose explanation style and response length
- Upload PDFs and get context-aware answers using RAG
- Generate summarized flashcards
- Attempt quizzes based on the searched topic
- Maintain learning history and favorites
- Download generated study material

## Tech Stack

- Python
- Streamlit
- LangChain
- Groq LLM API
- FAISS
- FastEmbed
- PyPDFLoader
- Plotly
- PIL

## Installation

```bash
git clone <your-repo-url>
cd <your-project-folder>
pip install -r requirements.txt

Environment Variables
Create a .env file and add:

GROQ_API_KEY=your_groq_api_key
Run the App: streamlit run app.py
