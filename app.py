from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

llm = ChatGroq(
    model='llama-3.3-70b-versatile'
)

template = PromptTemplate(
    template="""
    Explain {topic} in a {way} way.
    Keep the explanation {length}.
    """,
    input_variables=['topic', 'way', 'length']
)

st.header("AI Learner")
topic = st.text_input("Enter Topic: ", placeholder="Quantization in LLMs")
way = st.selectbox(
    "Select explanation style",
    ["easy", "technical", "5 year old", "professional"]
)
length = st.selectbox(
    "Enter the length of the answer e.g., short, medium, long etc.",
    ["very short","short", "long", "very long"]
)


if st.button('Generate'):

    if topic and way and length:

        chain = template | llm

        result = chain.invoke({
            'topic': topic,
            'way': way,
            'length': length
        })

        st.write(result.content)

    else:
        st.warning("Please fill all fields")


