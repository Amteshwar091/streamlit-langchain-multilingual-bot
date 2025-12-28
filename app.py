import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot With OPENAI"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond to the user queries in {language}."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, api_key, model_name, temperature, max_tokens, language):
    try:
        llm = ChatOpenAI(
            model=model_name,
            openai_api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        
        answer = chain.invoke({'question': question, 'language': language})
        return answer
        
    except Exception as e:
        return f"Error: {str(e)}"

st.title("🤖 Q&A Chatbot With OpenAI")
st.markdown("Ask any question and adjust the model settings in the sidebar.")

st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

engine = st.sidebar.selectbox(
    "Select OpenAI model",
    ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
)

language = st.sidebar.selectbox(
    "Response Language",
    ["English", "French", "Hindi", "Punjabi", "Spanish", "German", "Japanese", "Tamil"]
)

temperature = st.sidebar.slider(
    "Temperature (Creativity)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.7
)
max_tokens = st.sidebar.slider(
    "Max Tokens (Response Length)", 
    min_value=50, 
    max_value=500, 
    value=150
)

user_input = st.text_input("Enter your question here:")

if st.button("Generate Response"):
    if not user_input:
        st.warning("Please enter a question.")
    elif not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar.")
    else:
        with st.spinner(f"Thinking in {language}..."):
           
            response = generate_response(user_input, api_key, engine, temperature, max_tokens, language)
            
            if response.startswith("Error:"):
                st.error(response)
            else:
                st.success("Response generated!")
                st.write(response)