import streamlit as st
from langchain_community.document_loaders import WebBaseLoader

from model import Model
from vectordb import Database
from utils import clean_text

def create_streamlit_app(llm, portfolio, clean_text):
    # Basic UI
    st.markdown(
        """
            <h1 style='text-align: center; color: #4B8BBE;'>
                📧 Cold Mail Generator
            </h1>
        """, unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Generate professional cold emails from job URLs.</p>", unsafe_allow_html=True)

    url_input = st.text_input("Enter a URL:")
    submit_button = st.button("Submit")
    
    # Main Code
    if submit_button:
        try:
            loader = WebBaseLoader([url_input])
            data = clean_text(loader.load()[0].page_content)
            
            portfolio.load_portfolios()
            
            jobs = model.extract_jobs(data)
            for job in jobs:
                skills = job.get('skills', [])
                links = portfolio.query_search(skills)
                email = llm.write_email(job, links)
                st.code(email.strip(), language='markdown')
        except Exception as e:
            st.error(f"An Error Occurred: {e}")


if __name__ == "__main__":
    model = Model()
    portfolio = Database()
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    create_streamlit_app(model, portfolio, clean_text)
