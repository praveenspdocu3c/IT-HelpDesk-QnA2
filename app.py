import sys
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_community.document_transformers import LongContextReorder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate

# Azure OpenAI settings
api_key = "783973291a7c4a74a1120133309860c0"  # Replace with your key
azure_endpoint = "https://theswedes.openai.azure.com/"
api_version = "2024-02-01"

# Streamlit UI
st.title("IT HelpDesk QnA")

# File uploader
uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file:
    # Load PDF content
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    loader = PyMuPDFLoader("temp.pdf")
    data = loader.load()

    # Split the loaded text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    texts = text_splitter.split_documents(data)

    # Create embeddings model
    embeddings_model = AzureOpenAIEmbeddings(
        model="text-embedding-3-large",
        deployment="TextEmbeddingLarge",
        api_version=api_version,
        azure_endpoint=azure_endpoint,
        openai_api_key=api_key
    )

    # Create the retriever
    retriever = Chroma.from_documents(embedding=embeddings_model, documents=texts).as_retriever(
        search_kwargs={"k": 10}
    )

    # User input for problem statement
    user_problem = st.text_input("Enter the problem statement for resolution:", "")

    if st.button("Get Resolution"):
        query = f"What is the Resolution for the below mentioned Problem: {user_problem}"
        
        # Retrieve relevant documents
        docs = retriever.invoke(query)

        # Reorder documents by relevance
        reordering = LongContextReorder()
        reordered_docs = reordering.transform_documents(docs)

        # Prepare LLM model for response generation
        llm = AzureChatOpenAI(
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            openai_api_key=api_key,
            model="gpt-4o-mini",
            base_url=None,
            azure_deployment="GPT-4o-mini"
        )

        llm.validate_base_url = False

        prompt_template = """
        Given these texts:
        -----
        {context}
        -----
        Please answer the following question:
        {query}
        """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "query"])

        # Create and invoke the chain:
        chain = create_stuff_documents_chain(llm, prompt)
        response = chain.invoke({"context": reordered_docs, "query": query})
        
        # Display response
        st.markdown("### Response")
        st.write(response)
