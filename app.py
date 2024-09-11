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
import pandas as pd
from fpdf import FPDF

# Azure OpenAI settings
api_key = "783973291a7c4a74a1120133309860c0"  
azure_endpoint = "https://theswedes.openai.azure.com/"
api_version = "2024-02-01"

# Streamlit UI
st.title("IT HelpDesk QnA")


from langchain_community.document_transformers.openai_functions import (
    create_metadata_tagger,
)
from langchain_core.documents import Document
from langchain_openai import AzureChatOpenAI

schema = {
    "properties": {
        "Short description:": {"type": "string"},
        "Question:": {"type": "string"},
        "Article body:": {
            "type": "string",
            "description": "The PROBLEM/ISSUE and RESOLUTION will be presented inside the Article body",
        },
    },
    "required": ["Short description:", "Question:", "Article body:", "KEYWORDS"],
}

# CSV File uploader
excel_files = st.sidebar.file_uploader("Upload a Input file", type=["xlsx"])
if excel_files:
    excel_file = excel_files
    csv_file = 'output.csv'   

    # Read the Excel file
    df = pd.read_excel(excel_file)
    # Convert the Excel file to CSV
    df.to_csv(csv_file, index=False)
    st.write(f"Excel file has been converted to CSV file")

    # Read CSV content
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')

    # Convert CSV to PDF
    output_pdf = "output.pdf"
    
    # CSV to PDF Conversion Function
    def csv_to_pdf(dataframe, output_filename):
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        
        for i in range(len(dataframe)):
            row = dataframe.iloc[i]
            for col in dataframe.columns:
                pdf.multi_cell(0, 10, f"{col}: {row[col]}", border=0)
            pdf.ln(10)  # Add space after each row

        pdf.output(output_filename)

    csv_to_pdf(df, output_pdf)
    st.write("CSV converted to PDF successfully!")

    # Load the generated PDF content
    loader = PyMuPDFLoader(output_pdf)
    data = loader.load()
    st.write("Document Ingested Successfully.")

    # Split the loaded text
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=218)
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
        query = f"""What is the Resolution for the exact below mentioned Problem:\n {user_problem}\n\n 
                    Try to extract the resolution until any of this title arrives "NOTES" , "Short description:"\n
                    Note: Extract the exact answer from the input context and the answer of the problem will be part of the Resolution that can be found in Article body"""
        
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
