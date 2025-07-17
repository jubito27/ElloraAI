import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from dotenv import dotenv_values
import streamlit as st

def get_medic_response(query):
    try:
        # Initialize QA system (cached for performance)
        qa_chain = initialize_qa_chain()
        if not qa_chain:
            return {"answer": "Knowledge system unavailable", "sources": []}
        
        # Get response from ChromaDB
        result = qa_chain({"query": query})
        
        # Format sources with metadata
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content,
                "reference": doc.metadata.get("source", "Ancient Text")
            })
        
        return {
            "answer": result["result"],
            "sources": sources
        }
        
    except Exception as e:
        return {"answer": f"Error retrieving wisdom: {str(e)}", "sources": []}


@st.cache_resource
def initialize_qa_chain(persist_directory="chroma-medic-store"):
    """Cached initialization of QA system"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
         # Load existing data from ChromaDB
        env_var = dotenv_values(".env")
        api_key = genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
        llm = ChatGoogleGenerativeAI(api_key=api_key,model="gemini-2.0-flash", temperature=0.7)
        return RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type="stuff",
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Knowledge system initialization failed: {e}")
        return None

# if __name__ == "__main__":
#     # Example usage
#     while True:
#         query = input("Enter your question about Medic texts (or type 'exit' to quit): ")
#         if query.lower() == 'exit':
#             break
#         get_medic_response(query)
