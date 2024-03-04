from dotenv import load_dotenv

load_dotenv()

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

import os
from typing import Any
from langchain.chains import RetrievalQA
from langchain_community.vectorstores.pinecone import Pinecone as PineconeLangChain
from pinecone import Pinecone


pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

INDEX_NAME = "documentation-helper"
MODEL = "dolphin-mistral"
DOCS_PATH = "langchain-docs/"


def run_llm(query: str) -> Any:
    embeddings = OllamaEmbeddings(model=MODEL, show_progress=True)
    docsearch = PineconeLangChain.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    chat = ChatOllama(model=MODEL,verbose=True, temperature=0)
    qa = RetrievalQA.from_chain_type(
        llm=chat,
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    return qa({"query": query})


if __name__ == "__main__":
    print(run_llm(query="What is LangChain chain?"))
