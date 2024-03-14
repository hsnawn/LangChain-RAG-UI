import pandas as pd
import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores.utils import filter_complex_metadata


class ShippingAssistant:
    def __init__(self, path_to_docx):
        self.path_to_docx = path_to_docx
        self.loader = Docx2txtLoader(path_to_docx)
        self.docs = self.loader.load()
        self.docs = filter_complex_metadata(self.docs)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.splits = self.text_splitter.split_documents(self.docs)
        self.vectorstore = Chroma.from_documents(documents=self.splits, embedding=OpenAIEmbeddings())
        self.retriever = self.vectorstore.as_retriever()
        self.mem = ""

    def ask_query(self, query):
        # self.mem = f"Assistant: Hello {carrier_name}, We’re pleased to inform you that you are the winning bidder on {shipment_id} from {departure_city} to {destination_city} picking etc etc"
        rag_template = """
        You are a shipping assistant for question-answering tasks. Use the following pieces of context to answer the question. If you don't know the answer, just refuse to answer it.and reply to greetings. Use three sentences maximum and keep the answer concise.
        CONTEXT:
        ```
            {docs}
        ```
        PREVIOUS DISCUSSION:
        ```
            {history}
        ```
        QUERY:
        ```
            {query}
        ```
        ANSWER:
        """
        prompt = PromptTemplate(
            partial_variables={"history": self.mem, "docs": self.retriever},
            input_variables=["query"],
            template=rag_template,
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        rag_chain = prompt | llm | StrOutputParser()

        resp = rag_chain.invoke({"query": f"{query}"})
        self.mem += f"\nUser: {query}"
        self.mem += f"\nAssistant: {resp}"

        print(self.mem)
        return resp
    
    
path_to_docx = "ALGO_VENTURE_FAQ.docx"

assistant = ShippingAssistant(path_to_docx)

# Streamlit UI code
st.title("ALGO VENTURE Assistant")

# with st.sidebar:
#     name = st.text_input("Your Name", key="name", type="default")
#     company_name = st.text_input("Company Name", key="company", type="default")

initial_message = f"""
Hello, I am ALGO VENTURE. I am here to help you.  
"""


if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": initial_message}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    response = assistant.ask_query(prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)