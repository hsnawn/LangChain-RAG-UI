import streamlit as st
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


class ShippingAssistant:
    def __init__(self, path_to_docx):
        self.path_to_docx = path_to_docx
        self.loader = Docx2txtLoader(path_to_docx)
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        self.splits = self.text_splitter.split_documents(self.docs)
        self.vectorstore = Chroma.from_documents(
            documents=self.splits, embedding=OpenAIEmbeddings()
        )

        self.mem = "Assistant: Hello, I am ALGO VENTURE. I am here to help you."

    def ask_query(self, query, hist):
        retriever = self.vectorstore.similarity_search(query)
        merged_content = "\n".join([doc.page_content for doc in retriever[:2]])
        
        rag_template = """
        You are ALGO VENTURE assistant. Use the following pieces of context to answer the question.
        If you don't know the answer, just refuse to answer it.
        Use three sentences maximum and keep the answer concise.
        Reply to greetings and refuse to answer irrelevant questions.
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
            partial_variables={"history": hist, "docs": merged_content},
            input_variables=["query"],
            template=rag_template,
        )
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        rag_chain = prompt | llm | StrOutputParser()

        resp = rag_chain.invoke({"query": f"{query}"})
        return resp


path_to_docx = "ALGO_VENTURE_FAQ.docx"

assistant = ShippingAssistant(path_to_docx)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Streamlit UI code
st.title("ALGO VENTURE Assistant")

hist = ""
if "messages" in st.session_state:
    for msg in st.session_state.messages:
        hist += f"\n{msg['role']}: {msg['content']}"

initial_message = f"Hello, I am ALGO VENTURE. I am here to help you."
# hist += f"\nAssistant: {initial_message}"

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": initial_message}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    print(hist)
    response = assistant.ask_query(prompt, hist)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
