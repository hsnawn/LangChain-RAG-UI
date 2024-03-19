import streamlit as st
import os
import gspread
from datetime import datetime
from oauth2client.service_account import ServiceAccountCredentials
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate


class ShippingAssistant:
    def __init__(self, docx):
        self.docx = docx
        for file in os.listdir(self.ocx):
            self.loader = Docx2txtLoader(file)
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
        You are ALGO VENTURE assistant.
        Algo Venture, a leader in AI algorithmic trading education, equips traders with the tools to harness AI’s potential in the dynamic financial market. By seamlessly blending strategies with practical applications, we foster an environment of learning and growth, turning trading enthusiasts into experts.
        Use the following pieces of context to answer the question.
        Use three sentences maximum and keep the answer concise.
        Reply to greetings and refuse to answer off-topic questions.
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

def write_to_google_sheets(time, name, phone_number, email):
    scopes=['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
    creds=ServiceAccountCredentials.from_json_keyfile_name('sheets.json', scopes)
    file=gspread.authorize(creds)
    workbook=file.open('Lead Coolectionsheet')
    sheet = workbook.sheet1
    sheet.append_row([time, name, phone_number, email])

def open_whatsapp_chat(whatsapp_number):
    whatsapp_link = f"https://api.whatsapp.com/send/?phone={whatsapp_number}&text=I%27m+interested+in+Algo+Venture+services&type=phone_number&app_absent=0"
    button_html = f'<a href="{whatsapp_link}" target="_blank"><button>Chat on WhatsApp</button></a>'
    st.markdown(button_html, unsafe_allow_html=True)

docx =[]
path_to_docx_folder = "KnowledgeBase"
for docx_file in os.listdir(path_to_docx_folder):
    if docx_file.endswith(".docx"):
        docx.append(os.path.join(path_to_docx_folder, docx_file))

assistant = ShippingAssistant(docx)

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

whatsapp_number = "6589264599"
open_whatsapp_chat(whatsapp_number)
# hist = ''

# Define the function to show the user information input form

def show_user_info_form():
    name = st.text_input("Enter your name:")
    phone_number = st.text_input("Enter your phone number:")
    email = st.text_input("Enter your email address:")
    submit_button = st.button("Submit")
    skip_button = st.button("Skip")

    if submit_button:
        # Process the submitted information (e.g., store in session_state)
        st.session_state.user_info = {"name": name, "phone_number": phone_number, "email": email}
        current_datetime = datetime.now()
        current_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        write_to_google_sheets(current_datetime, name, phone_number, email)
        st.success("Information submitted successfully!")
        return True
    elif skip_button:
        st.session_state.user_info = {"name": "", "phone_number": "", "email": ""}
        st.success("Skipping user information.")
        return True
    else:
        return False
    
def show_chat_interface():
    hist = ""  # Initialize hist variable
    if "messages" in st.session_state:
        for msg in st.session_state.messages:
            hist += f"\n{msg['role']}: {msg['content']}"

    initial_message = f"Hello, I am ALGO VENTURE Assistant. I am here to help you."
    # hist += f"\nAssistant: {initial_message}"

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            {"role": "assistant", "content": initial_message}
        ]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        print(hist)  # Consider removing this line unless necessary
        response = assistant.ask_query(prompt, hist)
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

# Main Streamlit code
if "user_info" not in st.session_state or not st.session_state.user_info:
    chat_placeholder = st.empty()
    # Show user information input form
    if show_user_info_form():
        # st.write("Moving to chat interface...")
        show_chat_interface()

    # Clear the placeholder if user info form is submitted
    if "user_info" in st.session_state and st.session_state.user_info:
        chat_placeholder.empty()
        
else:
    # Create an empty placeholder for the chat interface
    chat_placeholder = st.empty()
    show_chat_interface()  # Show chat interface

    # Clear the placeholder if user info form is submitted
    if "user_info" in st.session_state and st.session_state.user_info:
        chat_placeholder.empty()
