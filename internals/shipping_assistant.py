import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug

class ShippingAssistant:
    def __init__(self):
        self.docx =[]
        self.path_to_docx_folder = "KnowledgeBase"

        print("###################### Hello From INIT ######################")
        print("Docs Count: ",os.listdir(self.path_to_docx_folder))
        for docx_file in os.listdir(self.path_to_docx_folder):
            if docx_file.endswith(".docx"):
                self.docx.append(os.path.join(self.path_to_docx_folder, docx_file))
                print("docx: ", os.path.join(self.path_to_docx_folder, docx_file))
        print("DOCX: ",self.docx)

        for file in self.docx:
            self.loader = Docx2txtLoader(file)
            self.docs = self.loader.load()
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            self.splits = self.text_splitter.split_documents(self.docs)
            print(len(self.splits))
            # print(self.splits)
            self.vectorstore = Chroma.from_documents(
                documents=self.splits, embedding=OpenAIEmbeddings()
            )
            self.vectorstore.persist()
            self.retriever = self.vectorstore.as_retriever(k=4)

            

        # self.mem = "Assistant: Hello, I am ALGO VENTURE. I am here to help you."

    def ask_query(self, query, hist):
        # set_debug(True)
        # retriever = self.vectorstore.similarity_search(query)
        docs = self.retriever.invoke(query)
        merged_content = "\n ###################### Next Chunk ###################### \n".join([doc.page_content for doc in docs])
        for doc in docs:
            print("**********************************************************************************")
            print(doc.page_content)
        rag_template = """
        YOUR ROLE:
        You are an AI virtual assistant employed by ALGO VENTURE, responsible for assisting customers with their inquiries.
        You are having conversation with a customer. 
        
        TASK:
        Your primary task is to provide helpful guidance and address any concerns they may have.
        To provide accurate support, please use the provided context.
        If you don't know the answer, just say that you don't know the answer. Do not try to make up an answer.
        
        CONTEXT:
        ```
        {docs}
        ```
        PREVIOUS DISCUSSION:
        {history}
        user: {query}
        assistant:
        """
        prompt = PromptTemplate(
            partial_variables={"history": hist, "docs": merged_content},
            input_variables=["query"],
            template=rag_template,
        )
        # print(merged_content)
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        
        rag_chain = prompt | llm | StrOutputParser()

        resp = rag_chain.invoke({"query": f"{query}"})
        return resp