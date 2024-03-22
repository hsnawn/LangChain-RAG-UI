import os

from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.globals import set_debug
import chromadb.utils.embedding_functions as embedding_functions
import chromadb
import uuid
class ShippingAssistant:
    def __init__(self):
            print("***************************************INIT*****************************************")
            self.openai_api_key = os.environ.get('OPENAI_API_KEY')
            # self.openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            #                 api_key=self.openai_api_key,
            #                 model_name="text-embedding-3-small",
            #             )
            # self.emb_fn = OpenAIEmbeddings()
            # self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=self.emb_fn)
            # # self.vectorstore.persist()
            # # self.retriever = self.vectorstore.as_retriever(k=4)

            self.collection1=self.create_collection_f('cust',r"/root/LangChain-RAG-UI/KnowledgeBase")
            

        # self.mem = "Assistant: Hello, I am ALGO VENTURE. I am here to help you."
    def context_retreiver(self, query):
        collection = self.collection1
        return collection.query(query_texts=[query], n_results=4, include=["documents"])
    
    def ask_query(self, query, hist):
        # set_debug(True)
        # retriever = self.vectorstore.similarity_search(query)

        docs = self.context_retreiver(query)
        docs = docs["documents"][0]
        
        merged_content = "\n ###################### Next Chunk ###################### \n".join([doc for doc in docs])
        for doc in docs:
            print("**********************************************************************************")
            print(doc)

        # print(merged_content)

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
    
    def create_collection_f(self,collection_name, folder_path):

        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=self.openai_api_key,
                model_name="text-embedding-3-small",
            )
        client = chromadb.EphemeralClient()
        try:
            
            collection = client.create_collection(
                    collection_name, embedding_function=openai_ef
                )
            for docx in os.listdir(folder_path):        
                loader = Docx2txtLoader(os.path.join(folder_path, docx))
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
                docs = text_splitter.split_documents(documents)

                for doc in docs:
                    collection.add(
                        ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
                    )
            print("NEW COLLECTION")
        except:
            print("COLLECTION ALREADY CREATED HAHAHAH")
            collection = client.get_collection(
                collection_name, embedding_function=openai_ef
            )



        
        return collection