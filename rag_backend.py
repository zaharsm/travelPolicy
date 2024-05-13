
#1. Import OS, Document Loader, Text Splitter, Bedrock Embeddings, Vector DB, VectorStoreIndex, Bedrock-LLM
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes.vectorstore import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock


#5c. Wrap within a function
def hr_index():
    #2. Define the data source and load data with PDFLoader(https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf)
    data_source = PyPDFLoader('https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf')
    # data_source = PyPDFLoader('https://jumpshare.com/s/GpdoYgNyUK5xNFSHqWqH')
    # pages = data_source.load_and_split()
    # print(pages[2])

    # https://python.langchain.com/v0.1/docs/modules/data_connection/document_transformers/recursive_text_splitter/


    #3. Split the Text based on Character, Tokens etc. - Recursively split by character - ["\n\n", "\n", " ", ""]
    data_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size = 100, # Breaks it down into small chunks of characters 
        chunk_overlap=10, # Avoid Leakages by overlapping
    )

    # sample_data= "Learning a little each day adds up. Research shows that students who make learning a habit are more likely to retain information and reach their goals. Set time aside to learn and get reminders using your learning event scheduler."
    # text_split= text_splitter.split_text(pages)
    # print(text_split)


    #4. Create Embeddings -- Client connection
    #https://api.python.langchain.com/en/latest/embeddings/langchain_community.embeddings.bedrock.BedrockEmbeddings.html
    create_embeddings=BedrockEmbeddings(
        credentials_profile_name='default',
        region_name='us-east-1',
        model_id='amazon.titan-embed-text-v1'
    )

    #5à Create Vector DB, Store Embeddings and Index for Search - VectorstoreIndexCreator
    # https://python.langchain.com/v0.1/docs/integrations/vectorstores/faiss/
    # https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html
    
    data_index=VectorstoreIndexCreator(
        embedding= create_embeddings,
        text_splitter=data_splitter,
        vectorstore_cls=FAISS
    )

    #5b Create index for HR Report
    db_index=data_index.from_loaders([data_source])
    return db_index

 
#6a. Write a function to connect to Bedrock Foundation Model
def hr_llm():
    llm=Bedrock(
        credentials_profile_name='default',
        region_name='us-east-1',
        model_id='anthropic.claude-v2',
        model_kwargs={
            "temperature": 0.5,
            "top_p": 1,
            "top_k": 250,
            "max_tokens_to_sample": 200
        }
    )
    return llm


#6b. Write a function which searches the user prompt, searches the best match from Vector DB and sends both to LLM.
def hr_rag_response(index,question):
    rag_llm=hr_llm()
    hr_rag_query=index.query(question=question,llm=rag_llm)
    return hr_rag_query
 
# Index creation --> https://api.python.langchain.com/en/latest/indexes/langchain.indexes.vectorstore.VectorstoreIndexCreator.html