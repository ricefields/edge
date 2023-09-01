import os

#from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownTextSplitter
from langchain.indexes import VectorstoreIndexCreator
from langchain.callbacks import get_openai_callback

from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.vectorstores import FAISS

from PyPDF2 import PdfReader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders import JSONLoader
#import unstructured
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain import ConversationChain
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
#import chromadb
#from chromadb.config import Settings
import yaml

#from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

from langchain.document_loaders import DirectoryLoader, TextLoader
import streamlit as st
index_name = "./index"
openshift_base_yaml_path = "./openshift-base.yaml"


# Prompt template
Prompt_template = PromptTemplate(
    input_variables = ["source_IaC", "edge_config_changes"],
    template = 
    """'
    As a subject matter expert on YAML syntax, I want you to process an input YAML snippet 
    and generate a formatted and indented output YAML snippet based on changes specified to the input YAML. 
    Use the following as the input YAML snippet: {source_IaC}.

    Use the following as the specified changes to be done to the above input YAML snippet: 
    {edge_config_changes}.
    Generate the output YAML only; do not add any additional text before or after the generated YAML.
    '"""
)

# When interpreting the specified changes, 
#    Assume that a worker node is another name for a replica. 
#    Assume that MAC address is another name for bootMACAddress. 
#    Assume that master01 can be called first master node, master02 can be called second master node 
#    and so on.
#    Assume that stwrk01 can be called first worker node, stwrk02 can be called second worker node
#    and so on.
#    Please mark the changes made in the output YAML snippet in red font. 
#    Retain unchanged values in the output.

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)

#response = llm("Tell me something unique about the Indian state of Kerala")
#print(response)

st.title("Edge Deployment Engine")
st.subheader("Auto-generate Edge Infrastructure-as-Code")

loader = TextLoader(openshift_base_yaml_path, encoding='utf8')
documents = loader.load()

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

db = FAISS.from_documents(documents, embeddings)

IaC = db.similarity_search_with_score("ocplabnk")
matched_IaC = IaC[0][0].page_content
print (matched_IaC)
print ("Similarity Score =", IaC[0][1])

edge_spec = st.text_input ("Please describe the site-specific changes for your edge node. You can specify changes step by step. Each specified change will apply to the YAML code generated in the previous step.", key="input")
if edge_spec:
    st.write ("Please wait. This might take a minute.. :sunglasses:")
    #edge_spec = "Decrease the number of worker nodes to 2. Delete sections associated with the removed worker nodes. Also change the IPv4 address of second worker node to 5.6.7.8 and its boot MAC address to 01:02:03:04:05:06."

    chain = LLMChain(llm=llm, prompt=Prompt_template)

    with get_openai_callback() as cb:
        generated_yaml = chain.run(source_IaC=matched_IaC, edge_config_changes=edge_spec)
        print (cb)

    print (generated_yaml)
    st.code(generated_yaml, language="yaml", line_numbers=False)

    matched_IaC = generated_yaml
    #edge_spec = st.text_input ("Please describe the site-specific changes for your edge node.", key=i)
    #st.write ("Please wait. This might take a minute.. :sunglasses:")
    #edge_spec = "Change the number of master node replicas to 4. Add sections corresponding to any additional master nodes. Keep existing sections unchanged."

print ("hello world")


