import os

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
    Assume that master01 can be called first master node, master02 can be called second master node and so on.
    Assume that stwrk01 can be called first worker node, stwrk02 can be called second worker node and so on.
    '"""
)

def reset_engine():
    st.session_state['coding'] = 0
    st.session_state['input'] = ""

def set_download_state():
    st.session_state['downloading'] = 1


# When interpreting the specified changes, 
#    Assume that a worker node is another name for a replica. 
#    Assume that MAC address is another name for bootMACAddress. 
#    Assume that master01 can be called first master node, master02 can be called second master node 
#    and so on.
#    Assume that stwrk01 can be called first worker node, stwrk02 can be called second worker node
#    and so on.
#    Retain unchanged values in the output.
#    Highlight produced changes in red font. Do not use red font to display any other text.
#    If the number of worker nodes reduce, ßremove the YAML sections associated with the unused worker nodes, keeping the master node sections unchanged.
#    If the number of master nodes reduce, remove the YAML sections associated with the unused master nodes, keeping the worker node sections unchanged.

llm = ChatOpenAI(model_name='gpt-3.5-turbo-16k', temperature=0)

#response = llm("Tell me something unique about the Indian state of Kerala")
#print(response)

st.title("Edge Deployment and Ops Engine")
st.subheader("_Auto-generate Infra-as-Code for the Containerized Edge_")

col1, col2, col3 = st.columns([2,2,3])

with col3:
    st.button(":violet[Start New Edge Conversation]", on_click = reset_engine)

with col1:
    option = st.selectbox(
        ':violet[Choose Base Edge Configuration]',
        ('EPC-on-Openshift', 'MME-on-OpenShift'))

    if option == "EPC-on-OpenShift":
        st.session_state['base'] = "ocplabnk"
    elif option == "MME-on-OpenShift":
        st.session_state['base'] = "ocplabnk" # To be changed


if 'coding' not in st.session_state:
    st.session_state['coding'] = 0

if 'downloading' not in st.session_state:
    st.session_state['downloading'] = 0

if 'base' not in st.session_state:
    st.session_state['base'] = "ocplabnk"

if (st.session_state['coding'] == 0) and (st.session_state['downloading'] == 0):
  
    loader = TextLoader(openshift_base_yaml_path, encoding='utf8')
    documents = loader.load()

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    db = FAISS.from_documents(documents, embeddings)

    IaC = db.similarity_search_with_score(st.session_state['base'])
    st.session_state['matched_IaC'] = IaC[0][0].page_content
    st.session_state['orig_IaC'] = st.session_state['matched_IaC']
    print (st.session_state['orig_IaC'])
    print ("Similarity Score =", IaC[0][1])

col1, col2 = st.columns([20, 1])

with col1:
    edge_spec = st.text_input ("""Please describe the site-specific changes for your edge node relative to 
the base configuration listed below. You may specify changes incrementally. 
Each change will apply on the YAML code generated in the previous step and will generate the resulting YAML. 
To start afresh from the base, click on the :violet[*Start New Edge Conversation*] button.
:violet[**Usage Examples**]: :orange[Change the password of the second master node to _violin_.]
:green[Reduce the number of master nodes to 2 and the number of worker nodes to 1.]
:blue[Change the MAC address of the second worker node to 01:02:03:04:05:06 and the
IP address of its bond0.3803 interface to 172.1.2.3.]
:green[Increase master nodes to 4 and add the section corresponding to the 4th master node.]
:orange[Fill missing worker node sections.]
:blue[Remove additional worker node sections.]
:green[Increase number of master nodes to 5.]
:orange[Add 2 more worker node sections.]""", 
key="input")

if edge_spec and st.session_state['downloading'] == 0:
    with col1:
        st.write ("Please wait. This might take a minute.. :sunglasses:")
    #edge_spec = "Decrease the number of worker nodes to 2. Delete sections associated with the removed worker nodes. Also change the IPv4 address of second worker node to 5.6.7.8 and its boot MAC address to 01:02:03:04:05:06."
    st.session_state["coding"] = 1

    chain = LLMChain(llm=llm, prompt=Prompt_template)

    with get_openai_callback() as cb:
        generated_yaml = chain.run(source_IaC=st.session_state['matched_IaC'], edge_config_changes=edge_spec)
        print (cb)

    print (generated_yaml)
    st.session_state['matched_IaC'] = generated_yaml
    #edge_spec = "Change the number of master node replicas to 4. Add sections corresponding to any additional master nodes. Keep existing sections unchanged."

col1, col2 = st.columns([1, 2])

if st.session_state['matched_IaC']:
    with col1:
        st.write (":violet[Please find generated YAML below]")

    with col2:
        st.download_button(':violet[Download]', st.session_state['matched_IaC'], 
                       file_name="edge-deploy.yaml", on_click=set_download_state)

    st.code(st.session_state['matched_IaC'], language="yaml", line_numbers=False)   

