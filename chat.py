from uuid import UUID
from langchain.schema.output import LLMResult
import streamlit as st

try:
    from langchain import PromptTemplate
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.memory import ConversationBufferMemory
except ImportError:
    raise ImportError("Could not import langchain: Please install ibm-generative-ai[langchain] extension.")
from langchain.embeddings import HuggingFaceHubEmbeddings
from genai.credentials import Credentials
from genai.extensions.langchain import LangChainInterface
from genai.schemas import GenerateParams
import os
from typing import Any
import os
from langchain.callbacks.base import BaseCallbackHandler
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
import chromadb
from pdf2image import extract_text_image

load_dotenv()

st.set_page_config(page_title="Chat with Documents", page_icon="💡")
st.title("RAG")
uploaded_file = st.file_uploader("Upload your PDF")


PREFIX_PROMPT = "<s>[INST] <<SYS>>"
# user_api_key = st.sidebar.text_input(
#     label="#### Your Bam API key 👇",
#     placeholder="Paste your Bam API key, pak-",
#     type="password")
user_api_key = os.environ.get("BAM_API_KEY")
INDEX_NAME = os.environ.get("INDEX_NAME")
system_prompt = st.sidebar.text_area(
    label="System prompt for model",
    placeholder= """\
    You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.\
    """
)


DEFAULT_SYSTEM_PROMPT = """\
    You are a helpful, respectful and honest assistant. You should answer the question directly from the given documents, you are responsible for finding the best answer among all the documents. Follow the rules below:\
    Summarize the related documents to user question using the following format, Use Markdown to display : Topic of the document, Step by Step instruction for user question, Image sources from document\
    Display the list of Image sources of related document in the following markdown format: ![image text](image sources "IMAGE"),\
    Note: If the question is not related to the given context, check the chat history to find answer, otherwise SAY "I can't answer the question!"\
"""

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if token:
            self.text += token
            self.container.markdown(self.text)
    def on_llm_end(self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        return super().on_llm_end(response, run_id=run_id, parent_run_id=parent_run_id, **kwargs)
    

params = GenerateParams(
        decoding_method="greedy",
        max_new_tokens=1024,
        min_new_tokens=1,
        stream=True,
        top_k=50,
        top_p=1,
    )



WX_MODEL = os.environ.get("WX_MODEL")
creds = Credentials(user_api_key, "https://bam-api.res.ibm.com/v1")

repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")



use_history = st.sidebar.checkbox(label="use mermory")

hf = HuggingFaceHubEmbeddings(
    task="feature-extraction",
    repo_id = repo_id,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
)
if uploaded_file:
    collection_name = uploaded_file.name.split('/')[-1].split('.')[0]
    bytes_data = uploaded_file.getvalue()
    with open("/app/data/"+uploaded_file.name,"wb") as f:
        f.write(bytes_data)
    document = extract_text_image("/app/data/"+uploaded_file.name)
    index = Chroma.from_documents(
            documents=document,
            embedding=hf,
            collection_name=collection_name,
            persist_directory=INDEX_NAME
        )
    os.remove("/app/data/"+uploaded_file.name)
    uploaded_file = ""

set_vectorstore = os.path.exists(INDEX_NAME)

print("\n\n\n",INDEX_NAME,"is set ",set_vectorstore,"\n\n\n")
if not set_vectorstore:
    
    vectorstore = Chroma.from_documents(
            documents=[],
            embedding=hf,
            persist_directory=INDEX_NAME
        )
else:
    collection_name = "langchain"
    client = chromadb.PersistentClient(path=os.environ.get("INDEX_NAME"))
    collection_name = st.sidebar.selectbox("Select the documents",
        set([cols.name for cols in client.list_collections()]))
    vectorstore = Chroma(
                    embedding_function=hf,
                    collection_name=collection_name,
                    persist_directory=INDEX_NAME
                
        )




def get_prompt_template(system_prompt=system_prompt, history=False):
    
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS
    document_with_metadata_prompt = PromptTemplate(
    input_variables=["page_content", "image_source"],
    template="\nDocument: {page_content}\n\tImage sources: {image_source}",
)
    if history:
        instruction = """
        Context: {history} \n {summaries}
        User question: {question}
        Answer the question in Markdown format,
        Markdown: """

        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["history", "summaries", "question",], template=prompt_template)
    else:
        instruction = """
        Context: {summaries}
        User: {question}
        Answer the question in Markdown format.
        Markdown:
        """

        prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
        prompt = PromptTemplate(input_variables=["summaries", "question"], template=prompt_template)
    
    memory = ConversationBufferMemory(input_key="question", memory_key="history")

    return (
        document_with_metadata_prompt,
        prompt,
        memory,
    )

def retrieval_qa_pipline(db, use_history, llm, system_prompt):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - db (vectorestore): Specifies the preload vector db
    - system_prompt (str): Define from default or from web UI
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQAWithSourcesChain: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    retriever = db.as_retriever(search_kwargs={'k': 3})

    # get the prompt template and memory if set by the user.
    doc_promt, prompt, memory = get_prompt_template( system_prompt=system_prompt,history=use_history)

    # load the llm pipeline

    if use_history:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            chain_type_kwargs={"prompt":prompt,
                "document_prompt":doc_promt, "memory": memory},
        )
    else:
        qa = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            chain_type_kwargs={
                "prompt":prompt,
                "document_prompt":doc_promt,
            },
        )

    return qa
clear_conversation = st.sidebar.button(
    label="Clear conversation"
)


if not system_prompt:
    system_prompt = DEFAULT_SYSTEM_PROMPT
if user_api_key:
    if "source" not in st.session_state:
        st.session_state.source = []


    if "messages" not in st.session_state:
        st.session_state.messages = []
        with st.chat_message("assistant"):
            st.markdown("Hi! I am a helpful assistant and I can help you answer question about your documents.")

    if clear_conversation:
        st.session_state.messages = []
        st.session_state.source = []
    for user_input,response in st.session_state.messages:
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            st.markdown(response)

    if prompt := st.chat_input("Ask your question.."):
        # st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            
            message_placeholder = st.empty()
            stream_handler = StreamHandler(message_placeholder)

            # message_placeholder            
            llm = LangChainInterface(
                model=WX_MODEL,
                credentials=creds,
                params=params,
                callbacks=[stream_handler]
            )
            qa_chain = retrieval_qa_pipline(vectorstore,True,llm,system_prompt)
            res = qa_chain(prompt,return_only_outputs=True)
            
        st.session_state.messages.append((prompt,res['answer']))
      