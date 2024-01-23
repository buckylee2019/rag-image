

from langchain.document_loaders import PyPDFLoader
from langchain.chains.summarize import load_summarize_chain
from genai.model import Credentials
from genai.schemas import GenerateParams
from genai.extensions.langchain.llm import LangChainInterface
from genai.exceptions import GenAiException
from os import environ
import os
import logging
from dotenv import load_dotenv
import json
import streamlit as st
from langchain import PromptTemplate
from pdf2image import extract_text_image
load_dotenv()

@st.cache_data
def store_document(filename):
    documents = extract_text_image(filename)
    
    return documents

with st.sidebar.form("my-form", clear_on_submit=True):
    uploaded_file = st.file_uploader("FILE UPLOADER")
    submitted = st.form_submit_button("UPLOAD!")

    if submitted and uploaded_file is not None:
        st.write("UPLOADED!")
        collection_name = uploaded_file.name.split('/')[-1].split('.')[0]
        bytes_data = uploaded_file.getvalue()
        with open(environ['DATA_ROOT']+uploaded_file.name,"wb") as f:
            f.write(bytes_data)
                


if uploaded_file:
    documents = store_document(environ['DATA_ROOT']+uploaded_file.name)

else:
    st.markdown("Upload your file first.")
prompt_summarize = {
    "stuff":{ 
        "prompt_template": """[INST]<<SYS>>Write a concise summary of the following text delimited by triple backquotes. 寫一段
Return your response in bullet points which covers the key points of the text.<</SYS>>
```{text}```
用繁體中文列點摘要: [/INST]
"""},
    "map_reduce":{
        "map_prompt_template" :"""
[INST]<<SYS>>
Write a summary of this chunk of text that includes the main points and any important details.<</SYS>>
{text}
用繁體中文寫出main points 以及import details[/INST]
""",
        "combine_prompt_template" : """
[INST]<<SYS>>
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.
<</SYS>>
```{text}```
用繁體中文列點摘要:[/INST]
"""
        },
    "refine":{
        "question_prompt_template":"""
                  [INST]<<SYS>>Please provide a summary of the following text. <</SYS>>
                  TEXT: {text}
                  SUMMARY:[/INST]
                  """,
        "refine_prompt_template":"""
              [INST]<<SYS>>Write a concise summary of the following text delimited by triple backquotes.
              Return your response in bullet points which covers the key points of the text. <</SYS>>
              ```{text}```
              BULLET POINT SUMMARY:[/INST]
              """
    }
                    
}

strategy = st.sidebar.selectbox("Select the documents",
        set(["stuff","map_reduce","refine"]))
num_pages = st.sidebar.slider('Number of pages', 0, 10, 5)
params = GenerateParams(
    decoding_method='greedy',
    repetition_penalty=1.0,
    min_new_tokens=1,
    max_new_tokens=1024
)

credentials = Credentials(environ['GENAI_KEY'], api_endpoint=environ['GENAI_API'])
llm = LangChainInterface(credentials=credentials, model=environ['WX_MODEL'], params=params)

run = st.sidebar.button('Run')
if run:
    
    try:
        if strategy == "stuff":
            chain = load_summarize_chain(llm, chain_type=strategy,prompt=PromptTemplate(
                template=prompt_summarize[strategy]['prompt_template'], input_variables=["text"]
            ))
        elif strategy == "map_reduce":
            chain = load_summarize_chain(llm, chain_type=strategy,map_prompt=PromptTemplate(template=prompt_summarize[strategy]['map_prompt_template'], input_variables=["text"]),
        combine_prompt=PromptTemplate(template=prompt_summarize[strategy]['combine_prompt_template'], input_variables=["text"]),
            )
        elif strategy == "refine":
            chain = load_summarize_chain(llm, chain_type=strategy,question_prompt=PromptTemplate(template=prompt_summarize[strategy]['question_prompt_template'], input_variables=["text"]),
        refine_prompt=PromptTemplate(template=prompt_summarize[strategy]['refine_prompt_template'], input_variables=["text"]),
            )
        message = chain.run(documents[:num_pages])
        failed = False
    except GenAiException as error:
        message = str(error)
        failed = True

    if failed:
        print(json.dumps({'type': 'error', 'message': message}))
    else:
        st.markdown(message)