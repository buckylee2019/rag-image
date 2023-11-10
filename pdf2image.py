# STEP 1
# import libraries
import fitz
import os
import json
from langchain.docstore.document import Document
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.vectorstores import Milvus
from dotenv import load_dotenv
from glob import glob
from minio import Minio
import io
from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(
    separator = "。",
    chunk_size = 300,
    chunk_overlap  = 50,
    length_function = len,
    is_separator_regex = False,
)

load_dotenv()

BUCKET_NAME = os.getenv("BUCKET_NAME")
MINIO_OBJECT_URL = os.environ.get("MINIO_OBJECT_URL")
MINIO_PUBLIC_URL = os.environ.get("MINIO_PUBLIC_IP")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
wx_model = os.getenv("WX_MODEL")
INDEX_NAME = os.getenv("INDEX_NAME")
PDF_DIR = "/app/data/"
PARENT_IMG_DIR = "/app/data/images"

repo_id = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

hf = HuggingFaceHubEmbeddings(
    task="feature-extraction",
    repo_id = repo_id,
    huggingfacehub_api_token = HUGGINGFACEHUB_API_TOKEN,
)



client = Minio(
    MINIO_OBJECT_URL,
    access_key=os.environ.get("MINIO_ACCESS_KEY"),
    secret_key=os.environ.get("MINIO_SECRET_KEY"),
    secure=False
)


found = client.bucket_exists(BUCKET_NAME)
if not found:
    client.make_bucket(BUCKET_NAME)
    policy = {"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetBucketLocation","s3:ListBucket","s3:ListBucketMultipartUploads"],"Resource":["arn:aws:s3:::chat-demo"]},{"Effect":"Allow","Principal":{"AWS":["*"]},"Action":["s3:GetObject","s3:ListMultipartUploadParts","s3:PutObject","s3:AbortMultipartUpload","s3:DeleteObject"],"Resource":["arn:aws:s3:::chat-demo/*"]}]}
    client.set_bucket_policy(BUCKET_NAME, json.dumps(policy))

else:
    print(f"Bucket {BUCKET_NAME} already exists")

# STEP 2
# file path you want to extract images from


def extract_text_image(file):

    IMAGE_DIR = file.split('/')[-1].split('.')[0]
    if not os.path.exists("/app/data/images/"+IMAGE_DIR):
        os.mkdir("/app/data/images/"+IMAGE_DIR)
    # open the file
    pdf_file = fitz.open(file)
    documents = []
    metadata_dict = {}
    # STEP 3
    # iterate over PDF pages
    for page_index in range(len(pdf_file)):

        # get the page itself
        page = pdf_file[page_index]
        image_list = page.get_images()
        image_sources = []
        # printing number of images found in this page
        
        if image_list:
            print(
                f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
            # metadata = ({'image_source': "", 'page':page_index+1})
            # documents.append(Document(page_content=page.get_text(),metadata=metadata))
        for image_index, img in enumerate(page.get_images(), start=0):

            # get the XREF of the image
            xref = img[0]

            # extract the image bytes
            base_image = pdf_file.extract_image(xref)
            
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            image_src = os.path.join(IMAGE_DIR, f'{page_index}-{image_index}.{image_ext}')
            image_absoulte = os.path.join(PARENT_IMG_DIR,image_src)
            with open(image_absoulte , 'wb') as image_file:
                image_file.write(image_bytes)
                image_file.close()
            client.fput_object(
                BUCKET_NAME, f"{image_src}",image_absoulte,
            )
            os.remove(image_absoulte)
            # image_sources.append(image_src)
            cos_url = os.path.join("http://",MINIO_PUBLIC_URL,BUCKET_NAME, image_src)
            image_sources.append(cos_url)
        text = page.get_text().replace('\n','')
        metadata = {'image_source': json.dumps(image_sources,ensure_ascii=False), 'page':page_index+1}
        splitted = len(text_splitter.split_text(text))
        docs = text_splitter.create_documents([text],metadatas=[metadata for i in range(splitted)])
        
        documents.extend(docs)

    return documents



