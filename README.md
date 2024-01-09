# rag-image
## Overview

A simple set up for running RAG with images,different from tradition RAG which normally process text-only content. 
For IBM internal development, feel free to use BAM as an alternative, for pilot case please use watsonx, in case you suffered from SDK transition.

Warning: Streaming method in ibm machine learning has issue displaying Chinese, hence in watsonx mode, result will be showed only when generation is completed.

## Prerequisites

Before running the script, ensure you have the following prerequisites:

- docker or podman
- docker-compose

## Installation

1. Clone the repository or download the script to your local machine.

2. Create a `.env` file from `.env_template`

3. To use watsonx, please make sure set the USE_WATSONX to True, and fill out the API Key and Project ID.


## Usage

To use service:

1. Execute `docker-compose up -d` under the repo
2. Open localhost:8015
3. Upload the PDF you want to chat with. 
![./data/images/rag-upload-pdf.png](https://github.com/buckylee2019/rag-image/blob/main/data/images/rag-upload-pdf.png)
4. Wait a moment
5. Select your document on the sidebar
   ![Select on sideBar](https://github.com/buckylee2019/rag-image/blob/main/data/images/rag-sidebar.png)
6. Start your chat
   ![chat](https://github.com/buckylee2019/rag-image/blob/main/data/images/rag-chat.png)
## License

This script is provided under the [MIT License](LICENSE). You are free to use, modify, and distribute it as needed.