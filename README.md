# rag-image
## Overview

A simple set up for running RAG with images,different from tradition RAG which normally process text-only content. 
Note: Currently, the repo works for IBM BAM only.

## Prerequisites

Before running the script, ensure you have the following prerequisites:

- docker or podman
- docker-compose

## Installation

1. Clone the repository or download the script to your local machine.

2. Create a `.env` file from `.env_template`:

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