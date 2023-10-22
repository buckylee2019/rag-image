# rag-image
## Overview

A simple set up for running RAG with images,different from tradition RAG which normally process text-only content. 

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

[!image](./data/images/rag-upload-pdf.png)

## License

This script is provided under the [MIT License](LICENSE). You are free to use, modify, and distribute it as needed.