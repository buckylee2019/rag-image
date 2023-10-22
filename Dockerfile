# Use an official Python base image from the Docker Hub
FROM python:3.10-slim

# Install browsers
RUN apt-get update && apt-get install build-essential -y
# Install utilities
RUN apt-get install -y curl wget git

# Declare working directory
WORKDIR /app

COPY . .

# Install any necessary packages specified in requirements.txt.
RUN pip install -r requirements.txt

