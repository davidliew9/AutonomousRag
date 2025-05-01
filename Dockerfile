# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy the rest of your project files
COPY . .

# Run your app
CMD ["python", "autorag.py"]