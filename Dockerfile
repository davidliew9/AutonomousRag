# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project files
COPY . .

# Run your app
CMD ["streamlit", "run", "autorag.py", "--server.port=8000", "--server.address=0.0.0.0"]
