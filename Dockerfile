# Use the official Python image as a base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . ./

# Expose the port the app runs on
EXPOSE 8080

# Command to run the application
CMD exec gunicorn -b 0.0.0.0:8080 --workers 1 --threads 8 --timeout 0 app:app