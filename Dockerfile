# Base IMAGE
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages
RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime && \
    pip install -r requirements.txt

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run server.py when the container launches
CMD ["python", "server.py"]