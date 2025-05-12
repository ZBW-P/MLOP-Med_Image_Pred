# Use a lightweight Python 3.9 image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy everything from the current directory into the container
COPY . .

# Install required dependencies
RUN pip install --no-cache-dir torch torchvision pillow requests

# Set the default command to run the simulation script
CMD ["python3", "simulate_requests_3.py"]
