# Pull base Python image from web
FROM python:3.12

# Install Python dependencies
COPY requirements.txt
RUN pip3 install -r requirements.txt

# Set working directory inside container
WORKDIR /code

# Install alignment tool
RUN apt-get update && apt-get install -y mafft

# Copy all repo files into container
COPY . /code

# Make all Python scripts in src executable
RUN chmod ugo+x /code/src/*.py

# Allow scripts in src to run directly
ENV PATH="/code/src:$PATH"

# Default command to run script
CMD ["python3", "src/main.py"]