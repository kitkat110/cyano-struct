# Pull base Python image
FROM python:3.12

# Set working directory inside container
WORKDIR /code

# Install Python dependencies (copy requirements first for Docker layer caching)
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# Install MAFFT alignment tool
RUN apt-get update && apt-get install -y mafft

# Copy all repo files into container
COPY . /code

# Make all Python scripts in src executable
RUN chmod ugo+x /code/src/*.py

# Allow scripts in src to run directly
ENV PATH="/code/src:$PATH"

# Default command to run pipeline
CMD ["python3", "src/main.py"]