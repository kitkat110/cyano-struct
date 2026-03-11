# Use Python 3.12 image
FROM python:3.12

# Set working directory inside container
WORKDIR /code

# Install alignment tool
RUN apt-get update && apt-get install -y mafft

# Install Python dependency
RUN pip3 install biopython pydantic

# Copy all repo files into container
COPY . /code

# Make all Python scripts in src executable
RUN chmod ugo+x /code/src/*.py

ENV PATH="/code/src:$PATH"

# Default command to run script
CMD ["python", "src/seq_alignment.py"]