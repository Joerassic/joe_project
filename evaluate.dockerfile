# ----- To run python ----- #

# Base image
FROM python:3.11-slim

# Install Python
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*


# ----- To run the app ----- #

# Copy the application
COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY src/ src/
COPY data/ data/
COPY models/ models/
COPY reports/ reports/

# Setting working directory in the container
WORKDIR /

# Commands to install dependencies
# first run: 
# RUN pip install -r requirements.txt --no-cache-dir
# otherwise next runs: 
RUN --mount=type=cache,target=~/pip/.cache pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

# The application we want to run
ENTRYPOINT ["python", "-u", "src/joe_project/evaluate.py"]
