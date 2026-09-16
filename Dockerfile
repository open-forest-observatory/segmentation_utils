# Base python image
FROM python:3.9-slim

# Install git, gcc/g++, and curl for downloading and compiling
# Also install libgl1 and libxrender1, needed by scikit-image/matplotlib/Pillow
RUN apt-get update && \
    apt-get install -y \
    git \
    build-essential \
    curl \
    libgl1 \
    libxrender1 && \
    rm -rf /var/lib/apt/lists/*

# Install poetry (pinned to a version that still supports Python 3.9)
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:${PATH}"

# Set the container workdir
WORKDIR /app
# Copy files from current directory into /app
COPY . /app

# Install the module dependencies with poetry without creating a virtual environment
RUN poetry config virtualenvs.create false && poetry install

CMD ["/bin/bash"]
