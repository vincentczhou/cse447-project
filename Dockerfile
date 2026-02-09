# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

# Install kenlm
RUN apt-get update && apt-get install -y \
    build-essential \
    libboost-all-dev \
    cmake \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*
RUN git clone https://github.com/kpu/kenlm /opt/kenlm
RUN mkdir -p /opt/kenlm/build && cd /opt/kenlm/build \
    && cmake .. \
    && make -j $(nproc) \
    && make install

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9.27 /uv /uvx /bin/

RUN mkdir /job
WORKDIR /job

# Copy project files and install dependencies
COPY pyproject.toml uv.lock .python-version ./
# Make sure torch is still visible in the venv
RUN uv venv --system-site-packages --clear
RUN uv sync --frozen --no-cache --no-dev
# RUN uv sync --frozen --no-cache

VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
