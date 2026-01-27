# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM pytorch/pytorch:2.10.0-cuda13.0-cudnn9-runtime

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
