# ============================================================
# MLinCP — Machine Learning in Computational Physics
# Docker image for workshop notebooks
# ============================================================
# Build:  docker build -t mlincp .
# Run:    docker run -p 8888:8888 -v $(pwd):/workspace/notebooks mlincp
#         docker compose up          (preferred — see docker-compose.yml)
# ============================================================

FROM python:3.12.3-slim

# ── System build dependencies ─────────────────────────────────
# Required by: dscribe, scipy (OpenBLAS), aseMolec
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gfortran \
    libopenblas-dev \
    git \
    curl \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# ── Install UV ────────────────────────────────────────────────
# Copy the UV binary from its official minimal image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# ── Project dependency installation ──────────────────────────
WORKDIR /opt/mlincp

# Copy the dependency manifest and lock file. Copying both together means
# Docker only invalidates this layer when either file changes.
# uv.lock pins all transitive dependencies for a fully reproducible build.
COPY pyproject.toml ./
COPY uv.lock ./

# Install all dependencies from the lock file.
# UV_TORCH_BACKEND=cpu overrides the pytorch-cu124 index source declared in
# pyproject.toml so the container gets lightweight CPU-only wheels even though
# it runs on Linux x86_64 (where CUDA wheels would otherwise be selected).
RUN UV_TORCH_BACKEND=cpu uv sync --no-dev --frozen

# ── Make the venv the default Python ─────────────────────────
ENV VIRTUAL_ENV="/opt/mlincp/.venv"
ENV PATH="/opt/mlincp/.venv/bin:$PATH"

# ── Register the venv as a named Jupyter kernel ───────────────
# Without this, VS Code's Jupyter extension discovers the system
# Python (which has no packages) and fails to find ipykernel.
# Registering explicitly writes a kernelspec into the global
# Jupyter kernels directory so both JupyterLab and VS Code find it.
RUN /opt/mlincp/.venv/bin/python -m ipykernel install \
    --name mlincp \
    --display-name "Python (MLinCP)"

# ── Workspace layout ──────────────────────────────────────────
# Notebooks are mounted at runtime; this dir just needs to exist.
RUN mkdir -p /workspace/notebooks
WORKDIR /workspace/notebooks

# ── JupyterLab ────────────────────────────────────────────────
EXPOSE 8888

# No token/password — access is local only (bound to 127.0.0.1 in compose).
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=", \
     "--ServerApp.password=", \
     "--ServerApp.root_dir=/workspace/notebooks"]
