# Running the Workshops: CPU and GPU

This document explains how to run the MLinCP workshop notebooks locally or via Docker.

PyTorch wheels are selected automatically based on your platform:

| Platform | Wheels installed by `uv sync` |
|----------|-------------------------------|
| Linux x86_64 (course server) | CUDA 12.4 — GPU-enabled |
| Windows / macOS x86_64 | CPU-only |
| Apple Silicon (M-series) | PyPI native — MPS-enabled |

---

## Prerequisites

- [UV](https://docs.astral.sh/uv/getting-started/installation/) — for local installs
- [Docker](https://docs.docker.com/get-docker/) + [Docker Compose](https://docs.docker.com/compose/install/) — for container-based runs
- For GPU use: an NVIDIA GPU with driver >= 525 (Linux only)

---

## Local (UV)

### Install and launch

```bash
uv sync
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

jupyter lab
```

`uv sync` picks the right PyTorch wheels for your platform automatically — no extra steps needed.

### Verify GPU availability (Linux x86_64)

```python
import torch
print(torch.cuda.is_available())    # True on the course server
print(torch.cuda.get_device_name(0))
```

### Verify MPS availability (Apple Silicon)

```python
import torch
print(torch.backends.mps.is_available())   # True on M-series Macs
```

**A note on MPS compatibility:**

Apple Silicon GPUs aren't compatible with CUDA code, and have a number of other limitations due to Apple's proprietary drivers. 
As a result, you won't be able to change `device="cuda"` to `device="mps"` 100% of the time and may run into errors that force you to run on CPU instead. 
(Luckily, you get pretty fast unified memory)


---

## Docker

### CPU image (default — for students)

Lightweight CPU-only image (~1 GB). Suitable for all machines.

```bash
docker compose up 
```

Open JupyterLab at [http://localhost:8888](http://localhost:8888).

To rebuild after dependency changes:

```bash
docker compose up --build
```

### GPU image (CUDA 12.4 — for the course server)

Full CUDA image (~5 GB). Requires an NVIDIA GPU with driver >= 525 and
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
installed on the host:

```bash
sudo systemctl restart docker   # after installing nvidia-container-toolkit
```

Start the GPU container:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

To rebuild:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

Open JupyterLab at [http://localhost:8888](http://localhost:8888). The kernel is named **Python (MLinCP GPU)**.

Verify GPU access from inside a notebook:

```python
import torch
print(torch.cuda.is_available())    # should print True
print(torch.cuda.get_device_name(0))
```

---

## Summary

| Setup | Command | PyTorch wheels | GPU required |
|-------|---------|----------------|--------------|
| Local (any platform) | `uv sync` | Auto-selected per platform | No |
| Docker CPU | `docker compose up` | CPU-only | No |
| Docker GPU | `docker compose -f docker-compose.yml -f docker-compose.gpu.yml up` | CUDA 12.4 | Yes |