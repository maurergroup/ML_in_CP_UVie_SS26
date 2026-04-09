# Getting Started

Welcome to **Machine Learning in Computational Physics (MLinCP)**!

This guide will get your laptop ready to run the workshop notebooks.  
We use a **Docker container** so that every student has an identical Python environment regardless of operating system.

---

## What you need to install (once)

| Tool | Purpose | Download |
|------|---------|----------|
| **Docker Desktop** | Runs the container | https://www.docker.com/products/docker-desktop |
| **VS Code** | Edit and run notebooks | https://code.visualstudio.com |
| **Git** | Clone the repository | https://git-scm.com |
| **Julia** | Required for the ACE descriptor (Module II) | https://julialang.org/downloads/ |

> **Julia note:** The recommended way to install Julia is via **juliaup**, the official Julia version manager.  
> On macOS/Linux run `curl -fsSL https://install.julialang.org | sh`; on Windows run `winget install julia -s msstore` or download the installer from the link above.

After installing VS Code, add the **Dev Containers** extension:

1. Open VS Code → click the Extensions icon (or press `Ctrl+Shift+X` / `Cmd+Shift+X`)
2. Search for **Dev Containers** (publisher: Microsoft)
3. Click **Install**

---

## Step 1 — Clone the repository

Open a terminal and run:

```bash
git clone https://github.com/maurergroup/ML_in_CP_UVie_SS26.git
cd MLinCP
```
(lecturers: you have to clone from the private course repo)

---

## Step 2 — Open in a Dev Container (recommended)

This is the easiest way to run the notebooks. VS Code builds the container and connects to it automatically.

1. Open VS Code.
2. Choose **File → Open Folder** and select the `MLinCP` folder you just cloned.
3. VS Code will detect the `.devcontainer/` folder and show a pop-up:  
   **"Reopen in Container"** — click it.  
   *(If the pop-up doesn't appear: press `F1` → type **Reopen in Container** → press Enter.)*
4. Wait for the container to build. **The first build takes 5–15 minutes** because it downloads and installs all Python packages and Julia dependencies. Subsequent starts are instant.
5. Once connected, open any `.ipynb` notebook from the Explorer panel and select the Python kernel when prompted.

> **ACE descriptor (Module II):** Julia and ACEpotentials.jl are pre-installed inside the container — no extra steps needed. The first time a notebook cell calls Julia you will see a ~30–60 s delay while Julia compiles; this is normal.

> **Tip:** The bar at the bottom-left of VS Code shows `Dev Container: MLinCP` when you are inside the container.

---

## Alternative — Run JupyterLab in a browser

If you prefer the classic Jupyter interface instead of VS Code:

### Option A — Docker Compose (simplest)

```bash
# From the MLinCP folder:
docker compose up
```

Open your browser and go to **http://localhost:8888**.  
Stop the server with `Ctrl+C` in the terminal, then `docker compose down`.

> **ACE descriptor (Module II):** Julia and ACEpotentials.jl are pre-installed in the container image. No extra steps needed; the first Julia call per session will take ~30–60 s to compile.

### Option B — Plain Docker

```bash
docker build -t mlincp .

# macOS / Linux
docker run -p 127.0.0.1:8888:8888 -v $(pwd):/workspace/notebooks mlincp

# Windows PowerShell
docker run -p 127.0.0.1:8888:8888 -v ${PWD}:/workspace/notebooks mlincp
```

Open **http://localhost:8888** in your browser.

---

## Alternative — Local installation without Docker

If you are comfortable with Python environments and do not want to use Docker,
you can install all dependencies directly using [UV](https://docs.astral.sh/uv/).

### Install UV

```bash
# macOS / Linux
curl -Ls https://astral.sh/uv/install.sh | sh

# Windows (PowerShell) #TODO Can't keep continuing to support Windows. the whole point of Docker is to avoid this.
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Create the environment and install packages

```bash
cd MLinCP
uv sync          # reads pyproject.toml, creates .venv, installs everything
```

### Activate the environment

```bash
# macOS / Linux
source .venv/bin/activate

# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

### Install Julia and ACEpotentials.jl

The `juliacall` Python package (already included in the UV environment) bridges Python and Julia, but Julia itself must be installed separately.

**1. Install Julia via juliaup:**

```bash
# macOS / Linux
curl -fsSL https://install.julialang.org | sh

# Windows (PowerShell)
winget install julia -s msstore
```

Restart your terminal after installation, then verify with `julia --version`.

**2. Install ACEpotentials.jl inside Julia:**

```bash
julia -e 'using Pkg; Pkg.add("ACEpotentials")'
```

This only needs to be done once. ACEpotentials.jl will be available to `juliacall` automatically because `juliacall` uses the default Julia environment.

> **First-call latency:** When a notebook first calls into Julia (via `juliacall`), expect a 30–60 s pause while Julia compiles. Subsequent calls in the same kernel session are fast.

---

### If you want to run jupyter notebooks within VSCode

Click on the Kernel Selector in the top right corner and go to "Python Environments" -> "Create Python Environment" -> "Venv (Creates a `.venv` virtual environment in the current workspace)" -> "Use Existing"

If you cannot find the kernel or you can find it but it does not work, then maybe you are mixing Windows and WSL. If you have so far done everything in the WSL, run VSCode out of the WSL (type `code .` in the WSL).

### If you want to run the notebooks in the browser, start JupyterLab

```bash
jupyter lab
```

---

## Keeping your environment up to date

When new packages are added during the course, re-sync:

```bash
# Inside the container (VS Code terminal):
uv sync

# If using Docker Compose:
docker compose up --build

# If using local UV:
uv sync
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Docker Desktop not running | Start Docker Desktop from your Applications / Start menu before opening VS Code |
| "Reopen in Container" pop-up doesn't appear | Press `F1` → type **Reopen in Container** |
| Kernel not found in notebook | Click the kernel selector (top-right of notebook) → choose **Python 3 (/opt/mlincp/.venv)** |
| Build fails with a network error | Check your internet connection; corporate firewalls sometimes block Docker Hub — try on a different network |
| Slow first build | Normal — Docker is downloading ~2 GB of packages. Subsequent starts are fast |
| ACE cell hangs for ~60 s | Normal — Julia is JIT-compiling ACEpotentials on first use. Wait it out; subsequent calls in the same session are fast |
| `juliacall` can't find Julia (local install) | Make sure `julia` is on your PATH (`julia --version` should work in your terminal). If not, re-run the juliaup installer and restart your terminal |
| `ACEpotentials` not found in Julia | Run `julia -e 'using Pkg; Pkg.add("ACEpotentials")'` in your terminal, then restart the notebook kernel |

---

## Getting help

Raise questions during the workshop or open an issue on the GitHub repository.
