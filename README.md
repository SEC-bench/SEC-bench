<div align="center">
  <h1>SEC-bench</h1>
  <p><strong>Automated Benchmarking of LLM Agents on Real-World Software Security Tasks</strong></p>
  <p align="center">
    <a href="https://hwiwonl.ee/" style="text-decoration: none;">Hwiwon Lee<sup>1</sup></a>,
    <a href="https://ziqi-zhang.github.io/" style="text-decoration: none;">Ziqi Zhang<sup>1</sup></a>,
    <a href="" style="text-decoration: none;">Hanxiao Lu<sup>2</sup></a>,
    <a href="https://lingming.cs.illinois.edu/" style="text-decoration: none;">Lingming Zhang<sup>1</sup></a>
  </p>
</div>

<p align="center">
  <sup>1</sup>UIUC, <sup>2</sup>Purdue University
</p>

<p align="center">
<a href="https://arxiv.org/abs/2506.11791" >📄 Paper</a>
<a href="https://sec-bench.github.io" >📊 Leaderboard</a>
<a href="https://hf.co/datasets/SEC-bench/SEC-bench" >🤗 Dataset</a>
</p>

---

## 🎯 Overview

SEC-bench is a comprehensive benchmarking framework designed to evaluate Large Language Model (LLM) agents on real-world software security tasks. It provides automated tools for collecting vulnerability data, building reproducible vulnerability instances, and evaluating agent performance on security-related tasks.

## ✨ Features

- **🔍 Automated Benchmark Generation**: Automated benchmark generation from OSV database and CVE records by using multi-agentic system
- **🐳 Containerized Environments**: Docker-based reproducible vulnerability instances
- **🤖 Agent-oriented Evaluation**: Evaluate agents on critical software security tasks (SWE-agent, OpenHands, Aider, and smolagents are supported)
- **📊 Comprehensive Security Assessment**: Both PoC generation and vulnerability patching assessment with extensibility to other tasks (e.g., fuzzing, static analysis, etc.)
- **📈 Rich Reporting**: Detailed progress tracking and result visualization with rich terminal output

---

## 🔧 Prerequisites

- **Python**: 3.12 or higher
- **Docker**: Latest version with sufficient disk space (>200GB recommended)
- **Git**: For repository cloning and submodule management
- **Conda**: For environment management (recommended)

---

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone --recurse-submodules https://github.com/SEC-bench/SEC-bench.git
cd SEC-bench
```

### 2. Create Python Environment

```bash
conda create -n secb python=3.12
conda activate secb
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Setup

Configure the following environment variables in your shell profile or `.env` file:

```bash
# Required API tokens
export GITHUB_TOKEN=<your_github_token>
export GITLAB_TOKEN=<your_gitlab_token>
export OPENAI_API_KEY=<your_openai_api_key>
export ANTHROPIC_API_KEY=<your_anthropic_api_key>

# Hugging Face configuration
export HF_TOKEN_PATH=$HOME/.cache/hf_hub_token
export HF_HOME=<path/to/huggingface>
```

---

## 📖 Usage

### 🗂️ Data Collection

The data collection process involves three main steps: seed generation, report extraction, and project configuration.

#### Step 1: Seed Generation
Extract metadata from OSV database files:

```bash
python -m secb.preprocessor.seed \
    --input-dir [OSV_DIR] \
    --output-file [SEED_OUTPUT_FILE_PATH] \
    --verbose
```

#### Step 2: Report Extraction
Extract bug reports from reference URLs:

```bash
python -m secb.preprocessor.report \
    --input-file [SEED_OUTPUT_FILE_PATH] \
    --output-file [REPORT_OUTPUT_FILE_PATH] \
    --reports-dir [REPORTS_DIR] \
    --lang [LANGUAGE] \
    --type [TYPE] \
    --whitelist [WHITELIST_PROJECTS] \
    --blacklist [BLACKLIST_PROJECTS] \
    --oss-fuzz
```

#### Step 3: Project Configuration
Generate project configurations for vulnerability reproduction:

```bash
python -m secb.preprocessor.project \
    --input-file [REPORT_OUTPUT_FILE_PATH] \
    --output-file [PROJECT_OUTPUT_FILE_PATH] \
    --tracking-file [TRACKING_FILE_PATH] \
    --verbose
```

#### 🛠️ Simplified Collection with Script

Use the provided script for streamlined processing:

```bash
./run_preprocessor.sh <mode> [options]
```

**Available modes:**
- `seed`: Parse CVE/OSV files and extract relevant information
- `report`: Extract bug descriptions from reference URLs  
- `project`: Generate project configurations for reproducing vulnerabilities

**Example workflows:**

> [!NOTE]
> Download the [OSV database](https://google.github.io/osv.dev/data/) and place it in the `output/osv` directory.
> The following example workflows are for C/C++ vulnerabilities. For other languages, you need to specify the language and type.

```bash
# Basic seed generation
./run_preprocessor.sh seed --input-dir ./output/osv --output-file ./output/seed.jsonl

# Filter for C/C++ CVEs in OSS-Fuzz projects
./run_preprocessor.sh report \
    --input-file ./output/seed.jsonl \
    --type CVE \
    --oss-fuzz \
    --lang C,C++

# Generate minimal project configurations
./run_preprocessor.sh project \
    --input-file ./output/report-cve-oss-c-cpp.jsonl \
    --sanitizer-only \
    --minimal
```

### 🏗️ Instance Building

#### Build Base Images
Create foundational Docker images:

```bash
python -m secb.preprocessor.build_base_images
```

#### Build Instance Images
Create specific vulnerability instances:

```bash
# Build specific instance
python -m secb.preprocessor.build_instance_images \
    --input-file [PROJECT_OUTPUT_FILE] \
    --ids [INSTANCE_IDS]

# Example: Build OpenJPEG CVE instance
python -m secb.preprocessor.build_instance_images \
    --input-file ./output/project-cve-oss-c-cpp-sanitizer-minimal.jsonl \
    --ids openjpeg.cve-2024-56827

# Example: Build all GPAC CVE instances
python -m secb.preprocessor.build_instance_images \
    --input-file ./output/project-cve-oss-c-cpp-sanitizer-minimal.jsonl \
    --filter gpac.cve
```

### ✅ Verification

Verify built instances using the [SecVerifier](https://github.com/SEC-bench/SecVerifier) repository. This step ensures that vulnerability instances are correctly configured and reproducible.

### 🧪 Evaluation

#### Option 1: Use Pre-built Images

Access verified evaluation images from [Docker Hub](https://hub.docker.com/search?q=hwiwonlee%2Fsecb.eval.x86_64):

```bash
docker pull hwiwonlee/secb.eval.x86_64.[instance_name]
```

#### Option 2: Build Evaluation Images

```bash
python -m secb.evaluator.build_eval_instances \
    --input-dir [VERIFIED_INSTANCE_DIR]
```

#### Run Evaluation

```bash
python -m secb.evaluator.eval_instances \
    --input-dir [AGENT_OUTPUT_DIR] \
    --type [TYPE] \
    --split [SPLIT] \
    --agent [AGENT] \
    --num-workers [NUM_WORKERS] \
    --output-dir [OUTPUT_DIR]
```

**Parameters:**
- `type`: Evaluation type (`patch` or `poc`)
- `split`: Dataset split to evaluate
- `agent`: Agent type (`swea`, `oh`, `aider`, `smolagent`)
- `num-workers`: Number of parallel workers

### 🤖 Running smolagent

[smolagents](https://github.com/SEC-bench/smolagents) is a lightweight agent framework from Hugging Face that supports diverse scaffolds, such as `CodeAgent` and `ToolCallingAgent`. We extended it for SEC-bench evaluation. Once installed, you can run evaluations with a configuration file.

#### Configuration

Copy `config.example.toml` and customize it for your setup:

```bash
cp config.example.toml config.toml
```

Edit `config.toml` to configure:
- **Model**: Set your model type, ID, and API keys
- **Agent**: Choose `CodeAgent` or `ToolCallingAgent`, set max steps and tools
- **Dataset**: Specify dataset split (`eval`, `cve`, or `oss`) and instance IDs
- **Task**: Select task type (`patch`, `poc-repo`, `poc-desc`, or `poc-san`)
- **Docker**: Configure Docker image prefix and runtime settings

#### PoC Generation

Generate proof-of-concept exploits for three scenarios:

**1. PoC with Repository Only (`poc-repo`)**
```bash
# Set task.type = "poc-repo" in config.toml
smolagent secb-run --config config.toml
```

**2. PoC with Repository and Bug Description (`poc-desc`)**
```bash
# Set task.type = "poc-desc" in config.toml
smolagent secb-run --config config.toml
```

**3. PoC with Repository, Bug Description, and Sanitizer Report (`poc-san`)**
```bash
# Set task.type = "poc-san" in config.toml
smolagent secb-run --config config.toml
```

#### Vulnerability Patching

Generate patches for vulnerabilities:

```bash
# Set task.type = "patch" in config.toml
smolagent secb-run --config config.toml
```

#### Additional Options

```bash
# Run with custom output directory
smolagent secb-run --config config.toml --output-dir ./results

# Run with parallel workers
smolagent secb-run --config config.toml --num-workers 4

# Run for a specific instance
smolagent secb-run --config config.toml --instance-id wasm3.ossfuzz-42496369
```

#### Evaluating smolagent Results

After running smolagent, evaluate the results using the standard evaluation command:

```bash
# For PoC tasks
python -m secb.evaluator.eval_instances \
    --input-dir ./results/[timestamp] \
    --type poc \
    --split eval \
    --agent smolagent \
    --output-dir ./output/eval/poc

# For patch tasks
python -m secb.evaluator.eval_instances \
    --input-dir ./results/[timestamp] \
    --type patch \
    --split eval \
    --agent smolagent \
    --output-dir ./output/eval/patch
```

### 📊 Results Viewing

#### Patch Results
```bash
python -m secb.evaluator.view_patch_results \
    --agent [AGENT] \
    --input-dir [EVALUATION_OUTPUT_DIR]
```

#### PoC Results
```bash
python -m secb.evaluator.view_poc_results \
    --agent [AGENT] \
    --input-dir [EVALUATION_OUTPUT_DIR]
```

---

## 🐳 Docker Images

> [!NOTE]
> In the Docker evaluation images, you can check a harness `secb` with options such as build, repro, and patch.

SEC-bench uses a hierarchical Docker image structure:

- **Base Images**: `hwiwonlee/secb.base:*` - Foundation images with build tools
- **Instance Images**: `hwiwonlee/secb.x86_64.*` - Vulnerability-specific environments  
- **Evaluation Images**: `hwiwonlee/secb.eval.x86_64.*` - Verified evaluation instances

### Image Naming Convention

```
hwiwonlee/secb.eval.x86_64.[project].[vulnerability_id]
```

Example: `hwiwonlee/secb.eval.x86_64.mruby.cve-2022-0240`

---

## 📚 Citation

If you use SEC-bench in your research, please cite our paper:

```bibtex
@inproceedings{lee2025secbench,
  author    = {Hwiwon Lee and Ziqi Zhang and Hanxiao Lu and Lingming Zhang},
  booktitle = {The Thirty-ninth Annual Conference on Neural Information Processing Systems},
  title     = {{SEC-bench: Automated Benchmarking of LLM Agents on Real-World Software Security Tasks}},
  url       = {https://openreview.net/forum?id=QQhQIqons0},
  year      = {2025}
}
```
