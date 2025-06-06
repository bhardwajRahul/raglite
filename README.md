[![Open in Dev Containers](https://img.shields.io/static/v1?label=Dev%20Containers&message=Open&color=blue&logo=data:image/svg%2bxml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0iI2ZmZiIgZD0iTTE3IDE2VjdsLTYgNU0yIDlWOGwxLTFoMWw0IDMgOC04aDFsNCAyIDEgMXYxNGwtMSAxLTQgMmgtMWwtOC04LTQgM0gzbC0xLTF2LTFsMy0zIi8+PC9zdmc+)](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/superlinear-ai/raglite) [![Open in GitHub Codespaces](https://img.shields.io/static/v1?label=GitHub%20Codespaces&message=Open&color=blue&logo=github)](https://github.com/codespaces/new/superlinear-ai/raglite)

# 🥤 RAGLite

RAGLite is a Python toolkit for Retrieval-Augmented Generation (RAG) with DuckDB or PostgreSQL.

## Features

##### Configurable

- 🧠 Choose any LLM provider with [LiteLLM](https://github.com/BerriAI/litellm), including local [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) models
- 💾 Choose either [DuckDB](https://duckdb.org) or [PostgreSQL](https://github.com/postgres/postgres) as a keyword & vector search database
- 🥇 Choose any reranker with [rerankers](https://github.com/AnswerDotAI/rerankers), including multilingual [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) as the default

##### Fast and permissive

- ❤️ Only lightweight and permissive open source dependencies (e.g., no [PyTorch](https://github.com/pytorch/pytorch) or [LangChain](https://github.com/langchain-ai/langchain))
- 🚀 Acceleration with Metal on macOS, and CUDA on Linux and Windows

##### Unhobbled

- 📖 PDF to Markdown conversion on top of [pdftext](https://github.com/VikParuchuri/pdftext) and [pypdfium2](https://github.com/pypdfium2-team/pypdfium2)
- 🧬 Multi-vector chunk embedding with [late chunking](https://weaviate.io/blog/late-chunking) and [contextual chunk headings](https://d-star.ai/solving-the-out-of-context-chunk-problem-for-rag)
- ✏️ Optimal sentence splitting with [wtpsplit-lite](https://github.com/superlinear-ai/wtpsplit-lite) by solving a [binary integer programming problem](https://en.wikipedia.org/wiki/Integer_programming)
- ✂️ Optimal [semantic chunking](https://www.youtube.com/watch?v=8OJC21T2SL4&t=1930s) by solving a [binary integer programming problem](https://en.wikipedia.org/wiki/Integer_programming)
- 🔍 [Hybrid search](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) with the database's native keyword & vector search ([FTS](https://duckdb.org/docs/stable/extensions/full_text_search)+[VSS](https://duckdb.org/docs/stable/extensions/vss); [tsvector](https://www.postgresql.org/docs/current/datatype-textsearch.html)+[pgvector](https://github.com/pgvector/pgvector))
- 💭 [Adaptive retrieval](https://arxiv.org/abs/2403.14403) where the LLM decides whether to and what to retrieve based on the query
- 💰 Improved cost and latency with a [prompt caching-aware message array structure](https://platform.openai.com/docs/guides/prompt-caching)
- 🍰 Improved output quality with [Anthropic's long-context prompt format](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/long-context-tips)
- 🌀 Optimal [closed-form linear query adapter](src/raglite/_query_adapter.py) by solving an [orthogonal Procrustes problem](https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem)

##### Extensible

- 🔌 A built-in [Model Context Protocol](https://modelcontextprotocol.io) (MCP) server that any MCP client like [Claude desktop](https://claude.ai/download) can connect with
- 💬 Optional customizable ChatGPT-like frontend for [web](https://docs.chainlit.io/deploy/copilot), [Slack](https://docs.chainlit.io/deploy/slack), and [Teams](https://docs.chainlit.io/deploy/teams) with [Chainlit](https://github.com/Chainlit/chainlit)
- ✍️ Optional conversion of any input document to Markdown with [Pandoc](https://github.com/jgm/pandoc)
- ✅ Optional evaluation of retrieval and generation performance with [Ragas](https://github.com/explodinggradients/ragas)

## Installing

> [!TIP]
> 🚀 If you want to use local models, it is recommended to install [an accelerated llama-cpp-python precompiled binary](https://github.com/abetlen/llama-cpp-python?tab=readme-ov-file#supported-backends) with:
> ```sh
> # Configure which llama-cpp-python precompiled binary to install (⚠️ not every combination is available):
> LLAMA_CPP_PYTHON_VERSION=0.3.9
> PYTHON_VERSION=310|311|312
> ACCELERATOR=metal|cu121|cu122|cu123|cu124
> PLATFORM=macosx_11_0_arm64|linux_x86_64|win_amd64
> 
> # Install llama-cpp-python:
> pip install "https://github.com/abetlen/llama-cpp-python/releases/download/v$LLAMA_CPP_PYTHON_VERSION-$ACCELERATOR/llama_cpp_python-$LLAMA_CPP_PYTHON_VERSION-cp$PYTHON_VERSION-cp$PYTHON_VERSION-$PLATFORM.whl"
> ```

Install RAGLite with:

```sh
pip install raglite
```

To add support for a customizable ChatGPT-like frontend, use the `chainlit` extra:

```sh
pip install raglite[chainlit]
```

To add support for filetypes other than PDF, use the `pandoc` extra:

```sh
pip install raglite[pandoc]
```

To add support for evaluation, use the `ragas` extra:

```sh
pip install raglite[ragas]
```

## Using

### Overview

1. [Configuring RAGLite](#1-configuring-raglite)
2. [Inserting documents](#2-inserting-documents)
3. [Retrieval-Augmented Generation (RAG)](#3-retrieval-augmented-generation-rag)
4. [Computing and using an optimal query adapter](#4-computing-and-using-an-optimal-query-adapter)
5. [Evaluation of retrieval and generation](#5-evaluation-of-retrieval-and-generation)
6. [Running a Model Context Protocol (MCP) server](#6-running-a-model-context-protocol-mcp-server)
7. [Serving a customizable ChatGPT-like frontend](#7-serving-a-customizable-chatgpt-like-frontend)

### 1. Configuring RAGLite

> [!TIP]
> 🧠 RAGLite extends [LiteLLM](https://github.com/BerriAI/litellm) with support for [llama.cpp](https://github.com/ggerganov/llama.cpp) models using [llama-cpp-python](https://github.com/abetlen/llama-cpp-python). To select a llama.cpp model (e.g., from [Unsloth's collection](https://huggingface.co/unsloth)), use a model identifier of the form `"llama-cpp-python/<hugging_face_repo_id>/<filename>@<n_ctx>"`, where `n_ctx` is an optional parameter that specifies the context size of the model.

> [!TIP]
> 💾 You can create a PostgreSQL database in a few clicks at [neon.tech](https://neon.tech).

First, configure RAGLite with your preferred DuckDB or PostgreSQL database and [any LLM supported by LiteLLM](https://docs.litellm.ai/docs/providers/openai):

```python
from raglite import RAGLiteConfig

# Example 'remote' config with a PostgreSQL database and an OpenAI LLM:
my_config = RAGLiteConfig(
    db_url="postgresql://my_username:my_password@my_host:5432/my_database",
    llm="gpt-4o-mini",  # Or any LLM supported by LiteLLM
    embedder="text-embedding-3-large",  # Or any embedder supported by LiteLLM
)

# Example 'local' config with a DuckDB database and a llama.cpp LLM:
my_config = RAGLiteConfig(
    db_url="duckdb:///raglite.db",
    llm="llama-cpp-python/unsloth/Qwen3-8B-GGUF/*Q4_K_M.gguf@8192",
    embedder="llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512", # More than 512 tokens degrades bge-m3's performance
)
```

You can also configure [any reranker supported by rerankers](https://github.com/AnswerDotAI/rerankers):

```python
from rerankers import Reranker

# Example remote API-based reranker:
my_config = RAGLiteConfig(
    db_url="postgresql://my_username:my_password@my_host:5432/my_database"
    reranker=Reranker("rerank-v3.5", model_type="cohere", api_key=COHERE_API_KEY, verbose=0)  # Multilingual
)

# Example local cross-encoder reranker per language (this is the default):
my_config = RAGLiteConfig(
    db_url="duckdb:///raglite.db",
    reranker={
        "en": Reranker("ms-marco-MiniLM-L-12-v2", model_type="flashrank", verbose=0),  # English
        "other": Reranker("ms-marco-MultiBERT-L-12", model_type="flashrank", verbose=0),  # Other languages
    }
)
```

### 2. Inserting documents

> [!TIP]
> ✍️ To insert documents other than PDF, install the `pandoc` extra with `pip install raglite[pandoc]`.

Next, insert some documents into the database. RAGLite will take care of the [conversion to Markdown](src/raglite/_markdown.py), [optimal level 4 semantic chunking](src/raglite/_split_chunks.py), and [multi-vector embedding with late chunking](src/raglite/_embed.py):

```python
# Insert documents given their file path
from pathlib import Path
from raglite import Document, insert_documents

documents = [
    Document.from_path(Path("On the Measure of Intelligence.pdf")),
    Document.from_path(Path("Special Relativity.pdf")),
]
insert_documents(documents, config=my_config)

# Insert documents given their text/plain or text/markdown content
content = """
# ON THE ELECTRODYNAMICS OF MOVING BODIES
## By A. EINSTEIN  June 30, 1905
It is known that Maxwell...
"""
documents = [
    Document.from_text(content)
]
insert_documents(documents, config=my_config)
```

### 3. Retrieval-Augmented Generation (RAG)

#### 3.1 Adaptive RAG

Now you can run an adaptive RAG pipeline that consists of adding the user prompt to the message history and streaming the LLM response:

```python
from raglite import rag

# Create a user message
messages = []  # Or start with an existing message history
messages.append({
    "role": "user",
    "content": "How is intelligence measured?"
})

# Adaptively decide whether to retrieve and then stream the response
chunk_spans = []
stream = rag(messages, on_retrieval=lambda x: chunk_spans.extend(x), config=my_config)
for update in stream:
    print(update, end="")

# Access the documents referenced in the RAG context
documents = [chunk_span.document for chunk_span in chunk_spans]
```

The LLM will adaptively decide whether to retrieve information based on the complexity of the user prompt. If retrieval is necessary, the LLM generates the search query and RAGLite applies hybrid search and reranking to retrieve the most relevant chunk spans (each of which is a list of consecutive chunks). The retrieval results are sent to the `on_retrieval` callback and are appended to the message history as a tool output. Finally, the assistant response is streamed and appended to the message history.

#### 3.2 Programmable RAG

If you need manual control over the RAG pipeline, you can run a basic but powerful pipeline that consists of retrieving the most relevant chunk spans with hybrid search and reranking, converting the user prompt to a RAG instruction and appending it to the message history, and finally generating the RAG response:

```python
from raglite import add_context, rag, retrieve_context, vector_search

# Choose a search method
from dataclasses import replace
my_config = replace(my_config, search_method=vector_search)  # Or `hybrid_search`, `search_and_rerank_chunks`, ...

# Retrieve relevant chunk spans with the configured search method
user_prompt = "How is intelligence measured?"
chunk_spans = retrieve_context(query=user_prompt, num_chunks=5, config=my_config)

# Append a RAG instruction based on the user prompt and context to the message history
messages = []  # Or start with an existing message history
messages.append(add_context(user_prompt=user_prompt, context=chunk_spans))

# Stream the RAG response and append it to the message history
stream = rag(messages, config=my_config)
for update in stream:
    print(update, end="")

# Access the documents referenced in the RAG context
documents = [chunk_span.document for chunk_span in chunk_spans]
```

> [!TIP]
> 🥇 Reranking can significantly improve the output quality of a RAG application. To add reranking to your application: first search for a larger set of 20 relevant chunks, then rerank them with a [rerankers](https://github.com/AnswerDotAI/rerankers) reranker, and finally keep the top 5 chunks.

RAGLite also offers more advanced control over the individual steps of a full RAG pipeline:

1. Searching for relevant chunks with keyword, vector, or hybrid search
2. Retrieving the chunks from the database
3. Reranking the chunks and selecting the top 5 results
4. Extending the chunks with their neighbors and grouping them into chunk spans
5. Converting the user prompt to a RAG instruction and appending it to the message history
6. Streaming an LLM response to the message history
7. Accessing the cited documents from the chunk spans

A full RAG pipeline is straightforward to implement with RAGLite:

```python
# Search for chunks
from raglite import hybrid_search, keyword_search, vector_search

user_prompt = "How is intelligence measured?"
chunk_ids_vector, _ = vector_search(user_prompt, num_results=20, config=my_config)
chunk_ids_keyword, _ = keyword_search(user_prompt, num_results=20, config=my_config)
chunk_ids_hybrid, _ = hybrid_search(user_prompt, num_results=20, config=my_config)

# Retrieve chunks
from raglite import retrieve_chunks

chunks_hybrid = retrieve_chunks(chunk_ids_hybrid, config=my_config)

# Rerank chunks and keep the top 5 (optional, but recommended)
from raglite import rerank_chunks

chunks_reranked = rerank_chunks(user_prompt, chunks_hybrid, config=my_config)
chunks_reranked = chunks_reranked[:5]

# Extend chunks with their neighbors and group them into chunk spans
from raglite import retrieve_chunk_spans

chunk_spans = retrieve_chunk_spans(chunks_reranked, config=my_config)

# Append a RAG instruction based on the user prompt and context to the message history
from raglite import add_context

messages = []  # Or start with an existing message history
messages.append(add_context(user_prompt=user_prompt, context=chunk_spans))

# Stream the RAG response and append it to the message history
from raglite import rag

stream = rag(messages, config=my_config)
for update in stream:
    print(update, end="")

# Access the documents referenced in the RAG context
documents = [chunk_span.document for chunk_span in chunk_spans]
```

### 4. Computing and using an optimal query adapter

RAGLite can compute and apply an [optimal closed-form query adapter](src/raglite/_query_adapter.py) to the prompt embedding to improve the output quality of RAG. To benefit from this, first generate a set of evals with `insert_evals` and then compute and store the optimal query adapter with `update_query_adapter`:

```python
# Improve RAG with an optimal query adapter
from raglite import insert_evals, update_query_adapter

insert_evals(num_evals=100, config=my_config)
update_query_adapter(config=my_config)  # From here, every vector search will use the query adapter
```

### 5. Evaluation of retrieval and generation

If you installed the `ragas` extra, you can use RAGLite to answer the evals and then evaluate the quality of both the retrieval and generation steps of RAG using [Ragas](https://github.com/explodinggradients/ragas):

```python
# Evaluate retrieval and generation
from raglite import answer_evals, evaluate, insert_evals

insert_evals(num_evals=100, config=my_config)
answered_evals_df = answer_evals(num_evals=10, config=my_config)
evaluation_df = evaluate(answered_evals_df, config=my_config)
```

### 6. Running a Model Context Protocol (MCP) server

RAGLite comes with an [MCP server](https://modelcontextprotocol.io) implemented with [FastMCP](https://github.com/jlowin/fastmcp) that exposes a `search_knowledge_base` [tool](https://github.com/jlowin/fastmcp?tab=readme-ov-file#tools). To use the server:

1. Install [Claude desktop](https://claude.ai/download)
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) so that Claude desktop can start the server
3. Configure Claude desktop to use `uv` to start the MCP server with:

```
raglite \
    --db-url duckdb:///raglite.db \
    --llm llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192 \
    --embedder llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512 \
    mcp install
```

To use an API-based LLM, make sure to include your credentials in a `.env` file or supply them inline:

```sh
export OPENAI_API_KEY=sk-...
raglite \
    --llm gpt-4o-mini \
    --embedder text-embedding-3-large \
    mcp install
```

Now, when you start Claude desktop you should see a 🔨 icon at the bottom right of your prompt indicating that the Claude has successfully connected with the MCP server.

When relevant, Claude will suggest to use the `search_knowledge_base` tool that the MCP server provides. You can also explicitly ask Claude to search the knowledge base if you want to be certain that it does.

<div align="center"><video src="https://github.com/user-attachments/assets/3a597a17-874e-475f-a6dd-cd3ccf360fb9" /></div>

### 7. Serving a customizable ChatGPT-like frontend

If you installed the `chainlit` extra, you can serve a customizable ChatGPT-like frontend with:

```sh
raglite chainlit
```

The application is also deployable to [web](https://docs.chainlit.io/deploy/copilot), [Slack](https://docs.chainlit.io/deploy/slack), and [Teams](https://docs.chainlit.io/deploy/teams).

You can specify the database URL, LLM, and embedder directly in the Chainlit frontend, or with the CLI as follows:

```sh
raglite \
    --db-url duckdb:///raglite.db \
    --llm llama-cpp-python/unsloth/Qwen3-4B-GGUF/*Q4_K_M.gguf@8192 \
    --embedder llama-cpp-python/lm-kit/bge-m3-gguf/*F16.gguf@512 \
    chainlit
```

To use an API-based LLM, make sure to include your credentials in a `.env` file or supply them inline:

```sh
OPENAI_API_KEY=sk-... raglite --llm gpt-4o-mini --embedder text-embedding-3-large chainlit
```

<div align="center"><video src="https://github.com/user-attachments/assets/a303ed4a-54cd-45ea-a2b5-86e086053aed" /></div>

## Contributing

<details>
<summary>Prerequisites</summary>

1. [Generate an SSH key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent#generating-a-new-ssh-key) and [add the SSH key to your GitHub account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account).
1. Configure SSH to automatically load your SSH keys:

    ```sh
    cat << EOF >> ~/.ssh/config
    
    Host *
      AddKeysToAgent yes
      IgnoreUnknown UseKeychain
      UseKeychain yes
      ForwardAgent yes
    EOF
    ```

1. [Install Docker Desktop](https://www.docker.com/get-started).
1. [Install VS Code](https://code.visualstudio.com/) and [VS Code's Dev Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers). Alternatively, install [PyCharm](https://www.jetbrains.com/pycharm/download/).
1. _Optional:_ install a [Nerd Font](https://www.nerdfonts.com/font-downloads) such as [FiraCode Nerd Font](https://github.com/ryanoasis/nerd-fonts/tree/master/patched-fonts/FiraCode) and [configure VS Code](https://github.com/tonsky/FiraCode/wiki/VS-Code-Instructions) or [PyCharm](https://github.com/tonsky/FiraCode/wiki/Intellij-products-instructions) to use it.

</details>

<details open>
<summary>Development environments</summary>

The following development environments are supported:

1. ⭐️ _GitHub Codespaces_: click on [Open in GitHub Codespaces](https://github.com/codespaces/new/superlinear-ai/raglite) to start developing in your browser.
1. ⭐️ _VS Code Dev Container (with container volume)_: click on [Open in Dev Containers](https://vscode.dev/redirect?url=vscode://ms-vscode-remote.remote-containers/cloneInVolume?url=https://github.com/superlinear-ai/raglite) to clone this repository in a container volume and create a Dev Container with VS Code.
1. ⭐️ _uv_: clone this repository and run the following from root of the repository:

    ```sh
    # Create and install a virtual environment
    uv sync --python 3.10 --all-extras

    # Activate the virtual environment
    source .venv/bin/activate

    # Install the pre-commit hooks
    pre-commit install --install-hooks
    ```

1. _VS Code Dev Container_: clone this repository, open it with VS Code, and run <kbd>Ctrl/⌘</kbd> + <kbd>⇧</kbd> + <kbd>P</kbd> → _Dev Containers: Reopen in Container_.
1. _PyCharm Dev Container_: clone this repository, open it with PyCharm, [create a Dev Container with Mount Sources](https://www.jetbrains.com/help/pycharm/start-dev-container-inside-ide.html), and [configure an existing Python interpreter](https://www.jetbrains.com/help/pycharm/configuring-python-interpreter.html#widget) at `/opt/venv/bin/python`.

</details>

<details open>
<summary>Developing</summary>

- This project follows the [Conventional Commits](https://www.conventionalcommits.org/) standard to automate [Semantic Versioning](https://semver.org/) and [Keep A Changelog](https://keepachangelog.com/) with [Commitizen](https://github.com/commitizen-tools/commitizen).
- Run `poe` from within the development environment to print a list of [Poe the Poet](https://github.com/nat-n/poethepoet) tasks available to run on this project.
- Run `uv add {package}` from within the development environment to install a run time dependency and add it to `pyproject.toml` and `uv.lock`. Add `--dev` to install a development dependency.
- Run `uv sync --upgrade` from within the development environment to upgrade all dependencies to the latest versions allowed by `pyproject.toml`. Add `--only-dev` to upgrade the development dependencies only.
- Run `cz bump` to bump the package's version, update the `CHANGELOG.md`, and create a git tag. Then push the changes and the git tag with `git push origin main --tags`.

</details>

## Star History

<a href="https://star-history.com/#superlinear-ai/raglite&Timeline">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=superlinear-ai/raglite&type=Timeline&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=superlinear-ai/raglite&type=Timeline" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=superlinear-ai/raglite&type=Timeline" />
 </picture>
</a>
