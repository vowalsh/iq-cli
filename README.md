<div align="center">

# 🧠 IQ

**Intelligent Question answering for your terminal**

Real-time web search powered by AI • Beautiful CLI experience • Smart caching

[Installation](#installation) • [Quick Start](#quick-start) • [Features](#features) • [Documentation](#documentation)

</div>

---

## Overview

IQ combines live web search with LLM synthesis to deliver comprehensive, cited answers directly in your terminal. Think Perplexity, but in your CLI.

```bash
$ iq "What are the latest developments in quantum computing?"
```

## Installation

```bash
# Clone the repository
git clone https://github.com/vowalsh/iq-cli.git
cd iq-cli

# Install dependencies
pip install -r requirements.txt

# Install globally
pip install -e .

# Set up API keys
cp .env.example .env
# Edit .env with your API keys
```

### API Keys Required

- **SerpAPI** - [Get free key](https://serpapi.com/)
- **LLM Provider** (choose one):
  - **OpenAI** - [Get API key](https://platform.openai.com/api-keys)
  - **Anthropic Claude** - [Get API key](https://console.anthropic.com/)
  - **Google Gemini** - [Get API key](https://makersuite.google.com/app/apikey)
  - **OpenRouter** - [Get API key](https://openrouter.ai/keys) (access multiple models)
  - **Perplexity** - [Get API key](https://www.perplexity.ai/settings/api)
  - **Ollama** - [Install locally](https://ollama.ai) (no API key needed, runs locally)

## Quick Start

```bash
# Interactive mode
iq

# Ask a question
iq "What is the capital of France?"

# Random interesting question
iq --random

# Use cached answers
iq --use-cache "Python best practices"

# Get help
iq --help
```

## Features

### 🔍 **Live Web Search**
Real-time results using SerpAPI with configurable result counts

### 🤖 **AI-Powered Synthesis**
Supports 6 LLM providers: OpenAI, Claude, Gemini, OpenRouter, Perplexity, Ollama (local) with streaming

### 📊 **Visual Data Charts**
Automatic chart generation for numeric/statistical queries

### 💾 **Smart Caching**
- Intelligent similarity matching (95%+ threshold)
- Automatic Q&A storage with timestamps
- Full-text search across cached content
- 30-day expiration (configurable)

### 🎨 **Beautiful Interface**
- Rich terminal formatting with colors
- Organized help system
- Interactive mode with command support
- Inline citations [1], [2], [3]...

### ⚡ **Fast & Lightweight**
Minimal dependencies, maximum performance

## Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `iq` | Start interactive mode |
| `iq "query"` | Ask a single question |
| `iq --help` | Show help screen |
| `iq --version` | Show version |
| `iq --random` | Get random question |

### Cache Management

| Command | Description |
|---------|-------------|
| `iq --cache-list` | List recent Q&As |
| `iq --cache-search "term"` | Search cache |
| `iq --cache-stats` | Show statistics |
| `iq --cache-clear` | Clear all entries |
| `iq --use-cache` | Enable retrieval |
| `iq --force-refresh` | Skip cache |

### Options

| Flag | Description |
|------|-------------|
| `--results N` | Number of search results (default: 8) |
| `--model MODEL` | LLM model (gpt-3.5-turbo, gpt-4, claude-3-sonnet, etc.) |
| `--no-streaming` | Show complete answer at once |
| `--no-cache` | Disable caching |
| `--no-color` | Disable colored output |
| `--verbose` | Enable debug output |

## Interactive Mode

When you run `iq` without arguments, you enter interactive mode:

```
❓ Your question: /cache list          # List recent entries
❓ Your question: /cache search python # Search for "python"
❓ Your question: /cache stats         # Show statistics
❓ Your question: quit                 # Exit
```

## Documentation

### Cache System

**Location**: `~/.iq_cache/qa_cache.json`

**Features**:
- Automatic storage of all Q&A pairs
- Fuzzy matching for similar questions
- Configurable expiration (30 days)
- Size limit (1000 entries)
- Automatic cleanup

**Behavior**:
- Storage: Always enabled (unless `--no-cache`)
- Retrieval: Requires `--use-cache` flag
- Override: Use `--force-refresh` for fresh results

### Visual Charts

IQ automatically generates charts for queries with:
- Percentages (market share, growth rates)
- Currency values (GDP, revenue)
- Statistical data (populations, rankings)
- Comparative numbers

Example queries:
```bash
iq "Compare GDP of G7 countries"
iq "Smartphone market share by company"
iq "Top programming languages by popularity"
```

### Architecture

```
iq.py          # Main CLI interface
search.py      # SerpAPI integration
llm.py         # OpenAI synthesis
formatting.py  # Terminal output
cache.py       # Q&A caching system
charts.py      # Data visualization
```

## Configuration

Edit `.env` with your API keys:

```bash
# Required
SERPAPI_KEY=your_serpapi_key_here

# LLM Provider (choose one or more)
OPENAI_API_KEY=your_openai_api_key_here        # For GPT models
ANTHROPIC_API_KEY=your_anthropic_api_key_here  # For Claude models
GOOGLE_API_KEY=your_google_api_key_here        # For Gemini models
OPENROUTER_API_KEY=your_openrouter_api_key     # Access to multiple models
PERPLEXITY_API_KEY=your_perplexity_api_key     # For Perplexity models

# Optional: Set default model (defaults to gpt-3.5-turbo)
IQ_MODEL=gpt-3.5-turbo
```

### Supported Models

**OpenAI:**
- `gpt-3.5-turbo` (default, fast & cost-effective)
- `gpt-4` (most capable)
- `gpt-4-turbo`

**Anthropic Claude:**
- `claude-3-haiku-20240307` (fastest)
- `claude-3-sonnet-20240229` (balanced)
- `claude-3-opus-20240229` (most capable)

**Google Gemini:**
- `gemini-pro` (fast & capable)
- `gemini-1.5-pro` (most capable)
- `gemini-1.5-flash` (fastest)

**OpenRouter** (via `--model` flag with `provider=openrouter`):
- `anthropic/claude-3-sonnet`
- `meta-llama/llama-3-70b-instruct`
- `google/gemini-pro`
- Many more at [openrouter.ai/models](https://openrouter.ai/models)

**Perplexity:**
- `pplx-7b-online` (fast, with web access)
- `pplx-70b-online` (most capable, with web access)

**Ollama** (local models, no API key needed):
- `llama2` - Meta's Llama 2
- `mistral` - Mistral 7B
- `mixtral` - Mixtral 8x7B
- `codellama` - Code Llama
- `phi` - Microsoft Phi
- Install from [ollama.ai](https://ollama.ai)

## Requirements

- Python 3.7+
- Internet connection
- SerpAPI account (free tier available)
- OpenAI API account

**Dependencies**:
- `requests` - HTTP requests
- `openai` - OpenAI API client
- `anthropic` - Anthropic Claude API client (optional)
- `google-generativeai` - Google Gemini API client (optional)
- `colorama` - Cross-platform colors
- `python-dotenv` - Environment variables
- `rich` - Beautiful terminal formatting

## Examples

### Basic Search
```bash
iq "Latest developments in AI?"
```

### With Options
```bash
iq --use-cache --results 5 "Python best practices"
```

### Cache Search
```bash
iq --cache-search "machine learning"
```

### Interactive Mode
```bash
iq
❓ Your question: What is quantum computing?
# ... answer appears ...
❓ Your question: /cache stats
# ... cache statistics ...
❓ Your question: quit
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made by [@vowalsh](https://github.com/vowalsh)**

[⭐ Star on GitHub](https://github.com/vowalsh/iq-cli) • [🐛 Report Bug](https://github.com/vowalsh/iq-cli/issues) • [💡 Request Feature](https://github.com/vowalsh/iq-cli/issues)

</div>
