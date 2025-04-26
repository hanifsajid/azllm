# azLLM: A Unified LLM Interface for Multi-Provider Access

`azLLM` is a Python package that provides a unified interface to work with multiple LLM providers including OpenAI, DeepSeek, Grok, Gemini, Meta's LLaMA, Anthropic, Ollama, and more.

> NOTE: For advanced usage, see the full azLLM docuemntation and/or examples.

---
## Features

- One unified interface for all major LLM APIs
- Batch and parallel prompt generation
- Structured outputs (parsing) with <a href="https://docs.pydantic.dev/latest/" target="_blank"> Pydantic</a>
- Per-model configurations and lazy initialization
- Clean error handling
- `.env`-based API key management
---

## Supported Clients

- <a href="https://platform.openai.com/docs/overview" target="_blank">OpenAI</a>
- <a href="https://api-docs.deepseek.com" target="_blank">DeepSeek</a>
- <a href="https://x.ai" target="_blank">Grok</a>
- <a href="https://www.anthropic.com/claude" target="_blank">Anthropic</a>
- <a href="https://fireworks.ai" target="_blank">Fireworks</a> for Meta's LLaMA and others.
- <a href="https://ai.google.dev/gemini-api/docs" target="_blank">Google's Gemini</a>
- <a href="https://ollama.com" target="_blank">Ollama</a>

**NOTE:**   If you would like to request support for additional LLMs, please open an issue on our <a href= "https://github.com/hanifsajid/azLLM/issues" target="_blank">GitHub page</a>.

## Installation

You can install the `azLLM` package via pip:

```bash
pip install azLLM
```

### Prerequisites

- Python 3.11+
- Create a `.env` file to store your API keys. For example:

    ```bash
    OPENAI_API_KEY=your_openai_api_key
    DEEPSEEK_API_KEY=your_deepseek_api_key
    XAI_API_KEY=your_xai_api_key
    GEMINI_API_KEY=your_gemini_api_key
    ANTHROPIC_API_KEY=your_anthropic_api_key
    FIREWORKS_API_KEY=your_fireworks_api_key
    ```
- <a href="https://ollama.com" target="_blank">Ollama</a> must be installed and running locally to use Ollama models.

## Quick Start

### Basic Initialization

```Python
from azllm import azLLM
manager = azLLM()  # Instantiated with default parameters 
```

### Generate Text from a Single Prompt 

```Python
prompt = 'What is the captial of France?'
generated_text = manager.generate_text('openai', prompt)
print(generated_text)
```
### Batch Generation

Generate responses for multiple prompts at once:

```Python
batch_prompts = [
    'What is the capital of France?',
    'Tell me a joke.'
    ]

results = manager.batch_generate('openai', batch_prompts)
for result in results:
    print(result)
```
### Parallel Generation 

Run a single prompt across multiple models simultaneously:

```python
prompt = 'What is the capital of France?'
models = [
    'openai',
    'grok',
    'ollama']

results = manager.generate_parallel(prompt, models)
for model, result in results.items():
    print(f"Model: {model},\nResult: {result}\n")
```

## License

```md
MIT License
```

## Citation

```
@misc{azLLM,
  title        = {azLLM},
  author       = {Hanif Sajid and Benjamin Radford and Yaoyao Dai and Jason Windett},
  year         = {2025},
  month        = apr,
  version      = {0.1.0},
  howpublished = {https://github.com/hanifsajid/azLLM},
  note         = {MIT License},
  abstract     = {azLLM is a Python package designed to interface with various large language models (LLMs) from different AI providers. It offers a unified interface for interacting with models from providers like OpenAI, DeepSeek, Grok, Gemini, Meta's Llama, Anthropic, Ollama, and others. The package allows for customizable configurations, batch generation, parallel generation, error handling, and the ability to parse structured responses from different models.}
}
```
