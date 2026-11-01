# VS Code + Ollama Integration Guide

## Overview
This guide documents the process of connecting Visual Studio Code's Chat and Agent features to local Ollama models.

## Prerequisites

### System Requirements
- Visual Studio Code 1.127 or newer
- Ollama installed and running
- At least one local model available in Ollama

### Ollama Requirements
- Ollama 0.17.6 or newer (recommended for cloud model sign-in)
- Model context length set appropriately for agent tasks

## Installation

### Install Ollama
1. Visit [Ollama website](https://ollama.com/download)
2. Download and install Ollama for your operating system
3. Verify installation with `ollama --version`

### Install VS Code Ollama Extension
1. Open VS Code
2. Go to Extensions view (Ctrl+Shift+X)
3. Search for "Ollama"
4. Install the [Ollama extension](https://marketplace.visualstudio.com/items?itemName=Ollama.ollama)

## Getting Started

### 1. Install a Model

#### Pull a Model
```bash
ollama pull <model-name>
```

#### Available Models
- **Qwen 2.5 Coder**: Good for code tasks, but limited context (4k-32k)
  - `qwen2.5-coder:0.5b`
  - `qwen2.5-coder:1.5b`
  - `qwen2.5-coder:3b`
  - `qwen2.5-coder:7b`
  - `qwen2.5-coder:14b`
  - `qwen2.5-coder:32b`

- **Llama 3.1**: Excellent for agent tasks with 128K context
  - `llama3.1:8b-instruct-q8_0` (recommended)
  - `llama3.1:70b-instruct-q8_0`

#### Recommended Models for Different Use Cases
| Model | Context | Best For |
|-------|---------|----------|
| qwen2.5-coder:7b | 4k-8k | Quick code edits |
| qwen2.5-coder:32b | 4k-8k | Complex code tasks |
| llama3.1:8b | 128K | Agent tasks, long documents |
| llama3.1:70b | 128K | Production agents |

### 2. Configure Context Length

Ollama defaults to context lengths based on VRAM:
- < 24 GiB VRAM: 4k context
- 24-48 GiB VRAM: 32k context
- ≥ 48 GiB VRAM: 256k context

#### Increase Context Length

**Using systemd (Linux):**
```bash
sudo systemctl edit ollama
```
Add to the editor:
```ini
[Service]
Environment="OLLAMA_CONTEXT_LENGTH=16384"
```
Save, exit, then restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**Using environment variable:**
```bash
OLLAMA_CONTEXT_LENGTH=32000 ollama serve
```
⚠️ Don't start a second Ollama server on the same port!

### 3. Check Ollama Status
```bash
ollama list        # List installed models
ollama ps          # Show currently running models and their context
curl http://localhost:11434  # Verify Ollama is running
```

## Using in VS Code

### Switch to Ollama Model

1. **Open Chat** in VS Code (Ctrl+L or use Command Palette)
2. **Open the model picker** at the bottom of the chat input
3. **Select** your model from the **Ollama** section
4. **Start chatting!**

### Use Agent Mode

1. Open Chat
2. Select your Ollama model
3. Switch Chat to **Agent** mode
4. Your prompts will now leverage the model with tool calling capabilities

### Model Configuration

Next to the model name in the picker, you can:
- Select **>** to open configuration submenu
- Set **thinking effort** (for reasoning models)
- Choose **context size** (if available)

### Auto Model Selection (Preview)
VS Code can automatically route requests to optimal models:
- Select **Auto** from the model picker
- Options: Efficiency, Balance, Intelligence (if available)

## Troubleshooting

### Model Doesn't Appear in Picker

1. **Make sure Ollama is running:**
   ```bash
   ollama list
   ```

2. **Refresh models in VS Code:**
   - Open Command Palette (Ctrl+Shift+P)
   - Run **Ollama: Refresh Models**

3. **Check Ollama is accessible:**
   ```bash
   curl http://localhost:11434
   ```

4. **Diagnose:**
   - Run **Ollama: Diagnose Models** from Command Palette
   - Check the **Ollama** output channel

### 502 Bad Gateway / Context Length Exceeded

This error means Ollama is running but your request is larger than the context window.

**Solution:**
1. **Increase context length** using steps in Section 2
2. **Use a model with larger context** (e.g., llama3.1:8b)
3. **Start a new chat** - old conversations can accumulate tokens
4. **Check model context** with `ollama ps`

### Ollama Not Found / Service Issues

```bash
# Check if installed as snap
snap services ollama

# If using systemd
systemctl status ollama

# Start service if needed
sudo systemctl start ollama
```

### Model Loading Issues

If a model fails to load:
```bash
# Unload all models
ollama stop <model-name>

# Restart Ollama
sudo systemctl restart ollama

# Try loading model again
ollama run <model-name>
```

### Context Length Still Too Small

After increasing `OLLAMA_CONTEXT_LENGTH`:
1. Restart Ollama
2. **Unload** the model: `ollama stop <model-name>`
3. **Reload** the model: `ollama run <model-name>`
4. **Check** new context with `ollama ps`
5. **Reload VS Code Chat window**
6. **Retry** your prompt

## Best Practices

### Model Selection Strategy
- **Small models** (< 10GB): Good for quick edits, less accurate
- **Medium models** (10-30GB): Good balance for most tasks
- **Large models** (≥ 70GB): Best for complex tasks, agent workflows

### Context Management
- **Keep chats shorter** - new chats prevent token accumulation
- **Summarize conversations** before long tasks
- **Use 128K context models** for agent tasks
- **Monitor VRAM usage** with `ollama ps`

### Performance Tips
- **Unload unused models**: `ollama stop <model-name>`
- **Keep one model loaded** to reduce load time
- **Monitor memory** - larger context needs more VRAM
- **Use smaller context** for quick queries (4k-8k)

## Advanced Configuration

### VS Code Settings

Create or edit `.vscode/settings.json`:
```json
{
  "ai.languageModel": "ollama",
  "github.copilot.chat.anthropic.thinking.effort": 0,
  "github.copilot.chat.responsesApiReasoningEffort": 0
}
```

### Custom Ollama Server

If Ollama runs on different host/port:
```json
{
  "github.copilot.models": [
    {
      "provider": "ollama",
      "name": "ollama",
      "model": "llama3.1:8b",
      "contextLength": 8192,
      "host": "http://localhost:11434"
    }
  ]
}
```

## Summary

1. Install Ollama and VS Code extension
2. Pull your preferred model (`llama3.1:8b` for agent tasks)
3. Increase context length if needed (16k-128k)
4. Select model in VS Code Chat
5. Use Agent mode for tool calling
6. Monitor with `ollama ps` and `curl localhost:11434`

## Commands Reference

```bash
# Install model
ollama pull <model-name>

# List models
ollama list

# Show running models with context
ollama ps

# Stop model
ollama stop <model-name>

# Restart Ollama
sudo systemctl restart ollama

# Check status
curl http://localhost:11434

# Set context length
OLLAMA_CONTEXT_LENGTH=32000 ollama serve
```

## Resources

- [Ollama Docs](https://ollama.com/docs)
- [Ollama Models](https://ollama.com/library)
- [Ollama VS Code Integration](https://docs.ollama.com/integrations/vscode)
- [VS Code AI Language Models](https://code.visualstudio.com/docs/agent-customization/language-models)
- [Context Length Guide](https://docs.ollama.com/context-length)

---
*Generated: October 2026*
*Documentation covers VS Code Chat + Agent integration with local Ollama models*
