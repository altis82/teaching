# Day 1: LangChain with Local Ollama - Complete Tutorial

## Overview

Today you'll learn how to build LLM applications using **Ollama** - a framework to run large language models locally on your machine. This approach offers:
- **No API costs** - models run on your hardware
- **No internet required** - full privacy and offline capability
- **Fast iteration** - no rate limits or latency from external APIs
- **Full control** - use any open-source model

## What is Ollama?

Ollama is a command-line tool that simplifies running large language models locally. It handles:
- Model downloading and caching
- Memory management
- GPU acceleration (if available)
- A simple API interface for applications

**Key Ollama models:**
- `mistral` - Fast, 7B parameters, great for coding and general tasks
- `llama2` - Meta's model, good for reasoning
- `neural-chat` - Optimized for conversation
- `dolphin-mixtral` - More capable, multimodal
- `openchat` - Fast inference, quality output
- `qwen` - 

## 🛠 Installation Guide

### Step 1: Install Ollama

**macOS:**
```bash
# Download from https://ollama.ai
# Or use Homebrew:
brew install ollama
```

**Linux:**
```bash
# Ubuntu/Debian
curl -fsSL https://ollama.ai/install.sh | sh

# Or download from https://ollama.ai/download
```

<!-- **Windows:**
```bash
# Download installer from https://ollama.ai/download/windows
# Or use winget:
winget install ollama
``` -->

### Step 2: Start the Ollama Server

Open a terminal and run:
```bash
ollama serve
```

This starts the Ollama daemon on `http://localhost:11434`

You should see:
```
Starting Ollama...
Listening on 127.0.0.1:11434
```

**Keep this terminal open** while using Ollama.

### Step 3: Download a Model

Open a **new terminal** and download a model:

```bash
# Lightweight, fast model (recommended for development)
ollama pull mistral

# Or try these alternatives:
ollama pull llama2
ollama pull neural-chat
ollama pull dolphin-mixtral
```

Each download takes a few minutes depending on model size and internet speed.

Verify the model is installed:
```bash
ollama list
```

Example output:
```
NAME              ID              SIZE     MODIFIED
mistral:latest    2df6ca6d9335    4.1 GB   2 minutes ago
llama2:latest     78e26419b446    3.8 GB   1 hour ago
```

### Step 4: Test Ollama CLI

```bash
# Simple test - ask a question
ollama run mistral "What is machine learning?"

# Or start interactive mode
ollama run mistral
# Then type: "Explain embeddings in simple terms"
# Type 'exit' to quit
```

check available models in Ollama
```
curl http://${OLLAMA_BASE_URL}/api/tags

{"models":[{"name":"qwen3:8b","model":"qwen3:8b","modified_at":"2026-02-25T21:23:15.257280472Z","size":5225388164,"digest":"500a1f067a9f782620b40bee6f7b0c89e17ae61f686b92c24933e4ca4b2b8b41","details":{"parent_model":"","format":"gguf","family":"qwen3","families":["qwen3"],"parameter_size":"8.2B","quantization_level":"Q4_K_M"}},{"name":"qwen3-embedding:8b","model":"qwen3-embedding:8b","modified_at":"2026-02-25T21:23:15.257254472Z","size":4676805193,"digest":"64b933495768fbd3b87c20583d379728a07471e0c66733a9df87cd1901b3c44b","details":{"parent_model":"","format":"gguf","family":"qwen3","families":["qwen3"],"parameter_size":"7.6B","quantization_level":"Q4_K_M"}}]}
```



## 📦 Project Setup

### Create Virtual Environment

```bash
cd /home/system/teaching/langchain

# Create environment
python3 -m venv langchain_ollama_env

# Activate
source langchain_ollama_env/bin/activate  # Linux/macOS
# or
langchain_ollama_env\Scripts\activate     # Windows
```

### Install Dependencies

```bash
pip install --upgrade pip

# Core dependencies
pip install langchain langchain-community python-dotenv

# Additional utilities
pip install pydantic requests
```

Check installation:
```bash
python -c "import langchain; print(langchain.__version__)"
```

## 🚀 Lab 1: AI Blog Title Generator with Ollama

### Project Structure

```
langchain_ollama_env/
blog_generator/
├── __init__.py
├── main.py
├── .env
└── config.py
```

### Step 1: Create Configuration File

Create `blog_generator/config.py`:

```python
"""Configuration for Ollama models"""

# Ollama server settings
OLLAMA_BASE_URL = "http://localhost:11434"

# Model to use
# Options: "mistral", "llama2", "neural-chat", "dolphin-mixtral"
OLLAMA_MODEL = "mistral"

# Generation settings
TEMPERATURE = 0.7        # 0 = deterministic, 1.0 = creative
TOP_P = 0.9             # Controls diversity
TOP_K = 40              # Top-k sampling

# Application settings
NUM_TITLES = 5
TIMEOUT_SECONDS = 60
```

### Step 2: Create Main Application

Create `blog_generator/main.py`:

```python
"""
AI Blog Title Generator using Local Ollama
"""

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field
from typing import List
import json
from config import (
    OLLAMA_BASE_URL, 
    OLLAMA_MODEL, 
    TEMPERATURE,
    NUM_TITLES
)


# Define output structure
class BlogTitles(BaseModel):
    """Blog title generator output"""
    titles: List[str] = Field(
        description=f"List of {NUM_TITLES} creative and engaging blog titles"
    )
    summary: str = Field(
        description="Brief explanation of the title strategy"
    )


# Initialize Ollama LLM
def initialize_ollama():
    """Connect to local Ollama instance"""
    print(f"🤖 Connecting to Ollama at {OLLAMA_BASE_URL}")
    print(f"📚 Using model: {OLLAMA_MODEL}")
    
    llm = Ollama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=TEMPERATURE,
        top_p=0.9,
        top_k=40
    )
    
    return llm


# Create prompt template
def create_template():
    """Create the prompt template for blog title generation"""
    
    template = """You are an expert SEO content strategist and blog writer.
Your task is to generate {num_titles} creative, engaging, and SEO-optimized blog titles.

Blog Topic: {topic}

Requirements for the titles:
1. Make them attention-catching and compelling
2. Include relevant keywords naturally
3. Keep them between 5-10 words
4. Vary the style (how-to, listicle, question, thought-leadership)
5. Make them suitable for the target audience

Generate exactly {num_titles} titles and provide a brief strategy explanation.

Format your response as JSON with this exact structure:
{{
    "titles": ["Title 1", "Title 2", "Title 3", "Title 4", "Title 5"],
    "summary": "Brief explanation of the title strategy used"
}}

Remember: The titles should be clickable, memorable, and optimized for search engines."""

    return PromptTemplate(
        input_variables=["topic", "num_titles"],
        template=template
    )


# Main generation function
def generate_blog_titles(topic: str) -> dict:
    """
    Generate blog titles for a given topic
    
    Args:
        topic: The blog topic/subject
        
    Returns:
        Dictionary with titles and strategy explanation
    """
    
    print(f"\n✨ Generating blog titles for: '{topic}'")
    print("⏳ This may take 10-20 seconds on first run...\n")
    
    try:
        # Initialize LLM
        llm = initialize_ollama()
        
        # Create prompt
        prompt = create_template()
        
        # Create chain
        chain = prompt | llm
        
        # Generate
        result = chain.invoke({
            "topic": topic,
            "num_titles": NUM_TITLES
        })
        
        # Parse JSON response
        try:
            # Clean up the response
            json_str = result.strip()
            
            # Remove markdown code blocks if present
            if json_str.startswith("```"):
                json_str = json_str.split("```")[1]
                if json_str.startswith("json"):
                    json_str = json_str[4:]
            
            json_str = json_str.strip()
            
            # Parse JSON
            output = json.loads(json_str)
            
            return {
                "success": True,
                "topic": topic,
                "titles": output.get("titles", []),
                "strategy": output.get("summary", ""),
                "model": OLLAMA_MODEL
            }
            
        except json.JSONDecodeError as e:
            # Fallback: extract titles from raw text
            print("⚠️  Could not parse JSON, using text extraction...")
            lines = result.split('\n')
            titles = [line.strip() for line in lines if line.strip() and not line.startswith('{')][:NUM_TITLES]
            
            return {
                "success": True,
                "topic": topic,
                "titles": titles,
                "strategy": "Generated from text output",
                "model": OLLAMA_MODEL
            }
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "hint": "Make sure: 1) Ollama is running, 2) Model is installed, 3) You have internet for first pull"
        }


# Display results nicely
def display_results(result: dict):
    """Pretty print the results"""
    
    if not result.get("success"):
        print(f"❌ Failed to generate titles")
        print(f"   Error: {result.get('error')}")
        print(f"   Hint: {result.get('hint')}")
        return
    
    print("\n" + "="*60)
    print(f"📝 Blog Titles for: '{result['topic']}'")
    print(f"🤖 Generated by: {result['model']}")
    print("="*60 + "\n")
    
    for i, title in enumerate(result['titles'], 1):
        print(f"  {i}. {title}")
    
    print(f"\n💡 Strategy: {result['strategy']}")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    
    # Test topics
    topics = [
        "Artificial Intelligence in Banking",
        "Remote Work Best Practices",
        "Sustainable Technology Trends"
    ]
    
    print("🚀 Blog Title Generator with Local Ollama\n")
    
    for topic in topics:
        result = generate_blog_titles(topic)
        display_results(result)
        print()
```

### Step 3: Create Entry Point

Create `blog_generator/__init__.py`:

```python
"""Blog Title Generator Package"""
from .main import generate_blog_titles, display_results

__version__ = "1.0.0"
__all__ = ["generate_blog_titles", "display_results"]
```

## ▶️ Running the Application

### Terminal 1: Start Ollama Server

```bash
ollama serve
```

Keep this running in the background.

### Terminal 2: Run Your Application

```bash
cd /home/system/teaching/langchain

# Activate environment
source langchain_ollama_env/bin/activate

# Run the generator
python blog_generator/main.py
```

### Example Output

```
🚀 Blog Title Generator with Local Ollama

✨ Generating blog titles for: 'Artificial Intelligence in Banking'
⏳ This may take 10-20 seconds on first run...

============================================================
📝 Blog Titles for: 'Artificial Intelligence in Banking'
🤖 Generated by: mistral
============================================================

  1. AI Revolution in Banking: Security Meets Innovation
  2. How Banks Are Using Machine Learning for Fraud Prevention
  3. The Future of Financial Services: AI Integration Guide
  4. 5 Ways Artificial Intelligence Reduces Banking Costs
  5. Regulatory Challenges of AI in Modern Banking Systems

💡 Strategy: The titles combine trending keywords (AI, machine learning) with banking-specific concerns (security, fraud, regulations) to attract both professionals and interested readers.
============================================================
```

## 🛠 Advanced: Interactive Mode

Create `blog_generator/interactive.py`:

```python
"""Interactive blog title generator"""

from main import generate_blog_titles, display_results
import readline  # Enables command history


def main():
    """Interactive CLI interface"""
    
    print("\n" + "="*60)
    print("🎯 Interactive Blog Title Generator")
    print("="*60)
    print("Commands:")
    print("  - Enter a topic to generate titles")
    print("  - 'help' for more options")
    print("  - 'quit' to exit\n")
    
    while True:
        try:
            topic = input("📝 Enter blog topic (or 'quit'): ").strip()
            
            if topic.lower() == 'quit':
                print("👋 Goodbye!")
                break
            
            if topic.lower() == 'help':
                print_help()
                continue
            
            if not topic:
                print("⚠️  Please enter a topic")
                continue
            
            # Generate titles
            result = generate_blog_titles(topic)
            display_results(result)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def print_help():
    """Show help information"""
    print("""
Available commands:
  - quit: Exit the program
  - help: Show this message
  - Clear screen: Cmd+K (Mac) or Ctrl+L (Linux)

Tips:
  - More specific topics generate better titles
  - Example: "Remote Work Productivity Tips" instead of "Work"
  - You can generate multiple topics in one session
    """)


if __name__ == "__main__":
    main()
```

Run interactive mode:
```bash
python blog_generator/interactive.py
```

## ⚡ Performance Tips

### Speed Up Inference

1. **Use a faster model:**
   ```bash
   ollama pull mistral  # Faster
   ollama pull llama2   # Slower but better quality
   ```

2. **Adjust temperature for speed:**
   - Lower values (0.3-0.5) = faster
   - Higher values (0.7-1.0) = slower but more creative

3. **Monitor resource usage:**
   ```bash
   # Check Ollama memory usage
   ps aux | grep ollama
   ```

### GPU Acceleration

If your system has a GPU (NVIDIA, AMD, Apple Metal):

```bash
# For NVIDIA CUDA:
# Install CUDA toolkit first, then Ollama automatically uses it

# For Apple Silicon (M1/M2):
# Ollama automatically detects and uses Metal GPU

# Check if GPU is being used:
ollama list  # Shows memory usage
```

## 🧪 Troubleshooting

### Problem: "Connection refused" error

**Solution:** Make sure Ollama server is running:
```bash
ollama serve  # In another terminal
```

### Problem: Model not found

**Solution:** Download the model:
```bash
ollama pull mistral
ollama list  # Verify it's installed
```

### Problem: Very slow inference

**Solution:** 
- Use a smaller model: `ollama pull mistral` instead of larger ones
- Check system resources: free disk space, RAM, CPU usage
- Close other applications

### Problem: Out of memory

**Solution:**
- Use a smaller model (7B parameters instead of 13B)
- Reduce batch size
- Increase swap space on your system

## 📊 Comparing Models

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| mistral | 4.1GB | ⚡⚡⚡ Fast | Good | General tasks, quick iteration |
| llama2 | 3.8GB | ⚡⚡ Medium | Excellent | Reasoning, complex tasks |
| neural-chat | 4.1GB | ⚡⚡ Medium | Good | Conversation, chat |
| dolphin-mixtral | 26GB | ⚡ Slow | Best | Complex analysis |

**Recommendation for Day 1:** Start with `mistral` for speed, then experiment with others.

## 📝 Assignment

1. **Run the blog title generator** with at least 5 different topics
2. **Compare output** between different models:
   ```bash
   ollama pull llama2  # Install second model
   # Edit config.py to switch between models
   ```
3. **Experiment with temperature:**
   - Set `TEMPERATURE = 0.3` for consistent titles
   - Set `TEMPERATURE = 0.9` for creative titles
   - Document the differences

4. **Create your own prompt** for a different task (e.g., email subject line generator)

## ✅ Checklist

- [ ] Ollama installed and running
- [ ] Model downloaded (`ollama list` shows it)
- [ ] Python virtual environment created
- [ ] Dependencies installed
- [ ] Blog title generator works
- [ ] Tested with multiple topics
- [ ] Experimented with different models/temperatures

## Next Steps

Once you're comfortable with Day 1:
- **Day 2:** Add memory management for chatbots
- **Day 3:** Build RAG system with local documents
- **Day 4:** Create tools and agents
- **Day 5:** Multi-agent orchestration

## Useful Resources

- **Ollama Official:** https://ollama.ai
- **Model Library:** https://ollama.ai/library
- **LangChain Ollama:** https://python.langchain.com/docs/integrations/llms/ollama
- **Community Discussions:** https://github.com/jmorganca/ollama/discussions

---

**Happy learning!** 🚀
