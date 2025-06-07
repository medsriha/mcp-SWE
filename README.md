# Code QA Evaluator

This repository contains a Code QA evaluation system that uses OpenAI's models to answer questions about code repositories.

## Prerequisites

- Python 3.11 or higher
- OpenAI API key
- `uv` package manager

## Installation with uv

1. First, install `uv` if you haven't already:
```bash
pip install uv
```

2. Create and activate a new virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

3. Install the package and its dependencies:
```bash
uv pip install .  # Install base dependencies
uv pip install ".[dev]"  # Install development dependencies (optional)
```

## Configuration

Before running the evaluator or the agent, you need to set up the following:

1. Set your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. Configure the settings in `code_qa/config/settings.py`:
   - `qa_pairs_dir`: Directory containing Q/A markdown files for evaluation
   - `repo_url`: Path to the repository to evaluate
   - Other settings can be customized as needed

Example settings:
```python
qa_pairs_dir: Path = "/path/to/your/qa/pairs"
repo_url: str = "/path/to/your/repository"
```

## Running the agent

The agent comes with an integrated MCP server hosted locally that starts automatically when you run the CLI - no additional setup required! You can run the agent:

```bash
uv run code_qa/cli.py
```

The agent provides an interactive CLI where you can:
- Ask questions about code repositories
- Analyze GitHub or local repositories
- Get explanations about code functionality

Example interaction:
```
Code Q&A Agent initialized!
You can ask questions about code in repositories.
Example: 'Analyze https://github.com/user/repo and explain the main function'
Type 'quit' to exit.
```

## Running the QA Evaluator

To run the evaluation script:

```bash
uv run code_qa/eval/qa_evaluator.py
```

The script will:
1. Load Q/A pairs from the specified directory
2. Run evaluations using the CodeQAAgent
3. Generate ROUGE scores for each response
4. Create two output files:
   - `evaluation_report.txt`: Detailed evaluation report
   - `evaluation_results.json`: Raw results in JSON format

## Expected File Structure

The Q/A pairs should be organized in the following format:
```
qa_pairs_dir/
├── question1.q.md
├── question1.a.md
├── question2.q.md
├── question2.a.md
...
```

- `.q.md` files contain questions
- `.a.md` files contain corresponding ground truth answers

## Output

The evaluation will generate:
- Detailed report with ROUGE scores for each Q/A pair
- Aggregate metrics across all evaluations
- Individual response comparisons
