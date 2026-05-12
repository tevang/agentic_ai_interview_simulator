# Agentic AI Interview Simulator

The Agentic AI Interview Simulator is a tool designed to help developers practice for technical interviews. It simulates a realistic interview environment by creating AI agents that act as interviewers. These agents use information from candidate CVs, job descriptions, and real-time web research to ask relevant questions and provide detailed feedback on your answers.

## Features

- **Multi-Interviewer Simulation:** Practice with up to three different interviewers, each with a unique role and style.
- **Context-Aware Questions:** Questions are generated based on your CV and the specific job description provided.
- **Web Research Integration:** The system uses Tavily to research the company and relevant technologies to make the interview more realistic.
- **Detailed Evaluation:** Get immediate feedback, scores, and missing points for each of your answers.
- **Session Transcripts:** Each session is saved as a JSON transcript for later review.

## Prerequisites

You will need the following API keys:
- **OpenAI API Key:** Required for the LLM agents.
- **Tavily API Key:** Required for web research (optional but recommended).

## Installation

1. **Clone the repository** (if you haven't already).
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate the virtual environment:**
   - On Linux/macOS:
     ```bash
     source .venv/bin/activate
     ```
   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. **Set up environment variables:**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=sk-your-openai-key
   TAVILY_API_KEY=tvly-your-tavily-key
   ```
   Alternatively, you can export them in your shell:
   ```bash
   export OPENAI_API_KEY=sk-your-openai-key
   export TAVILY_API_KEY=tvly-your-tavily-key
   ```

2. **Configure the interview:**
   Modify `config.yaml` to specify the candidate info, job description, and interviewers.
   
   **Note on Interviewers:**
   The application supports one, two, or three interviewers. To use fewer interviewers, remove entries from `documents.interviewers` in your configuration file.

## Usage

### Run the Interview Simulation
```bash
python main.py --config config.yaml
```

### Ingest Data Only
To ingest PDFs and perform web research without starting the interview:
```bash
python main.py --config config.yaml --bootstrap-only
```

### Force Re-ingestion
To force a full refresh of ingested data (useful if you updated your CV or the job description):
```bash
python main.py --config config.yaml --force-reingest
```
