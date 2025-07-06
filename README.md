# Researcher - Multi-Agent Research System

A sophisticated multi-agent research orchestration system powered by Google's Gemini models, designed for comprehensive web research and analysis.

## 🚀 Features

- **Multi-Agent Architecture**: Main orchestrator (Gemini 2.5 Pro) + specialized research subagents (Gemini 2.5 Flash)
- **Live Status Dashboard**: Real-time tracking of agent progress with Rich terminal UI
- **Multiple Search Providers**: 8 free search providers with intelligent fallback and cycling (Crawl4ai, Brave, Serper, ScrapingAnt, DuckDuckGo, etc.)
- **Function Calling System**: AI agents can execute research, analysis, and report generation functions
- **Intelligent Workflows**: Iterative research with temp reports, additional research, and final synthesis
- **Rate Limiting & API Management**: Automatic compliance with Gemini API limits and circuit breakers
- **URL Deduplication**: Global tracking prevents duplicate processing across sessions
- **Comprehensive Reporting**: Final report synthesis with source attribution

## 🛠️ Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
crawl4ai-setup  # Set up Crawl4ai browser components
```

2. **Set Environment Variables**
```bash
export GEMINI_API_KEY="your_gemini_api_key"

# Optional: Additional search provider API keys for enhanced coverage
export SERPER_API_KEY="your_serper_key"
export BRAVE_API_KEY="your_brave_key" 
export SCRAPINGANT_API_KEY="your_scrapingant_key"
# ... other search provider keys
```

3. **Run**
```bash
python researcher.py
```

## 🎯 Usage Examples

```
🔬 Researcher> Analyze the current state of quantum computing research and its commercial applications

🔬 Researcher> Compare renewable energy adoption policies in Germany, France, and the UK over the past 2 years

🔬 Researcher> Research the economic impact of AI on healthcare systems across different countries
```

## 📊 System Architecture

- **Main Agent**: Orchestrates research, manages subagents, synthesizes final reports
- **Subagents**: Perform focused research with iterative analysis cycles
- **Search Layer**: Multi-provider search with intelligent fallback and rate limiting
- **Content Layer**: Async web scraping and content extraction
- **Reporting Layer**: Structured output with comprehensive final reports

## 📁 Output Structure

```
research_results/
├── final_comprehensive_report.md     # Main synthesized report
├── query_1_*/                        # Individual query results
├── additional_research/               # Subagent additional research
├── agent_logs/                       # Detailed agent workflows
└── prompt_logs/                      # API interaction logs
```

## 🔄 Workflow

1. **Query Analysis**: Main agent determines research requirements
2. **Search Orchestration**: Deploy subagents with URL budgets and token limits
3. **Iterative Research**: Subagents analyze → create temp reports → gather additional sources → finalize
4. **Synthesis**: Main agent combines all research into comprehensive final report

## ⚡ Advanced Features

- **Token Management**: Dynamic token allocation per agent (250K / num_agents)
- **Memory Management**: Adaptive browser session management based on system resources
- **Circuit Breakers**: Prevent API abuse with automatic backoff
- **Comprehensive Logging**: Full interaction history and workflow tracking
- **Live UI**: Real-time agent status with spinner animations and progress tracking

---

**Note**: This is a research prototype. An improved and significantly enhanced version of this system will be integrated as a subsystem into **ATLAS2**, featuring expanded capabilities, better performance, and additional research modalities.

## 🔗 Requirements

- Python 3.8+
- Google Gemini API access
- 4GB+ RAM recommended for concurrent browser sessions
- Optional: Search provider API keys for enhanced coverage
