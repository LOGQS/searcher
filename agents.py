import os
import time
import asyncio
import warnings
from collections import deque
from datetime import datetime
from typing import List, Dict, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import threading
from contextlib import contextmanager
from google import genai
from google.genai import types

# Global warning suppression for Gemini API warnings
warnings.filterwarnings("ignore", message=".*non-text parts.*")
# Rich imports for live UI
from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from prompts import sub_agent_prompt
# Import research functions
from query2url import query2url
from url2file import url2file
import builtins

# Initialize console for UI updates
console = Console()

# Context manager to suppress specific Gemini API warnings
@contextmanager
def suppress_gemini_warnings():
    """
    Context manager to suppress specific Gemini API warnings about non-text parts.
    These warnings occur when accessing response.text on responses containing function calls,
    but they're not relevant to our use case since we handle both text and function calls.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*non-text parts.*")
        yield

# Global console lock for thread-safe output
_console_lock = threading.Lock()

# Thread-safe console output functions
def safe_console_print(*args, **kwargs):
    """Thread-safe wrapper for console.print()"""
    with _console_lock:
        console.print(*args, **kwargs)

def safe_builtin_print(*args, **kwargs):
    """Thread-safe wrapper for builtin print()"""
    with _console_lock:
        _original_print(*args, **kwargs)

# --- Suppress noisy external prints -------------------------------------------------
_original_print = builtins.print

def _silent_external_print(*args, **kwargs):
    """Intercept global print calls.
    Suppress lines coming from external libraries (debug/info markers) that would
    otherwise break the Rich live table UI. We allow prints that our application
    intentionally emits (animations/emojis etc.) and drop typical log prefixes.
    """
    text = " ".join(str(a) for a in args)
    noisy_prefixes = (
        "[INFO]", "[DEBUG]", "[FETCH]", "[SCRAPE]", "[COMPLETE]", "[INIT]",
        "[WARN]", "[ERROR]"
    )
    # If the text starts with any noisy prefix, swallow it.
    if text.lstrip().startswith(noisy_prefixes):
        return  # Suppress
    # Otherwise, pass through to the original print
    _original_print(*args, **kwargs)

# Activate interception as early as possible
builtins.print = _silent_external_print

# Coordinated output system
class CoordinatedConsole:
    """Console that coordinates animations with Rich output."""
    
    def __init__(self, rich_console):
        self.console = rich_console
        # When a Rich Live context is active, this will hold the Live instance.
        # It allows CoordinatedConsole to route output through the Live console
        # to avoid breaking the Live-rendered table.
        self.live_console: Optional[Live] = None
        
    def print(self, *args, **kwargs):
        """Print with animation coordination. Automatically routes output through
        an active Rich Live console (if any) to keep live tables stable."""
        # Stop all animations globally during output
        animation_manager.stop_all_animations()
        time.sleep(0.1)  # Allow animations to fully stop
        
        # When a Live context is active we suppress arbitrary prints to avoid
        # breaking the table layout. Instead, messages should be surfaced via
        # the status tracker / panel. Only allow printing when no live table
        # is active.
        if self.live_console is None:
            # Clear any residual animation line
            safe_builtin_print("\r" + " " * 120 + "\r", end="", flush=True)
            self.console.print(*args, **kwargs)
        # When live console is active, we completely suppress prints to keep table clean
        # All important information should flow through status_tracker updates
        
        # Reset animation manager for new animations after output
        time.sleep(0.05)
        animation_manager.reset()

# Create coordinated console
coordinated_console = CoordinatedConsole(console)

# Global animation control to prevent overlapping animations
class AnimationManager:
    def __init__(self):
        self.active_animation = None
        self.lock = threading.Lock()
        self.suppress_lower_level = False
        self.paused = False
        self.should_stop_all = False
        self.active_threads = {}  # Track animation threads for proper joining
    
    def start_animation(self, name: str, priority: int = 0) -> bool:
        """Start an animation if none is active or if this has higher priority. Returns True if started."""
        with self.lock:
            # Don't start if globally stopped
            if self.should_stop_all:
                return False
                
            # High priority animations (like report generation) can suppress lower level ones
            if priority >= 2:  # High priority
                self.suppress_lower_level = True
            
            if self.active_animation is None:
                self.active_animation = name
                self._current_priority = priority
                return True
            elif priority >= 2 and priority > getattr(self, '_current_priority', 0):
                # High priority animation can take over
                self.active_animation = name
                self._current_priority = priority
                return True
            return False
    
    def register_thread(self, name: str, thread: threading.Thread):
        """Register an animation thread for proper cleanup."""
        with self.lock:
            self.active_threads[name] = thread
    
    def stop_animation(self, name: str):
        """Stop an animation if it's the active one and join its thread."""
        thread_to_join = None
        with self.lock:
            if self.active_animation == name:
                self.active_animation = None
                self.suppress_lower_level = False
                self._current_priority = 0
                self.paused = False
                # Get thread reference for joining outside the lock
                thread_to_join = self.active_threads.pop(name, None)
        
        # Join the thread outside the lock to avoid deadlock
        if thread_to_join and thread_to_join.is_alive():
            thread_to_join.join(timeout=0.5)  # Wait up to 500ms for clean shutdown
    
    def pause_all_animations(self):
        """Pause all animations temporarily for other output."""
        with self.lock:
            self.paused = True
    
    def resume_all_animations(self):
        """Resume paused animations."""
        with self.lock:
            self.paused = False
    
    def stop_all_animations(self):
        """Stop all animations globally and join all threads."""
        threads_to_join = []
        with self.lock:
            self.should_stop_all = True
            self.active_animation = None
            self.suppress_lower_level = False
            self._current_priority = 0
            # Get all thread references for joining
            threads_to_join = list(self.active_threads.values())
            self.active_threads.clear()
        
        # Join all threads outside the lock
        for thread in threads_to_join:
            if thread.is_alive():
                thread.join(timeout=0.5)
    
    def reset(self):
        """Reset animation manager to allow new animations."""
        threads_to_join = []
        with self.lock:
            self.should_stop_all = False
            self.paused = False
            self.active_animation = None
            self.suppress_lower_level = False
            self._current_priority = 0
            # Clean up any remaining threads
            threads_to_join = list(self.active_threads.values())
            self.active_threads.clear()
        
        # Join any remaining threads outside the lock
        for thread in threads_to_join:
            if thread.is_alive():
                thread.join(timeout=0.2)
    
    def is_animation_active(self, name: str = None) -> bool:
        """Check if any animation (or a specific one) is active."""
        with self.lock:
            if self.should_stop_all or self.paused:
                return False
            if name:
                return self.active_animation == name
            return self.active_animation is not None
    
    def should_suppress_lower_level(self) -> bool:
        """Check if lower level animations should be suppressed."""
        with self.lock:
            return self.suppress_lower_level or self.should_stop_all

# Global animation manager instance
animation_manager = AnimationManager()

# Global URL mapping store
global_url_mapping_store = {}

@dataclass
class RateLimitConfig:
    """Rate limiting configuration for Gemini models."""
    rpm: int  # Requests per minute
    tpm: int  # Tokens per minute  
    rpd: int  # Requests per day
    
# Model-specific rate limits
RATE_LIMITS = {
    "gemini-2.5-pro": RateLimitConfig(rpm=4, tpm=200000, rpd=80),  # More conservative limits
    "gemini-2.5-flash": RateLimitConfig(rpm=8, tpm=200000, rpd=200),  # More conservative limits
}

class RateLimiter:
    """Advanced rate limiter with sliding window tracking."""
    
    def __init__(self, model: str):
        self.model = model
        self.config = RATE_LIMITS.get(model, RateLimitConfig(rpm=10, tpm=250000, rpd=200))
        
        # Sliding window tracking using deques
        self.request_times = deque()
        self.token_usage = deque()  # (timestamp, token_count) pairs
        self.daily_requests = deque()
        
        # Current period tracking
        self.minute_start = time.time()
        self.day_start = time.time()
        
    async def wait_if_needed(self, estimated_tokens: int = 1000) -> Dict[str, Any]:
        """Check rate limits and wait if necessary. Returns status info."""
        current_time = time.time()
        
        # Clean old entries
        self._clean_old_entries(current_time)
        
        # Check each limit
        delays = []
        status = {"waiting": False, "reasons": []}
        
        # RPM check
        if len(self.request_times) >= self.config.rpm:
            delay = 60 - (current_time - self.request_times[0])
            if delay > 0:
                delays.append(delay)
                status["reasons"].append(f"RPM limit ({self.config.rpm})")
        
        # TPM check
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        if current_tokens + estimated_tokens > self.config.tpm:
            # Find when enough tokens will be freed
            needed_tokens = (current_tokens + estimated_tokens) - self.config.tpm
            freed_tokens = 0
            delay = 0
            for timestamp, tokens in self.token_usage:
                freed_tokens += tokens
                delay = max(delay, 60 - (current_time - timestamp))
                if freed_tokens >= needed_tokens:
                    break
            
            if delay > 0:
                delays.append(delay)
                status["reasons"].append(f"TPM limit ({self.config.tpm})")
        
        # RPD check
        if len(self.daily_requests) >= self.config.rpd:
            delay = 86400 - (current_time - self.daily_requests[0])
            if delay > 0:
                delays.append(delay)
                status["reasons"].append(f"RPD limit ({self.config.rpd})")
        
        # Wait if needed
        if delays:
            max_delay = max(delays)
            status["waiting"] = True
            status["delay_seconds"] = max_delay
            
            # Show user feedback during wait
            coordinated_console.print(f"[yellow]⏱️ Rate limit reached for {self.model}. Waiting {max_delay:.1f}s...[/yellow]")
            coordinated_console.print(f"[dim]Limits hit: {', '.join(status['reasons'])}[/dim]")
            
            await asyncio.sleep(max_delay)
        
        # Record this request
        self.request_times.append(current_time)
        self.token_usage.append((current_time, estimated_tokens))
        self.daily_requests.append(current_time)
        
        return status
    
    def _clean_old_entries(self, current_time: float):
        """Remove entries outside the tracking window."""
        # Remove requests older than 1 minute
        while self.request_times and current_time - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Remove token usage older than 1 minute
        while self.token_usage and current_time - self.token_usage[0][0] > 60:
            self.token_usage.popleft()
        
        # Remove daily requests older than 24 hours
        while self.daily_requests and current_time - self.daily_requests[0] > 86400:
            self.daily_requests.popleft()
    
    def get_status(self) -> Dict[str, Any]:
        """Get current rate limit status."""
        current_time = time.time()
        self._clean_old_entries(current_time)
        
        current_tokens = sum(tokens for _, tokens in self.token_usage)
        
        return {
            "model": self.model,
            "requests_per_minute": len(self.request_times),
            "tokens_per_minute": current_tokens,
            "requests_per_day": len(self.daily_requests),
            "limits": {
                "rpm": self.config.rpm,
                "tpm": self.config.tpm,
                "rpd": self.config.rpd
            },
            "utilization": {
                "rpm_percent": (len(self.request_times) / self.config.rpm) * 100,
                "tpm_percent": (current_tokens / self.config.tpm) * 100,
                "rpd_percent": (len(self.daily_requests) / self.config.rpd) * 100
            }
        }

# Global rate limiters
rate_limiters = {
    model: RateLimiter(model) for model in RATE_LIMITS.keys()
}

# Global rate limiting lock to prevent parallel API calls
_global_api_lock = asyncio.Lock()

# Circuit breaker to prevent rapid retries when API is consistently failing
class CircuitBreaker:
    """Circuit breaker to prevent rapid retries when API is consistently failing."""
    
    def __init__(self, model: str, failure_threshold: int = 5, timeout: int = 60):
        self.model = model
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half-open
        
    def record_success(self):
        """Record a successful API call."""
        self.failure_count = 0
        self.state = "closed"
        
    def record_failure(self):
        """Record a failed API call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            coordinated_console.print(f"[bold yellow]⚠️ Circuit breaker opened for {self.model} after {self.failure_count} failures[/bold yellow]")
        
    def can_proceed(self) -> bool:
        """Check if we can proceed with API call."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half-open"
                coordinated_console.print(f"[dim yellow]Circuit breaker for {self.model} moving to half-open[/dim yellow]")
                return True
            return False
        elif self.state == "half-open":
            return True
        return False
    
    def should_wait(self) -> int:
        """Return seconds to wait before next attempt."""
        if self.state == "open":
            return max(0, self.timeout - int(time.time() - self.last_failure_time))
        return 0

# Global circuit breakers for each model
circuit_breakers = {
    model: CircuitBreaker(model) for model in RATE_LIMITS.keys()
}

@dataclass 
class AgentState:
    """Track state of individual research agents with iteration and temp report management."""
    agent_id: str
    query: str
    state: str = "🔄 Initializing"
    iteration: int = 0
    file_count: int = 0
    urls_remaining: int = 0
    total_urls_used: int = 0
    temp_reports: List[Dict[str, str]] = field(default_factory=list)  # Track temp reports with paths and iterations
    completed: bool = False
    final_report_path: Optional[str] = None
    error_details: Optional[str] = None
    query_directory: Optional[str] = None  # Track query directory for file management
    token_limit: int = 0  # Token limit for final synthesis (250K / num_agents)
    decision_history: List[str] = field(default_factory=list)  # Track SUFFICIENT/INSUFFICIENT decisions
    
    def add_temp_report(self, report_path: str, iteration: int, content_preview: str = ""):
        """Add a temp report to the agent's history"""
        self.temp_reports.append({
            "path": report_path,
            "iteration": iteration,
            "content_preview": content_preview[:100] + "..." if len(content_preview) > 100 else content_preview,
            "timestamp": datetime.now().isoformat()
        })
    
    def get_latest_temp_report(self) -> Optional[Dict[str, str]]:
        """Get the most recent temp report"""
        return self.temp_reports[-1] if self.temp_reports else None
    
    def calculate_token_limit(self, total_agents: int) -> int:
        """Calculate token limit based on total agents as per flow.md (250K / num_agents)"""
        self.token_limit = max(10000, 230000 // total_agents)  # Min 10K tokens as per flow.md, max 230K / num_agents
        return self.token_limit

class LiveStatusTracker:
    """Live status tracking for research agents."""
    
    def __init__(self):
        self.agents: Dict[str, AgentState] = {}
        self.global_urls_used: Set[str] = set()
        self.start_time = datetime.now()
        # Track global API status (rate-limit / retry info etc.)
        self.api_status: str = ""
        # Track research progress information
        self.files_loaded: int = 0
        self.queries_processed: int = 0
        self.current_operation: str = ""
        self.research_updates: List[str] = []
    
    def add_agent(self, agent_id: str, query: str, max_urls: int, query_directory: str = None, total_agents: int = 1):
        """Add new agent to tracking with enhanced state management."""
        agent_state = AgentState(
            agent_id=agent_id,
            query=query[:50] + "..." if len(query) > 50 else query,
            urls_remaining=max_urls,
            query_directory=query_directory
        )
        
        # Calculate token limit based on total agents (250K / num_agents as per flow.md)
        agent_state.calculate_token_limit(total_agents)
        
        self.agents[agent_id] = agent_state
        
        coordinated_console.print(f"[dim cyan]🤖 Agent {agent_id} initialized: Token limit = {agent_state.token_limit:,}[/dim cyan]")
    
    def update_agent(self, agent_id: str, **kwargs):
        """Update agent state."""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            for key, value in kwargs.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
    
    def update_research_progress(self, **kwargs):
        """Update research progress tracking."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def add_research_update(self, message: str):
        """Add a research update message."""
        self.research_updates.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        # Keep only last 5 updates
        if len(self.research_updates) > 5:
            self.research_updates = self.research_updates[-5:]
    
    def create_status_table(self) -> Table:
        """Create Rich table showing agent status."""
        table = Table(title="🔬 Live Research Agent Status", show_header=True, header_style="bold magenta")
        table.add_column("Agent", style="cyan", width=8)
        table.add_column("Query", style="white", width=25)
        table.add_column("State", style="yellow", width=15)
        table.add_column("Iter", style="green", width=5)
        table.add_column("Files", style="blue", width=6)
        table.add_column("URLs Left", style="red", width=9)
        
        for agent in self.agents.values():
            status_color = "green" if agent.completed else "yellow"
            table.add_row(
                agent.agent_id,
                agent.query,
                f"[{status_color}]{agent.state}[/{status_color}]",
                str(agent.iteration),
                str(agent.file_count),
                str(agent.urls_remaining)
            )
        
        # Append research progress info
        if self.current_operation:
            table.add_row(
                "SYSTEM",
                self.current_operation[:25],
                "[cyan]Working[/cyan]",
                "",
                str(self.files_loaded),
                ""
            )
        
        # Append API status if present
        if self.api_status:
            table.add_row(
                "API",
                "",
                f"[yellow]{self.api_status}[/yellow]",
                "",
                "",
                ""
            )
        
        return table
    
    def create_summary_panel(self) -> Panel:
        """Create summary panel with overall statistics."""
        total_agents = len(self.agents)
        completed_agents = sum(1 for a in self.agents.values() if a.completed)
        total_urls_used = sum(a.total_urls_used for a in self.agents.values())
        
        runtime = datetime.now() - self.start_time
        runtime_str = str(runtime).split('.')[0]
        
        summary_text = f"""[bold cyan]Research Session Overview[/bold cyan]
        
🤖 Total Agents: {total_agents}
✅ Completed: {completed_agents}
🔄 Active: {total_agents - completed_agents}
🌐 URLs Processed: {total_urls_used}
⏱️ Runtime: {runtime_str}
"""
        
        return Panel(summary_text, border_style="blue", padding=(1, 2))


# Global status tracker
status_tracker = LiveStatusTracker()


@dataclass
class AgentConfig:
    """Simple configuration for Gemini agents."""
    api_key: str
    model: str = "gemini-2.5-flash"
    system_instruction: Optional[str] = None


class ResearchLimits:
    """Global research limits tracking with URL deduplication."""
    def __init__(self):
        self.total_urls_processed = 0
        self.max_total_urls = 2500
        self.processed_urls: Set[str] = set()  # Track all processed URLs globally
        
    def can_process_urls(self, count: int) -> bool:
        """Check if we can process additional URLs without exceeding global limit."""
        return (self.total_urls_processed + count) <= self.max_total_urls
    
    def add_processed_urls(self, urls: List[str]):
        """Add URLs to processed set and update count."""
        new_urls = [url for url in urls if url not in self.processed_urls]
        self.processed_urls.update(new_urls)
        self.total_urls_processed += len(new_urls)
        return new_urls  # Return only new URLs
    
    def filter_new_urls(self, urls: List[str]) -> List[str]:
        """Filter out already processed URLs."""
        return [url for url in urls if url not in self.processed_urls]
    
    def get_remaining_urls(self) -> int:
        """Get remaining URL allowance."""
        return self.max_total_urls - self.total_urls_processed


# Global research limits instance
research_limits = ResearchLimits()

async def research_function(
    queries_config: List[Dict[str, Any]], 
    output_dir: str = "research_results",
    global_task_context: str = ""
) -> Dict[str, Any]:
    """
    Research function that gets called by AI models via function calling.
    This function is properly executed by us when the AI model requests it.
    """
    # Log function call
    function_logger.log_function_call(
        caller="MainAgent(gemini-2.5-pro)",
        function_name="research",
        queries_config=queries_config,
        output_dir=output_dir,
        global_task_context=global_task_context
    )
    # Validate inputs
    if len(queries_config) > 50:
        raise ValueError("Cannot process more than 50 queries in a single call")
    
    # Validate each query config
    for i, config in enumerate(queries_config):
        if not isinstance(config, dict):
            raise ValueError(f"Query {i} must be a dictionary")
        
        required_keys = ['query', 'init_source_count', 'use_subagents', 'max_source_count']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Query {i} missing required key: {key}")
        
        if config['init_source_count'] > 10:
            raise ValueError(f"Query {i}: init_source_count cannot exceed 10")
        
        if config['max_source_count'] > 50:
            raise ValueError(f"Query {i}: max_source_count cannot exceed 50")
        
        if config['init_source_count'] > config['max_source_count']:
            raise ValueError(f"Query {i}: init_source_count cannot exceed max_source_count")
    
    # Check global URL limit
    total_requested_urls = sum(config['max_source_count'] for config in queries_config)
    if not research_limits.can_process_urls(total_requested_urls):
        remaining = research_limits.get_remaining_urls()
        raise ValueError(f"Requested {total_requested_urls} URLs exceeds remaining global limit of {remaining}")
    
    # Initialize status tracking
    status_tracker.__init__()  # Reset tracker for new research session
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Initialize results tracking
    results = {
        "queries_processed": 0,
        "total_urls_fetched": 0,
        "successful_queries": [],
        "failed_queries": [],
        "subagent_reports": [],
        "research_files": [],
        "search_results_table": [],
        "final_reports": [],
        "global_url_mapping": {},  # Combined URL mapping from all queries
        "statistics": {
            "global_urls_remaining": research_limits.get_remaining_urls(),
            "queries_with_subagents": 0,
            "average_urls_per_query": 0,
            "start_time": datetime.now().isoformat()
        }
    }
    
    # Create search results table for display
    search_table = Table(title="🔍 Search Results by Query", show_header=True, header_style="bold cyan")
    search_table.add_column("Query", style="white", width=30)
    search_table.add_column("URLs Found", style="green", width=12)
    search_table.add_column("Status", style="yellow", width=15)
    
    coordinated_console.print(Panel.fit("🚀 Starting Research Operation", border_style="green", padding=(1, 2)))
    
    # Phase 1: Initial URL Search and Display
    coordinated_console.print("\n[bold cyan]Phase 1: Initial URL Search[/bold cyan]")
    
    for i, query_config in enumerate(queries_config):
        query = query_config['query']
        init_count = query_config['init_source_count']
        use_subagents = query_config['use_subagents']
        max_count = query_config['max_source_count']
        
        try:
            coordinated_console.print(f"\n[bold white]Processing query {i+1}/{len(queries_config)}:[/bold white] {query}")
            
            # Step 1: Get initial URLs with rate limiting
            limiter = rate_limiters.get("gemini-2.5-flash")  # Using 2.5 Flash for search
            if limiter:
                await limiter.wait_if_needed(500)  # Estimate tokens for search
            
            coordinated_console.print(f"[cyan]  Fetching {init_count} initial URLs...[/cyan]")
            urls = query2url(query, num_results=init_count)
            
            # Filter for new URLs only
            if urls:
                new_urls = research_limits.filter_new_urls(urls)
                filtered_count = len(urls) - len(new_urls)
                if filtered_count > 0:
                    coordinated_console.print(f"[yellow]  Filtered out {filtered_count} already processed URLs[/yellow]")
                urls = new_urls
            
            if not urls:
                search_table.add_row(query[:28] + "..." if len(query) > 28 else query, "0", "[red]No URLs[/red]")
                results["failed_queries"].append({
                    "query": query,
                    "error": "No new URLs found for query",
                    "index": i
                })
                continue
            
            # Add to search results table
            search_table.add_row(
                query[:28] + "..." if len(query) > 28 else query, 
                str(len(urls)), 
                "[green]Success[/green]"
            )
            
            # Step 2: Convert URLs to files and track URL mapping
            query_output_dir = output_path / f"query_{i+1}_{query[:30].replace(' ', '_').replace('/', '_')}"
            await url2file(urls, str(query_output_dir))
            
            # Create URL-to-filename mapping for this query
            url_mapping = {}
            for url in urls:
                safe_name = (
                    url
                    .replace("https://", "")
                    .replace("http://", "")
                    .replace("/", "_")
                    .replace("?", "_")
                    .replace("&", "_")
                )[:100]  # Same logic as url2file
                filename = f"{safe_name}.md"
                url_mapping[filename] = url
            
            # Update tracking
            new_urls_added = research_limits.add_processed_urls(urls)
            results["total_urls_fetched"] += len(new_urls_added)
            
            # Add URL mapping to global mapping
            results["global_url_mapping"].update(url_mapping)
            # Also store in global mapping store for access across the system
            global_url_mapping_store.update(url_mapping)
            
            # Step 3: Initialize subagent if enabled
            if use_subagents:
                agent_id = f"AG{i+1:02d}"
                status_tracker.add_agent(agent_id, query, max_count - len(urls), str(query_output_dir), len([q for q in queries_config if q.get('use_subagents', False)]))
                file_count = len(list(query_output_dir.glob("*.md"))) if query_output_dir.exists() else 0
                status_tracker.update_agent(agent_id, 
                                          file_count=file_count,
                                          total_urls_used=len(urls),
                                          urls_remaining=max_count - len(urls))
                results["statistics"]["queries_with_subagents"] += 1
            
            # Track successful query with URL mapping
            results["successful_queries"].append({
                "query": query,
                "index": i,
                "urls_fetched": len(urls),
                "files_created": len(list(query_output_dir.glob("*.md"))) if query_output_dir.exists() else 0,
                "used_subagents": use_subagents,
                "output_directory": str(query_output_dir),
                "agent_id": f"AG{i+1:02d}" if use_subagents else None,
                "url_mapping": url_mapping
            })
            
            results["research_files"].extend([str(f) for f in query_output_dir.glob("*.md")] if query_output_dir.exists() else [])
            results["queries_processed"] += 1
            
        except Exception as e:
            coordinated_console.print(f"[red]  Error processing query: {e}[/red]")
            search_table.add_row(query[:28] + "..." if len(query) > 28 else query, "0", f"[red]Error[/red]")
            results["failed_queries"].append({
                "query": query,
                "error": str(e),
                "index": i
            })
    
    # Display initial search results
    coordinated_console.print("\n")
    coordinated_console.print(search_table)
    
    # Phase 2: Subagent Processing (if any)
    active_subagents = [q for q in results["successful_queries"] if q["used_subagents"]]
    
    if active_subagents:
        coordinated_console.print(f"\n[bold cyan]Phase 2: Subagent Research ({len(active_subagents)} agents)[/bold cyan]")
        
        # Start live status display
        with Live(status_tracker.create_status_table(), refresh_per_second=2) as live:
            # Inform coordinated_console that a Live context is active
            _prev_live = coordinated_console.live_console
            coordinated_console.live_console = live
            try:
                # Lightweight spinner so users know work is ongoing while prints are suppressed
                spinner_chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                spinner_index = 0
                spinner_stop_event = threading.Event()

                def _spinner_updater():
                    import time  # Import time in the function scope
                    nonlocal spinner_index
                    while not spinner_stop_event.is_set():
                        live.update(
                            Group(
                                status_tracker.create_status_table(),
                                Text(f"{spinner_chars[spinner_index % len(spinner_chars)]} Working...", style="dim")
                            )
                        )
                        spinner_index += 1
                        time.sleep(0.1)

                spinner_thread = threading.Thread(target=_spinner_updater, daemon=True)
                spinner_thread.start()

                # Execute subagents SEQUENTIALLY to prevent rate limit conflicts
                subagent_results = []
            
                for query_info in active_subagents:
                    agent_id = query_info["agent_id"]
                    query = query_info["query"]
                    query_output_dir = Path(query_info["output_directory"])
                    remaining_urls = status_tracker.agents[agent_id].urls_remaining
                    
                    if remaining_urls > 0:
                        try:
                            # Execute each subagent sequentially with global rate limiting
                            result = await run_simplified_subagent_research(
                                agent_id=agent_id,
                                query=query,
                                initial_files_dir=query_output_dir,
                                remaining_url_budget=remaining_urls,
                                output_dir=query_output_dir,
                                global_task_context=global_task_context,
                                live_display=live,
                                url_mapping=query_info.get("url_mapping", {}),
                                total_agents=len(active_subagents)  # Fix: Pass total agent count for token limit calculation
                            )
                            subagent_results.append(result)
                        except Exception as e:
                            coordinated_console.print(f"[red]Critical error in subagent {agent_id}: {e}[/red]")
                            subagent_results.append(e)
                    else:
                        # CRITICAL: Force final report generation when URL budget is exhausted (as per flow.md)
                        coordinated_console.print(f"[yellow]🚨 Agent {agent_id} URL budget exhausted - forcing final report with existing materials[/yellow]")
                        try:
                            result = await run_simplified_subagent_research(
                                agent_id=agent_id,
                                query=query,
                                initial_files_dir=query_output_dir,
                                remaining_url_budget=0,  # Force with 0 budget
                                output_dir=query_output_dir,
                                global_task_context=global_task_context,
                                live_display=live,
                                url_mapping=query_info.get("url_mapping", {}),
                                total_agents=len(active_subagents),
                                force_final_report=True  # Flag to force final report
                            )
                            subagent_results.append(result)
                        except Exception as e:
                            coordinated_console.print(f"[red]Critical error in forced final report for {agent_id}: {e}[/red]")
                            subagent_results.append(e)
                
                # Process results (same as before)
                if subagent_results:
                    
                    for i, result in enumerate(subagent_results):
                        if isinstance(result, Exception):
                            coordinated_console.print(f"[red]Subagent {i+1} failed: {result}[/red]")
                        else:
                            results["subagent_reports"].append(result)
                            # Check if subagent created final subreport
                            if result.get("final_subreport_created") and result.get("success"):
                                # Find the subreport file that should have been created
                                subreport_files = list(Path("research_results").glob("final_subreport_*.md"))
                                if subreport_files:
                                    # Get the most recent subreport file
                                    latest_subreport = max(subreport_files, key=lambda p: p.stat().st_mtime)
                                    results["final_reports"].append(str(latest_subreport))
                
                # Final status update
                    spinner_stop_event.set()
                    spinner_thread.join(timeout=1.0)
                    # Show final table without spinner
                live.update(status_tracker.create_status_table())
            finally:
                # Restore previous live console (or None) on exit
                coordinated_console.live_console = _prev_live
    
    # Phase 3: Research Complete - Always ready for final synthesis
    coordinated_console.print(f"\n[bold cyan]Phase 3: Research Complete - Ready for Final Synthesis[/bold cyan]")
    coordinated_console.print(f"[green]✅ Research phase completed successfully[/green]")
    
    if results["final_reports"]:
        coordinated_console.print(f"[yellow]📋 {len(results['final_reports'])} subreports ready for synthesis[/yellow]")
        
        # Add subreport summaries to results for the main agent to use
        subreport_summaries = []
        for report_path in results["final_reports"]:
            if Path(report_path).exists():
                with open(report_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Extract first 500 characters as summary
                    summary = content[:500] + "..." if len(content) > 500 else content
                    subreport_summaries.append({
                        "path": report_path,
                        "summary": summary
                    })
        results["subreport_summaries"] = subreport_summaries
    else:
        coordinated_console.print(f"[yellow]📋 Research files ready for synthesis[/yellow]")
    
    # ALWAYS indicate research is complete and ready for final synthesis
    coordinated_console.print(f"[cyan]💡 Next step: Call write_final_comprehensive_report to create final document[/cyan]")
    results["ready_for_final_synthesis"] = True
    results["research_complete"] = True
    results["next_action"] = "CALL write_final_comprehensive_report() TO CREATE FINAL DOCUMENT"
    results["final_synthesis_instruction"] = "🚨 RESEARCH COMPLETE! DO NOT call research() again. MUST call write_final_comprehensive_report() to create the final document."
    
    # Calculate final statistics
    if results["queries_processed"] > 0:
        results["statistics"]["average_urls_per_query"] = results["total_urls_fetched"] / results["queries_processed"]
    
    results["statistics"]["global_urls_remaining"] = research_limits.get_remaining_urls()
    
    # CRITICAL: Load all file contents for subsequent functions
    all_file_contents = []
    output_path = Path(output_dir)
    
    status_tracker.update_research_progress(current_operation="Loading file contents")
    
    # Load files from all query directories with URL mapping
    for query_info in results["successful_queries"]:
        query_dir = Path(query_info["output_directory"])
        query_url_mapping = query_info.get("url_mapping", {})
        
        if query_dir.exists():
            for file_path in query_dir.glob("*.md"):
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                        content = f.read()
                    
                    # Get the original URL for this file
                    original_url = query_url_mapping.get(file_path.name, f"Unknown URL for {file_path.name}")
                    
                    # Format the content with clear file boundaries and URL info
                    formatted_content = f"=== SOURCE: {original_url} ===\n{content}\n\n"
                    all_file_contents.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "content": content,
                        "formatted_content": formatted_content,
                        "query": query_info["query"],
                        "original_url": original_url
                    })
                    status_tracker.update_research_progress(files_loaded=len(all_file_contents))
                except Exception as e:
                    status_tracker.add_research_update(f"Failed to load {file_path.name}: {str(e)}")
    
    # Load additional research files if they exist
    additional_dir = output_path / "additional_research"
    if additional_dir.exists():
        for file_path in additional_dir.rglob("*.md"):
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                    content = f.read()
                
                # Try to get URL from global mapping, fallback to filename
                original_url = results["global_url_mapping"].get(file_path.name, f"Additional Research File: {file_path.name}")
                
                formatted_content = f"=== SOURCE: {original_url} ===\n{content}\n\n"
                all_file_contents.append({
                    "filename": file_path.name,
                    "path": str(file_path),
                    "content": content,
                    "formatted_content": formatted_content,
                    "query": "additional_research",
                    "original_url": original_url
                })
                status_tracker.update_research_progress(files_loaded=len(all_file_contents))
            except Exception as e:
                status_tracker.add_research_update(f"Failed to load additional file {file_path.name}: {str(e)}")
    
    # Add file contents to results for passing to subsequent functions
    results["file_contents"] = all_file_contents
    results["total_files_loaded"] = len(all_file_contents)
    
    # Create formatted research materials for AI models
    formatted_materials = "\n".join([file_data["formatted_content"] for file_data in all_file_contents])
    results["formatted_research_materials"] = formatted_materials
    
    status_tracker.update_research_progress(current_operation="Files loaded", files_loaded=len(all_file_contents))
    status_tracker.add_research_update(f"Loaded {len(all_file_contents)} research files successfully")
    
    coordinated_console.print(f"[green]✅ Loaded {len(all_file_contents)} research files[/green]")
    
    # DO NOT AUTO-GENERATE REPORT - Let main agent handle final synthesis
    # Research function only provides the data, main agent creates final report
    coordinated_console.print(f"\n[bold cyan]Phase 3: Research Complete[/bold cyan]")
    coordinated_console.print(f"[green]✅ Research completed successfully[/green]")
    coordinated_console.print(f"[cyan]📋 Ready for final synthesis by main agent[/cyan]")
    
    return results


async def run_simplified_subagent_research(
    agent_id: str,
    query: str, 
    initial_files_dir: Path, 
    remaining_url_budget: int,
    output_dir: Path,
    global_task_context: str,
    live_display: Live,
    url_mapping: Dict[str, str] = None,
    total_agents: int = 1,
    force_final_report: bool = False
) -> Dict[str, Any]:
    """
    Enhanced subagent research with proper iteration tracking and decision framework as per flow.md.
    """
    try:
        # Agent should already be initialized by research function
        if agent_id not in status_tracker.agents:
            return {"error": f"Agent {agent_id} not properly initialized", "success": False}
        
        # Update agent status
        status_tracker.update_agent(agent_id, state="🔄 Analyzing", iteration=1)
        live_display.update(status_tracker.create_status_table())
        
        # Log workflow start
        if agent_id in agent_loggers:
            agent_loggers[agent_id].log_workflow_step("WORKFLOW_START", f"Beginning iterative analysis for query: {query}")

        # Create subagent with function calling capabilities
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return {"error": "GEMINI_API_KEY not found", "success": False}

        subagent = GeminiAgent(AgentConfig(
            api_key=api_key,
            model="gemini-2.5-flash",
            system_instruction=sub_agent_prompt
        ))
        
        # Set agent context for proper logging and workflow tracking
        subagent.set_agent_context(agent_id, query)
        
        # Initialize agent logger
        if agent_id not in agent_loggers:
            agent_loggers[agent_id] = AgentLogger(agent_id)
        
        # Get token limit for this agent
        agent_state = status_tracker.agents[agent_id]
        token_limit = agent_state.token_limit
        
        agent_loggers[agent_id].log_workflow_step("AGENT_INITIALIZED", f"Query: {query}, Model: gemini-2.5-flash, Token Limit: {token_limit:,}")

        # **MAIN ITERATIVE LOOP** - Continue until final report or max iterations
        current_iteration = 1
        max_iterations = 10  # Safety limit
        
        # FORCE FINAL REPORT if URL budget is exhausted (as per flow.md)
        if force_final_report:
            coordinated_console.print(f"[yellow]⚠️ Agent {agent_id} forced to generate final report due to URL budget exhaustion[/yellow]")
            agent_loggers[agent_id].log_workflow_step("FORCED_FINAL_REPORT", "URL budget exhausted, forcing final subreport generation")
            
            # Load whatever research materials exist
            research_materials = await load_agent_research_materials(
                agent_id, initial_files_dir, url_mapping, token_limit
            )
            
            # Create forced final report prompt
            forced_prompt = f"""⚠️ CRITICAL: Your URL budget has been exhausted. You MUST create a final subreport with the existing research materials.

**YOUR QUERY:** {query}
**GLOBAL TASK CONTEXT:** {global_task_context}
**STATUS:** URL budget exhausted - no additional research possible

You must immediately call write_final_subreport() with whatever information is available in your research materials.

**RESEARCH MATERIALS:**
{research_materials}

**CURRENT DATE:** {datetime.now().strftime("%Y-%m-%d")}
"""
            
            # Force final report generation
            async with _global_api_lock:
                estimated_tokens = len(forced_prompt) // 4
                limiter = rate_limiters.get("gemini-2.5-flash")
                if limiter:
                    await limiter.wait_if_needed(estimated_tokens)
                
                status_tracker.update_agent(agent_id, state="🚨 Forced Final Report")
                live_display.update(status_tracker.create_status_table())
                
                result = await subagent.generate(forced_prompt)
                
                # The result should trigger write_final_subreport function call
                status_tracker.update_agent(agent_id, state="✅ Forced Complete", completed=True)
                live_display.update(status_tracker.create_status_table())
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "iterations_completed": 1,
                    "decision_history": ["Forced final report due to URL budget exhaustion"],
                    "temp_reports_created": 0,
                    "final_subreport_created": True,
                    "forced_completion": True,
                    "message": f"Agent {agent_id} forced to complete due to URL budget exhaustion"
                }
        
        while current_iteration <= max_iterations:
            
            # Update iteration tracking
            status_tracker.update_agent(agent_id, 
                                      state=f"🔄 Iteration {current_iteration}",
                                      iteration=current_iteration)
            live_display.update(status_tracker.create_status_table())
            
            # Load current research materials (initial files + temp reports)
            research_materials = await load_agent_research_materials(
                agent_id, initial_files_dir, url_mapping, token_limit
            )
            
            # Create iteration-specific prompt following flow.md guidelines (informative, not instructional)
            iteration_prompt = create_iteration_prompt(
                agent_id=agent_id,
                query=query,
                global_task_context=global_task_context,
                current_iteration=current_iteration,
                research_materials=research_materials,
                urls_remaining=agent_state.urls_remaining,
                token_limit=token_limit,
                temp_reports=agent_state.temp_reports
            )

            # Execute subagent with rate limiting
            async with _global_api_lock:
                # Estimate token count for rate limiting
                estimated_tokens = len(iteration_prompt) // 4
                
                # Wait for rate limits with the 2.5-flash model
                limiter = rate_limiters.get("gemini-2.5-flash")
                if limiter:
                    await limiter.wait_if_needed(estimated_tokens)
                
                status_tracker.update_agent(agent_id, state=f"🤖 API Call Iter {current_iteration}")
                live_display.update(status_tracker.create_status_table())
                
                agent_loggers[agent_id].log_workflow_step("API_CALL_START", f"Iteration {current_iteration}, Prompt: {len(iteration_prompt):,} chars")
                
                # Generate response
                result = await subagent.generate(iteration_prompt)
                
                agent_loggers[agent_id].log_workflow_step("API_CALL_COMPLETE", f"Iteration {current_iteration} response: {str(result)[:200]}...")

            # Parse decision from response (looking for SUFFICIENT/INSUFFICIENT as per flow.md)
            decision = parse_agent_decision(result, agent_id, current_iteration)
            agent_state.decision_history.append(f"Iter {current_iteration}: {decision}")
            
            if decision == "FINAL_REPORT_COMPLETED":
                # Agent completed final subreport
                status_tracker.update_agent(agent_id, state="✅ Completed", completed=True)
                live_display.update(status_tracker.create_status_table())
                
                agent_loggers[agent_id].log_completion(f"Final subreport completed in {current_iteration} iterations")
                
                return {
                    "success": True,
                    "agent_id": agent_id,
                    "iterations_completed": current_iteration,
                    "decision_history": agent_state.decision_history,
                    "temp_reports_created": len(agent_state.temp_reports),
                    "final_subreport_created": True,
                    "message": f"Agent {agent_id} completed research in {current_iteration} iterations"
                }
                
            elif decision == "INSUFFICIENT_CONTINUING":
                # Agent created temp report and will continue
                current_iteration += 1
                continue
                
            elif decision == "ERROR":
                # Error in agent processing
                error_msg = f"Agent {agent_id} encountered error in iteration {current_iteration}"
                status_tracker.update_agent(agent_id, state="❌ Error", completed=True, error_details=error_msg)
                live_display.update(status_tracker.create_status_table())
                
                agent_loggers[agent_id].log_error(error_msg)
                
                return {
                    "success": False,
                    "agent_id": agent_id,
                    "error": error_msg,
                    "iterations_completed": current_iteration,
                    "decision_history": agent_state.decision_history
                }
            
            # If we get here, something unexpected happened
            current_iteration += 1
        
        # Max iterations reached
        error_msg = f"Agent {agent_id} reached maximum iterations ({max_iterations}) without completion"
        status_tracker.update_agent(agent_id, state="❌ Max Iterations", completed=True, error_details=error_msg)
        live_display.update(status_tracker.create_status_table())
        
        return {
            "success": False,
            "agent_id": agent_id,
            "error": error_msg,
            "iterations_completed": max_iterations,
            "decision_history": agent_state.decision_history
        }

    except Exception as e:
        error_msg = f"Critical error in subagent {agent_id}: {str(e)}"
        status_tracker.update_agent(agent_id, state=f"❌ Critical Error", completed=True, error_details=error_msg)
        live_display.update(status_tracker.create_status_table())
        
        return {
            "success": False,
            "agent_id": agent_id,
            "error": error_msg,
            "error_type": type(e).__name__
        }


class GeminiAgent:
    """Enhanced Gemini AI agent with streaming support, function calling, and disabled safety settings."""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.client = genai.Client(api_key=config.api_key)
        self._setup_generation_config()
        self.conversation_history = []
        self.last_research_materials = ""  # Store research materials for subsequent function calls
        
        # Initialize main agent logger if this is the main agent
        global main_agent_logger
        if self.config.model == "gemini-2.5-pro" and main_agent_logger is None:
            main_agent_logger = AgentLogger("MainAgent")

    
    def _get_agent_id_from_config(self) -> Optional[str]:
        """Extract agent ID from configuration or system instruction."""
        # Check if this is a subagent by looking at the model type and system instruction
        if self.config.model == "gemini-2.5-flash" and "Sub-Agent" in (self.config.system_instruction or ""):
            # For subagents, try to extract agent ID from recent conversation history or context
            # This is a simple approach - in practice you might store this more explicitly
            return getattr(self, '_agent_id', None)
        return None
    
    def _get_current_query(self) -> str:
        """Get the current query being processed by this agent."""
        # Extract from conversation history or stored context
        return getattr(self, '_current_query', "the assigned research query")
    
    def set_agent_context(self, agent_id: str, query: str):
        """Set agent context for logging and workflow tracking."""
        self._agent_id = agent_id
        self._current_query = query
    
    def _setup_generation_config(self):
        """Setup generation config with disabled safety settings and function calling."""
        safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            )
        ]
        
        # Define function declarations based on agent type
        if self.config.model == "gemini-2.5-pro":
            # Main agent functions
            research_function_declaration = {
                "name": "research",
                "description": "Initiate comprehensive web research with multiple queries and subagents. This starts the research workflow but does not create the final comprehensive report.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries_config": {
                            "type": "array",
                            "description": "List of query configurations for research",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "query": {
                                        "type": "string",
                                        "description": "The search query to research"
                                    },
                                    "init_source_count": {
                                        "type": "integer",
                                        "description": "Initial number of URLs to fetch (max 10)",
                                        "minimum": 1,
                                        "maximum": 10
                                    },
                                    "use_subagents": {
                                        "type": "boolean",
                                        "description": "Whether to use subagents for deeper research on this query"
                                    },
                                    "max_source_count": {
                                        "type": "integer",
                                        "description": "Maximum total URLs this query can use (max 50)",
                                        "minimum": 1,
                                        "maximum": 50
                                    }
                                },
                                "required": ["query", "init_source_count", "use_subagents", "max_source_count"]
                            },
                            "maxItems": 50
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Output directory for research results (optional)",
                            "default": "research_results"
                        },
                        "global_task_context": {
                            "type": "string",
                            "description": "Overall research context/task description to provide to subagents for comprehensive understanding of the overall task",
                            "default": ""
                        }
                    },
                    "required": ["queries_config"]
                }
            }
            
            final_report_function_declaration = {
                "name": "write_final_comprehensive_report",
                "description": "Create the final comprehensive report by synthesizing all research materials. Call this after research is complete to generate the final document. You will have access to all downloaded research files content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "report_content": {
                            "type": "string",
                            "description": "The complete final comprehensive report content written by you, including all analysis, synthesis, and conclusions. This should be a thorough, well-structured report that integrates all research findings from the provided research materials."
                        },
                        "executive_summary": {
                            "type": "string",
                            "description": "A concise executive summary of the key findings and conclusions (3-5 paragraphs) based on the actual research materials provided"
                        },
                        "research_materials": {
                            "type": "string",
                            "description": "All research file contents formatted with clear file boundaries. This parameter will be automatically populated with the actual downloaded content from all research files."
                        },
                        "methodology_notes": {
                            "type": "string",
                            "description": "Brief notes about the research methodology and approach used",
                            "default": ""
                        },
                        "output_dir": {
                            "type": "string",
                            "description": "Directory where the research was conducted",
                            "default": "research_results"
                        }
                    },
                    "required": ["report_content", "executive_summary", "research_materials"]
                }
            }

            tools = [types.Tool(function_declarations=[research_function_declaration, final_report_function_declaration])]

        elif self.config.model == "gemini-2.5-flash":
            # Subagent functions
            temp_report_declaration = {
                "name": "temp_report",
                "description": "WRITE A SUMMARY REPORT of what you learned from analyzing all the research files provided to you. This saves your analysis before you decide if you need more research or can write the final report.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "compressed_report_content": {
                            "type": "string",
                            "description": "Write a comprehensive summary report analyzing ALL the research materials you received. Include: 1) Key findings from the sources, 2) Important facts and insights you discovered, 3) Main themes and trends, 4) Specific information relevant to your query. This should be YOUR ANALYSIS OF THE SOURCES, not a plan or strategy."
                        }
                    },
                    "required": ["compressed_report_content"]
                }
            }

            sub_research_declaration = {
                "name": "sub_research",
                "description": "Perform additional focused research when current information is insufficient. Use this to fill specific information gaps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "research_queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of specific search queries to address information gaps"
                        },
                        "max_urls_per_query": {
                            "type": "integer",
                            "description": "Maximum URLs to fetch per query (default 5)",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 10
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why additional research is needed and what gaps you're trying to fill"
                        }
                    },
                    "required": ["research_queries", "reasoning"]
                }
            }

            final_subreport_declaration = {
                "name": "write_final_subreport",
                "description": "WRITE YOUR FINAL COMPREHENSIVE SUBREPORT based on all the research materials you analyzed. This completes your research assignment.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subreport_content": {
                            "type": "string",
                            "description": "Write your complete final subreport with: 1) Analysis of all research materials, 2) Key findings and insights, 3) Conclusions and recommendations, 4) Evidence from the sources. Make this comprehensive and well-structured."
                        },
                        "key_findings": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List the 3-8 most important findings from your research"
                        }
                    },
                    "required": ["subreport_content", "key_findings"]
                }
            }

            tools = [types.Tool(function_declarations=[temp_report_declaration, sub_research_declaration, final_subreport_declaration])]
        else:
            tools = []  # Other models don't get function calling

        # Configure thinking for the main orchestrator agent only
        thinking_config = None
        if self.config.model == "gemini-2.5-pro":
            thinking_config = types.ThinkingConfig(
                thinking_budget=-1,  # Dynamic thinking - model decides based on complexity
                include_thoughts=True  # Enable thought summaries
            )
        
        self.generation_config = types.GenerateContentConfig(
            system_instruction=self.config.system_instruction,
            safety_settings=safety_settings,
            tools=tools,
            thinking_config=thinking_config
        )
    
    async def _call_gemini_api(self, *, model: str, contents, config):
        """Gemini API call with retry on quota/overload errors and proactive rate limiting."""
        import re, time as _time
        
        # Check circuit breaker first
        breaker = circuit_breakers.get(model)
        if breaker and not breaker.can_proceed():
            wait_time = breaker.should_wait()
            if wait_time > 0:
                coordinated_console.print(f"[bold red]🚫 Circuit breaker is open for {model}. Please wait {wait_time} seconds before retrying.[/bold red]")
                raise Exception(f"Circuit breaker is open for {model}. Wait {wait_time} seconds before retrying.")
        
        # PROACTIVE RATE LIMITING before API call
        content_size = sum(len(str(content)) for content in contents)
        estimated_tokens = content_size // 4 + 1000  # Basic estimation with buffer
        limiter = rate_limiters.get(model)
        if limiter:
            await limiter.wait_if_needed(estimated_tokens)
        
        max_retries = 3
        attempt = 0
        while True:
            try:
                # Log prompt and history before API call
                prompt_history_logger.log_api_call(contents, f"_call_gemini_api({model})", config, model)
                
                response = self.client.models.generate_content(model=model, contents=contents, config=config)
                # Log response to interactions.log
                gemini_response_logger.log_response(response, f"_call_gemini_api({model})")
                
                # Record success with circuit breaker
                if breaker:
                    breaker.record_success()
                    
                return response
            except Exception as err:
                err_str = str(err)
                transient = any(tok in err_str for tok in ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate limit", "quota"])
                
                # Record failure with circuit breaker
                if breaker and transient:
                    breaker.record_failure()
                
                if attempt >= max_retries or not transient:
                    # Persist the prompt for post-mortem analysis
                    try:
                        debug_dir = Path("research_results") / "debug_prompts"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dump_path = debug_dir / f"prompt_dump_{ts}.txt"
                        with open(dump_path, "w", encoding="utf-8") as f:
                            plain_contents = "\n".join(str(c) for c in contents)
                            f.write(plain_contents)
                            f.write(f"\n\n---\nCharacter length: {len(plain_contents)}\nWord count: {len(plain_contents.split())}\n")
                        coordinated_console.print(f"[red]❌ API failure. Prompt saved to {dump_path}[/red]")
                    except Exception:
                        pass
                    raise

                retry_sec = 5 * (attempt + 1)
                match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", err_str)
                if match:
                    try:
                        retry_sec = int(match.group(1))
                    except ValueError:
                        pass

                # Update global API status for live dashboard
                status_tracker.api_status = f"Rate-limit: retry in {retry_sec}s (try {attempt+1}/{max_retries})"
                if coordinated_console.live_console is not None:
                    coordinated_console.live_console.update(
                        Group(
                            status_tracker.create_status_table()
                        )
                    )
                else:
                    coordinated_console.print(f"[yellow]⏳ Gemini API quota hit. Retrying in {retry_sec}s (attempt {attempt+1}/{max_retries})…[/yellow]")
                _time.sleep(retry_sec)
                attempt += 1
                # After sleep, clear status
                status_tracker.api_status = ""
                continue

    async def _call_gemini_api_stream(self, *, model: str, contents, config):
        """Gemini API streaming call with retry on quota/overload errors and proactive rate limiting."""
        import re, time as _time
        
        # Check circuit breaker first
        breaker = circuit_breakers.get(model)
        if breaker and not breaker.can_proceed():
            wait_time = breaker.should_wait()
            if wait_time > 0:
                coordinated_console.print(f"[bold red]🚫 Circuit breaker is open for {model}. Please wait {wait_time} seconds before retrying.[/bold red]")
                raise Exception(f"Circuit breaker is open for {model}. Wait {wait_time} seconds before retrying.")
        
        # PROACTIVE RATE LIMITING before API call
        content_size = sum(len(str(content)) for content in contents)
        estimated_tokens = content_size // 4 + 1000  # Basic estimation with buffer
        limiter = rate_limiters.get(model)
        if limiter:
            await limiter.wait_if_needed(estimated_tokens)
        
        max_retries = 3
        attempt = 0
        while True:
            try:
                # Log prompt and history before API call
                prompt_history_logger.log_api_call(contents, f"_call_gemini_api_stream({model})", config, model)
                
                response_stream = self.client.models.generate_content_stream(model=model, contents=contents, config=config)
                
                # Record success with circuit breaker
                if breaker:
                    breaker.record_success()
                    
                return response_stream
            except Exception as err:
                err_str = str(err)
                transient = any(tok in err_str for tok in ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate limit", "quota"])
                
                # Record failure with circuit breaker
                if breaker and transient:
                    breaker.record_failure()
                
                if attempt >= max_retries or not transient:
                    # Persist the prompt for post-mortem analysis
                    try:
                        debug_dir = Path("research_results") / "debug_prompts"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dump_path = debug_dir / f"prompt_dump_stream_{ts}.txt"
                        with open(dump_path, "w", encoding="utf-8") as f:
                            plain_contents = "\n".join(str(c) for c in contents)
                            f.write(plain_contents)
                            f.write(f"\n\n---\nCharacter length: {len(plain_contents)}\nWord count: {len(plain_contents.split())}\n")
                        coordinated_console.print(f"[red]❌ API streaming failure. Prompt saved to {dump_path}[/red]")
                    except Exception:
                        pass
                    raise

                retry_sec = 5 * (attempt + 1)
                match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", err_str)
                if match:
                    try:
                        retry_sec = int(match.group(1))
                    except ValueError:
                        pass

                # Update global API status for live dashboard
                status_tracker.api_status = f"Streaming rate-limit: retry in {retry_sec}s (try {attempt+1}/{max_retries})"
                if coordinated_console.live_console is not None:
                    coordinated_console.live_console.update(
                        Group(
                            status_tracker.create_status_table()
                        )
                    )
                else:
                    coordinated_console.print(f"[yellow]⏳ Gemini API streaming quota hit. Retrying in {retry_sec}s (attempt {attempt+1}/{max_retries})…[/yellow]")
                await asyncio.sleep(retry_sec)
                attempt += 1
                # After sleep, clear status
                status_tracker.api_status = ""
                continue

    async def generate(self, prompt: str) -> str:
        """Generate complete response for text input with proper function calling support."""
        global main_agent_logger
        try:
            # Log main agent activity
            if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                main_agent_logger.log_workflow_step("GENERATE_START", f"Starting generation with prompt length: {len(prompt)}")
            
            # Use robust wrapper with automatic retry (rate limiting handled in _call_gemini_api)
            response = await self._call_gemini_api(
                model=self.config.model,
                contents=[prompt],
                config=self.generation_config
            )
            
            # Log the complete response to interactions.log
            gemini_response_logger.log_response(response, f"generate({self.config.model})")
            
            # Log response received
            if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                main_agent_logger.log_workflow_step("API_RESPONSE_RECEIVED", f"Response type: {type(response)}")
            
            # ADD DEBUG LOGGING HERE:
            coordinated_console.print(f"[dim cyan]DEBUG: Response received, candidates: {len(response.candidates) if hasattr(response, 'candidates') and response.candidates else 0}[/dim cyan]")
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content'):
                    coordinated_console.print(f"[dim cyan]DEBUG: Content exists, parts: {type(candidate.content.parts) if hasattr(candidate.content, 'parts') else 'No parts attr'}[/dim cyan]")
                    if hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                        coordinated_console.print(f"[dim cyan]DEBUG: Parts count: {len(candidate.content.parts)}[/dim cyan]")
                    else:
                        coordinated_console.print(f"[dim cyan]DEBUG: Parts is None or missing[/dim cyan]")
                else:
                    coordinated_console.print(f"[dim cyan]DEBUG: No content in candidate[/dim cyan]")
            


            # Check for function calls in the response (native function calling)
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') 
                    and candidate.content.parts is not None):
                    
                    # Log parts analysis for main agent
                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                        main_agent_logger.log_workflow_step("CHECKING_PARTS", f"Found {len(candidate.content.parts)} parts in response")
                        for i, part in enumerate(candidate.content.parts):
                            part_type = "unknown"
                            if hasattr(part, 'function_call') and part.function_call:
                                part_type = f"function_call: {part.function_call.name}"
                            elif hasattr(part, 'text') and part.text:
                                part_type = f"text: {len(part.text)} chars"
                            elif hasattr(part, 'thought') and part.thought:
                                part_type = "thought"
                            main_agent_logger.log_workflow_step("PART_ANALYSIS", f"Part {i}: {part_type}")
                    
                    for part in candidate.content.parts:
                        # Check if this part contains a function call
                        if hasattr(part, 'function_call') and part.function_call:
                            function_call = part.function_call
                            
                            # Log function call detection
                            if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                main_agent_logger.log_workflow_step("FUNCTION_CALL_DETECTED", f"Function: {function_call.name}")
                                main_agent_logger.log_workflow_step("FUNCTION_ARGS", f"Args: {function_call.args}")
                            
                            # Execute the appropriate function based on name
                            if function_call.name == "research":
                                try:
                                    coordinated_console.print(f"[cyan]🤖 AI Model requesting research function call[/cyan]")
                                    
                                    # Extract and validate parameters 
                                    args = {}
                                    if hasattr(function_call, 'args'):
                                        args = function_call.args
                                    
                                    # Call the actual research function implementation
                                    result = await research_function(**args)
                                    
                                    coordinated_console.print(f"[green]✅ Research function completed successfully[/green]")

                                    # Reduce payload size sent back to the LLM to avoid hitting input
                                    # token limits. Keep only essential, lightweight metadata and omit
                                    # large textual fields suchs as raw file contents.
                                    lightweight_keys = {
                                        "queries_processed", "total_urls_fetched", "successful_queries",
                                        "failed_queries", "final_report_generated", "final_report_path",
                                        "statistics", "display_report"
                                    }
                                    safe_result = {k: v for k, v in result.items() if k in lightweight_keys}

                                    # Create function response to send back to model with trimmed data
                                    function_response = types.FunctionResponse(
                                        name="research",
                                        response=safe_result
                                    )
                                    
                                    # Decide follow-up contents structure based on model version
                                    if self.config.model.startswith("gemini-2.5"):
                                        follow_up_contents = [
                                            prompt,
                                            types.Content(parts=[types.Part(function_call=function_call)]),
                                            types.Content(parts=[types.Part(function_response=function_response)])
                                        ]
                                    else:
                                        follow_up_contents = [
                                            prompt,
                                            types.Content(parts=[
                                                types.Part(function_call=function_call),
                                                types.Part(function_response=function_response)
                                            ])
                                        ]

                                    # Send function result back to model for final response with retry support
                                    follow_up_response = await self._call_gemini_api(
                                        model=self.config.model,
                                        contents=follow_up_contents,
                                        config=types.GenerateContentConfig(
                                            system_instruction=self.config.system_instruction,
                                            safety_settings=self.generation_config.safety_settings,
                                            tools=self.generation_config.tools  # Keep tools available so model can call write_final_comprehensive_report
                                        )
                                    )
                                    # Log follow-up response
                                    gemini_response_logger.log_response(follow_up_response, f"research_followup({self.config.model})")
                                    # Handle the follow-up response through normal workflow to allow subsequent function calls
                                    return await self._handle_workflow_response(follow_up_response, follow_up_contents)
                                    
                                except Exception as e:
                                    coordinated_console.print(f"[red]❌ Error executing research function: {e}[/red]")
                                    return f"I encountered an error while conducting the research: {str(e)}. Please try rephrasing your request or asking for a simpler research task."

                            elif function_call.name == "write_final_comprehensive_report":
                                try:
                                    # Log detailed progress for main agent
                                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                        main_agent_logger.log_workflow_step("FINAL_REPORT_START", "Starting write_final_comprehensive_report execution")
                                    
                                    coordinated_console.print(f"[cyan]🤖 AI Model requesting final report writing[/cyan]")

                                    # Extract parameters
                                    args = {}
                                    if hasattr(function_call, 'args'):
                                        args = function_call.args
                                    
                                    # Log extracted args
                                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                        main_agent_logger.log_workflow_step("ARGS_EXTRACTED", f"Extracted args keys: {list(args.keys())}")

                                    # CRITICAL: Inject research materials for native function calls
                                    if "research_materials" not in args or not args["research_materials"]:
                                        args["research_materials"] = self.last_research_materials
                                        coordinated_console.print(f"[dim cyan]📖 Injecting research materials ({len(self.last_research_materials)} chars) into native function call[/dim cyan]")
                                        if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                            main_agent_logger.log_workflow_step("MATERIALS_INJECTED", f"Injected {len(self.last_research_materials)} chars of research materials")
                                    else:
                                        coordinated_console.print(f"[dim cyan]📖 Using provided research materials ({len(args['research_materials'])} chars) from native function call[/dim cyan]")
                                        if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                            main_agent_logger.log_workflow_step("MATERIALS_PROVIDED", f"Using provided {len(args['research_materials'])} chars of research materials")

                                    # Log before function call
                                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                        main_agent_logger.log_workflow_step("CALLING_FUNCTION", "About to call write_final_comprehensive_report_function")
                                    
                                    # Call the new final report function
                                    result = await write_final_comprehensive_report_function(**args)
                                    
                                    # Log after function call
                                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                        main_agent_logger.log_workflow_step("FUNCTION_COMPLETED", f"write_final_comprehensive_report_function returned: {type(result)}")

                                    coordinated_console.print(f"[green]✅ Final comprehensive report created successfully[/green]")

                                    # Create function response
                                    function_response = types.FunctionResponse(
                                        name="write_final_comprehensive_report",
                                        response=result
                                    )

                                    # Log before follow-up call
                                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                        main_agent_logger.log_workflow_step("FOLLOWUP_START", "Starting follow-up API call with function response")
                                    
                                    # Send function result back to model for final response with retry support
                                    follow_up_contents = [
                                        prompt,
                                        types.Content(parts=[
                                            types.Part(function_call=function_call),
                                            types.Part(function_response=function_response)
                                        ])
                                    ]
                                    
                                    # keep all original tools available
                                    follow_up_response = await self._call_gemini_api(
                                        model=self.config.model,
                                        contents=follow_up_contents,
                                        config=types.GenerateContentConfig(
                                            system_instruction=self.config.system_instruction,
                                            safety_settings=self.generation_config.safety_settings,
                                            tools=self.generation_config.tools   # instead of []
                                        )
                                    )
                                    # Log follow-up response for final report
                                    gemini_response_logger.log_response(follow_up_response, f"final_report_followup({self.config.model})")
                                    
                                    # Log follow-up response
                                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                        with suppress_gemini_warnings():
                                            main_agent_logger.log_workflow_step("FOLLOWUP_COMPLETE", f"Follow-up response received: {len(follow_up_response.text) if hasattr(follow_up_response, 'text') else 0} chars")
                                    
                                    # THEN pipe the answer through the regular function-handling path
                                    return await self._handle_workflow_response(follow_up_response, follow_up_contents)

                                except Exception as e:
                                    # Log error for main agent
                                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                                        main_agent_logger.log_error(f"Error in write_final_comprehensive_report: {e}")
                                    coordinated_console.print(f"[red]❌ Error writing final report: {e}[/red]")
                                    return f"I encountered an error while writing the final report: {str(e)}. Please check the research results and try again."

                            elif function_call.name == "temp_report":
                                try:
                                    coordinated_console.print(f"[cyan]🤖 Subagent creating temporary report[/cyan]")

                                    args = {}
                                    if hasattr(function_call, 'args'):
                                        args = function_call.args

                                    result = await temp_report_function_new(**args)

                                    coordinated_console.print(f"[green]✅ Temporary report created[/green]")
                                    
                                    # Log workflow progress
                                    agent_id = self._get_agent_id_from_config()
                                    if agent_id and agent_id in agent_loggers:
                                        agent_loggers[agent_id].log_workflow_step("TEMP_REPORT_COMPLETE", "temp_report completed via native function call, continuing workflow")
                                    
                                    # WORKFLOW CONTINUATION: Create clean context with ONLY temp report
                                    function_response = types.FunctionResponse(
                                        name="temp_report",
                                        response=result
                                    )
                                    
                                    # Get the temp report content that was just created
                                    temp_report_content = args.get("compressed_report_content", "Temporary report created")
                                    agent_id = self._get_agent_id_from_config()
                                    current_query = self._get_current_query()
                                    
                                    # Create NEW context with ONLY temp report (removing original research files)
                                    clean_prompt = f"""You have created a temp report analyzing research for: "{current_query}"

Is the information sufficient to fully answer the query?

- If INSUFFICIENT: Call sub_research() with specific queries
- If SUFFICIENT: Call write_final_subreport() with your analysis"""

                                    # Create clean conversation with temp report context only
                                    clean_contents = [
                                        types.Content(
                                            parts=[types.Part(text=clean_prompt)],
                                            role="user"
                                        )
                                    ]

                                    # Continue conversation with CLEAN context (no original research files)
                                    follow_up_response = await self._call_gemini_api(
                                        model=self.config.model,
                                        contents=clean_contents,
                                        config=types.GenerateContentConfig(
                                            system_instruction=self.config.system_instruction,
                                            safety_settings=self.generation_config.safety_settings,
                                            tools=self.generation_config.tools  # Keep tools for next function call
                                        )
                                    )
                                    # Log temp report follow-up response
                                    gemini_response_logger.log_response(follow_up_response, f"temp_report_followup({self.config.model})")
                                    
                                    # Handle the follow-up response (which should be sub_research or write_final_subreport)
                                    return await self._handle_workflow_response(follow_up_response, clean_contents)

                                except Exception as e:
                                    coordinated_console.print(f"[red]❌ Error creating temp report: {e}[/red]")
                                    return f"I encountered an error while creating the temporary report: {str(e)}."

                            elif function_call.name == "sub_research":
                                try:
                                    coordinated_console.print(f"[cyan]🤖 Subagent performing additional research[/cyan]")

                                    args = {}
                                    if hasattr(function_call, 'args'):
                                        args = function_call.args

                                    result = await sub_research_function_new(**args)

                                    coordinated_console.print(f"[green]✅ Additional research completed[/green]")
                                    
                                    # Log workflow progress
                                    agent_id = self._get_agent_id_from_config()
                                    if agent_id and agent_id in agent_loggers:
                                        agent_loggers[agent_id].log_workflow_step("SUB_RESEARCH_COMPLETE", "sub_research completed via native function call, continuing workflow")
                                    
                                    # WORKFLOW CONTINUATION: After sub_research, load updated research materials
                                    
                                    # Reload research materials to include new files from sub_research
                                    await self._reload_research_materials()
                                    current_query = self._get_current_query()
                                    
                                    # Create clean prompt with updated research materials
                                    clean_prompt = f"""Additional research completed for: "{current_query}"
                                    
You now have updated research materials. Is the information sufficient to answer the query?

- If INSUFFICIENT: Call sub_research() again with different queries
- If SUFFICIENT: Call write_final_subreport() with your analysis

**RESEARCH MATERIALS:**
{self.last_research_materials if self.last_research_materials else "No research materials available"}
"""

                                    # Create clean conversation with updated research materials
                                    clean_contents = [
                                        types.Content(
                                            parts=[types.Part(text=clean_prompt)],
                                            role="user"
                                        )
                                    ]

                                    # Continue conversation with updated context
                                    follow_up_response = await self._call_gemini_api(
                                        model=self.config.model,
                                        contents=clean_contents,
                                        config=types.GenerateContentConfig(
                                            system_instruction=self.config.system_instruction,
                                            safety_settings=self.generation_config.safety_settings,
                                            tools=self.generation_config.tools  # Keep tools for next function call
                                        )
                                    )
                                    # Log sub research follow-up response
                                    gemini_response_logger.log_response(follow_up_response, f"sub_research_followup({self.config.model})")
                                    
                                    # Handle the follow-up response (which should be sub_research or write_final_subreport)
                                    return await self._handle_workflow_response(follow_up_response, clean_contents)

                                except Exception as e:
                                    coordinated_console.print(f"[red]❌ Error in additional research: {e}[/red]")
                                    return f"I encountered an error while conducting additional research: {str(e)}."

                            elif function_call.name == "write_final_subreport":
                                try:
                                    coordinated_console.print(f"[cyan]🤖 Subagent writing final subreport[/cyan]")

                                    args = {}
                                    if hasattr(function_call, 'args'):
                                        args = function_call.args

                                    result = await write_final_subreport_function_new(**args)

                                    coordinated_console.print(f"[green]✅ Final subreport completed[/green]")
                                    
                                    # Log workflow completion
                                    agent_id = self._get_agent_id_from_config()
                                    if agent_id and agent_id in agent_loggers:
                                        agent_loggers[agent_id].log_completion(f"Final subreport completed successfully via native function call")
                                    
                                    # WORKFLOW COMPLETION: Return success message
                                    return f"✅ Agent {agent_id} workflow completed successfully. Final subreport has been generated."
                                    
                                except Exception as e:
                                    coordinated_console.print(f"[red]❌ Error writing final subreport: {e}[/red]")
                                    return f"I encountered an error while writing the final subreport: {str(e)}."
            
            # Enhanced error handling and fallback for empty response
            try:
                # Log main agent response processing
                if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                    with suppress_gemini_warnings():
                        has_text = hasattr(response, 'text') and response.text is not None
                        main_agent_logger.log_workflow_step("PROCESSING_RESPONSE", f"Has text: {has_text}, Response text length: {len(response.text) if has_text else 0}")
                    main_agent_logger.log_workflow_step("REACHED_TEXT_PROCESSING", "Reached text processing section")
                
                # If no function calls, return the regular text response
                with suppress_gemini_warnings():
                    if not hasattr(response, 'text') or response.text is None:
                        # Log empty response for main agent
                        if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                            main_agent_logger.log_error("Empty text response received from API")
                        coordinated_console.print(f"[yellow]⚠️ No text response received from API[/yellow]")
                    
                    # EMERGENCY: Check if this is a subagent that should have called temp_report
                    if (self.config.model == "gemini-2.5-flash" and 
                        "temp_report" in str(prompt).lower() and 
                        "initial research files" in str(prompt).lower()):
                        coordinated_console.print(f"[cyan]🚨 Emergency: Executing temp_report for empty response[/cyan]")
                        try:
                            result = await self._execute_function_call("temp_report", {
                                "compressed_report_content": "Emergency temp report due to API response failure. Initial research files analyzed and compressed."
                            })
                            return f"Emergency temp_report executed due to API failure. Result: {result}"
                        except Exception as e:
                            coordinated_console.print(f"[red]❌ Emergency temp_report failed: {e}[/red]")
                    
                    return "I received an empty response from the API. Please try again."
                
            except Exception as e:
                # Log processing error for main agent
                if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                    main_agent_logger.log_error(f"Error processing response parts: {e}")
                coordinated_console.print(f"[red]DEBUG: Error processing response parts: {e}[/red]")
                coordinated_console.print(f"[red]Response type: {type(response)}[/red]")
                if hasattr(response, 'candidates'):
                    coordinated_console.print(f"[red]Candidates: {response.candidates}[/red]")
                raise  # Re-raise for now to get full traceback
            
        except Exception as e:
            # Log generation error for main agent
            if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                main_agent_logger.log_error(f"Error in agent generation: {e}")
            coordinated_console.print(f"[red]❌ Error in agent generation: {e}[/red]")
            return f"I encountered an error while processing your request: {str(e)}. Please try again."
    
    async def chat_stream(self, message: str) -> None:
        """Stream chat response with conversation history, thinking display, and function calling support."""
        global main_agent_logger
        
        # Log chat stream start for main agent
        if self.config.model == "gemini-2.5-pro" and main_agent_logger:
            main_agent_logger.log_workflow_step("CHAT_STREAM_START", f"Message length: {len(message)}")
        
        self.conversation_history.append({"role": "user", "content": message})
        
        # Build conversation context
        context = self._build_context()
        
        try:
            # Only show thinking for main orchestrator agent (gemini-2.5-pro)
            if self.config.model == "gemini-2.5-pro":
                coordinated_console.print("\n[bold bright_magenta]🧠 Thinking:[/bold bright_magenta]")
                
                # Start thinking animation
                thinking_name = "main_thinking"
                
                def show_thinking_animation():
                    """Display animated thinking indicator"""
                    import time  # Import time in the function scope
                    # Don't start if higher-level animation is suppressing us
                    if animation_manager.should_suppress_lower_level():
                        return
                    
                    if not animation_manager.start_animation(thinking_name, priority=1):
                        return  # Another animation is already active
                    
                    chars = "🧠💭🤔💡"
                    i = 0
                    try:
                        while animation_manager.is_animation_active(thinking_name):
                            safe_builtin_print(f"\r{chars[i % len(chars)]} Analyzing and planning...", end="", flush=True)
                            time.sleep(0.5)
                            i += 1
                    finally:
                        animation_manager.stop_animation(thinking_name)
                        safe_builtin_print("\r" + " " * 120 + "\r", end="", flush=True)  # Clear the line
                
                # Start thinking animation thread
                thinking_thread = threading.Thread(target=show_thinking_animation)
                thinking_thread.daemon = True
                thinking_thread.start()
                animation_manager.register_thread(thinking_name, thinking_thread)
            
            # Log streaming start
            if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                main_agent_logger.log_workflow_step("STREAMING_START", f"Starting content stream with {len(context)} context parts")
            
            # APPLY RATE LIMITING before streaming API call
            context_size = sum(len(str(content)) for content in context)
            estimated_tokens = context_size // 4 + 2000  # Basic estimation with buffer
            limiter = rate_limiters.get(self.config.model)
            if limiter:
                await limiter.wait_if_needed(estimated_tokens)
            
            # Use robust streaming wrapper with automatic retry
            response_stream = await self._call_gemini_api_stream(
                model=self.config.model,
                contents=context,
                config=self.generation_config
            )
            
            # Collect the complete response from streaming
            response = None
            thinking_parts = []
            response_parts = []
            all_text_parts = []  # Collect all non-thinking text parts
            
            current_thought = ""
            chunk_count = 0
            streaming_complete = False
            
            # Log stream processing start
            if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                main_agent_logger.log_workflow_step("PROCESSING_CHUNKS", "Starting to process response chunks")
            
            for chunk in response_stream:
                chunk_count += 1
                # Log every response chunk to interactions.log
                gemini_response_logger.log_response(chunk, f"chat_stream_chunk_{chunk_count}({self.config.model})")
                # Log every 10th chunk to track progress
                if self.config.model == "gemini-2.5-pro" and main_agent_logger and chunk_count % 10 == 0:
                    main_agent_logger.log_workflow_step("CHUNK_PROGRESS", f"Processed {chunk_count} chunks")
                    
                # Log chunk details for debugging
                if self.config.model == "gemini-2.5-pro" and main_agent_logger and chunk_count <= 3:
                    has_candidates = hasattr(chunk, 'candidates') and chunk.candidates
                    main_agent_logger.log_workflow_step("CHUNK_DEBUG", f"Chunk {chunk_count}: has_candidates={has_candidates}")
                
                if (hasattr(chunk, 'candidates') and chunk.candidates and 
                    len(chunk.candidates) > 0 and chunk.candidates[0] is not None):
                    candidate = chunk.candidates[0]
                    if (hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts):
                        for part in candidate.content.parts:
                            if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
                                # Stream thinking in real-time for main agent only
                                if self.config.model == "gemini-2.5-pro":
                                    current_thought += part.text
                                    coordinated_console.print(f"[dim bright_magenta]{part.text}[/dim bright_magenta]", end="")
                                thinking_parts.append(part.text)
                            elif hasattr(part, 'text') and part.text and not getattr(part, 'thought', False):
                                # Collect all non-thinking text parts (don't stream live, we'll show in Answer section)
                                all_text_parts.append(part.text)
                                response_parts.append(part.text)
                            elif hasattr(part, 'function_call') and part.function_call:
                                response = chunk  # Save the response with function calls
                response = chunk  # Keep updating with latest chunk
            
            # CRITICAL: Mark streaming as complete and ensure all content is flushed
            streaming_complete = True
            import sys
            sys.stdout.flush()  # Force flush all output to terminal
            
            # Give terminal time to render all content before any cleanup
            time.sleep(0.1)  # Small delay to ensure terminal rendering is complete
            
            if current_thought and self.config.model == "gemini-2.5-pro":
                coordinated_console.print("\n")
            
            # Show Answer section and protect the response content
            if all_text_parts:
                # Start Answer section
                coordinated_console.print("\n[bold bright_blue]💬 Answer:[/bold bright_blue]")
                
                # Display the complete answer content in a protected way
                complete_answer = "".join(all_text_parts)
                console.print(complete_answer, style="bright_white")
                
                # Ensure answer section is properly terminated
                console.print()  # Final newline to complete the answer section
                sys.stdout.flush()  # Ensure everything is rendered
                time.sleep(0.1)  # Give terminal time to render completely
            
            # ONLY stop animations AFTER streaming is complete and content is rendered
            if self.config.model == "gemini-2.5-pro" and 'thinking_name' in locals() and streaming_complete:
                animation_manager.stop_animation(thinking_name)
                # Wait a bit more before clearing animation line to avoid race condition
                time.sleep(0.1)
                if not animation_manager.should_suppress_lower_level():
                    safe_builtin_print(f"\r" + " " * 120 + "\r", end="", flush=True)  # Clear animation line
            
            # Check for function calls in the response
            function_calls = []
            if (hasattr(response, 'candidates') and response.candidates and 
                len(response.candidates) > 0 and response.candidates[0] is not None):
                candidate = response.candidates[0]
                if (hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            function_calls.append(part.function_call)
            
            # Handle function calls if present
            if function_calls:
                # Log function calls detection for main agent
                if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                    func_names = [fc.name for fc in function_calls]
                    main_agent_logger.log_workflow_step("FUNCTION_CALLS_DETECTED", f"Functions: {func_names}")
                
                coordinated_console.print(f"\n[bold bright_yellow]╭─────────────────────────────────────────────────────────╮[/bold bright_yellow]")
                coordinated_console.print(f"[bold bright_yellow]│  🤖 AI FUNCTION EXECUTION PIPELINE ACTIVATED           │[/bold bright_yellow]")
                coordinated_console.print(f"[bold bright_yellow]│  📊 Functions to execute: {len(function_calls):2d}                           │[/bold bright_yellow]")
                coordinated_console.print(f"[bold bright_yellow]╰─────────────────────────────────────────────────────────╯[/bold bright_yellow]")
                
                # Show processing status with animation (only if no higher-level animation)
                processing_animation_name = "function_processing"
                
                def show_processing_animation():
                    """Display animated processing indicator"""
                    import time  # Import time in the function scope
                    # Don't start if higher-level animation is suppressing us
                    if animation_manager.should_suppress_lower_level():
                        return
                    
                    if not animation_manager.start_animation(processing_animation_name, priority=1):
                        return  # Another animation is already active, skip this one
                    
                    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                    i = 0
                    try:
                        while animation_manager.is_animation_active(processing_animation_name):
                            safe_builtin_print(f"\r{chars[i % len(chars)]} Processing function calls...", end="", flush=True)
                            time.sleep(0.1)
                            i += 1
                    finally:
                        animation_manager.stop_animation(processing_animation_name)
                        safe_builtin_print("\r" + " " * 120 + "\r", end="", flush=True)  # Clear the line
                
                # Start animation thread
                animation_thread = threading.Thread(target=show_processing_animation)
                animation_thread.daemon = True
                animation_thread.start()
                animation_manager.register_thread(processing_animation_name, animation_thread)
                
                # Add the assistant response to conversation history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response.candidates[0].content.parts if hasattr(response, 'candidates') else []
                })
                
                # Execute each function call and collect results
                function_responses = []
                for func_call in function_calls:
                    func_name = func_call.name
                    func_args = func_call.args
                    
                    coordinated_console.print(f"[bold cyan]🚀 Executing: {func_name}[/bold cyan]")
                    
                    # Execute the function
                    result = await self._execute_function_call(func_name, func_args)
                    
                    function_responses.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response=result
                        )
                    ))
                
                # Stop processing animation when function execution completes
                animation_manager.stop_animation(processing_animation_name)
                safe_builtin_print("\r" + " " * 120 + "\r", end="", flush=True)  # Clear animation line
                
                # Get model's follow-up response after function execution
                coordinated_console.print(f"\n[bold bright_blue]🤖 Processing function results...[/bold bright_blue]")
                
                # Send function results back to model
                follow_up_context = self._build_context()
                follow_up_context.append(types.Content(
                    role="user",
                    parts=function_responses
                ))
                
                # Stop processing animation
                animation_manager.stop_animation(processing_animation_name)
                if not animation_manager.should_suppress_lower_level():
                    safe_builtin_print(f"\r✅ Function execution completed" + " " * 20)
                    safe_builtin_print()  # Add a new line after completion
                
                # Continue streaming the follow-up response (with recursive function call handling)
                await self._stream_response_with_function_handling(follow_up_context)
                
            else:
                # No function calls, just add to conversation history
                complete_response_text = "".join(all_text_parts) if all_text_parts else ""
                if complete_response_text:
                    # Log successful response for main agent
                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                        main_agent_logger.log_workflow_step("STREAM_RESPONSE_SUCCESS", f"Response length: {len(complete_response_text)}")
                    
                    self.conversation_history.append({"role": "assistant", "content": complete_response_text})
                    console.print(f"\n[dim green]✅ Response completed[/dim green]")
                else:
                    # Debug information when no response is received
                    if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                        main_agent_logger.log_error(f"No text response in stream - Thinking: {len(thinking_parts)}, Response: {len(response_parts)}, All text: {len(all_text_parts)}")
                    
                    coordinated_console.print(f"\n\n[dim red]🔍 DEBUG: No text response received[/dim red]")
                    coordinated_console.print(f"[dim red]   - Thinking parts collected: {len(thinking_parts)}[/dim red]")
                    coordinated_console.print(f"[dim red]   - Response parts collected: {len(response_parts)}[/dim red]")
                    coordinated_console.print(f"[dim red]   - All text parts collected: {len(all_text_parts)}[/dim red]")
                    if hasattr(response, 'candidates') and response.candidates:
                        coordinated_console.print(f"[dim red]   - Final response has candidates: {len(response.candidates)}[/dim red]")
                        if response.candidates[0] and hasattr(response.candidates[0], 'content'):
                            coordinated_console.print(f"[dim red]   - Final response has content: {response.candidates[0].content is not None}[/dim red]")
                            if response.candidates[0].content and hasattr(response.candidates[0].content, 'parts'):
                                coordinated_console.print(f"[dim red]   - Final response parts: {len(response.candidates[0].content.parts) if response.candidates[0].content.parts else 0}[/dim red]")
                    coordinated_console.print(f"\n[dim yellow]💡 This suggests the model may be generating only thinking content.[/dim yellow]")
                    coordinated_console.print(f"[dim yellow]   The model should generate function calls or response text after thinking.[/dim yellow]")
            
        except Exception as e:
            # Log streaming error for main agent
            if self.config.model == "gemini-2.5-pro" and main_agent_logger:
                main_agent_logger.log_error(f"Error during streaming: {e}")
            
            err_str = str(e)
            # Check if this is a rate limit error that should be retried
            if any(tok in err_str for tok in ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate limit", "quota"]):
                coordinated_console.print(f"\n[bold red]❌ API quota/rate limit error during streaming: {str(e)}[/bold red]")
                coordinated_console.print(f"[dim red]The system has already attempted retries. Please wait before sending another request.[/dim red]")
            else:
                coordinated_console.print(f"\n[bold red]❌ Error during streaming: {str(e)}[/bold red]")

    async def _stream_response_with_function_handling(self, context):
        """Helper method to handle streaming with recursive function call support."""
        try:
            # Only show thinking for main orchestrator agent (gemini-2.5-pro)
            if self.config.model == "gemini-2.5-pro":
                coordinated_console.print("\n[bold bright_magenta]🧠 Thinking about results:[/bold bright_magenta]")
                
                # Start follow-up thinking animation
                followup_thinking_name = "followup_thinking"
                
                def show_followup_thinking_animation():
                    """Display animated thinking indicator for follow-up"""
                    import time  # Import time in the function scope
                    # Don't start if higher-level animation is suppressing us
                    if animation_manager.should_suppress_lower_level():
                        return
                    
                    if not animation_manager.start_animation(followup_thinking_name, priority=1):
                        return  # Another animation is already active
                    
                    chars = "🔄⚡💫✨"
                    i = 0
                    try:
                        while animation_manager.is_animation_active(followup_thinking_name):
                            safe_builtin_print(f"\r{chars[i % len(chars)]} Processing results and deciding next steps...", end="", flush=True)
                            time.sleep(0.4)
                            i += 1
                    finally:
                        animation_manager.stop_animation(followup_thinking_name)
                        safe_builtin_print("\r" + " " * 120 + "\r", end="", flush=True)  # Clear the line
                
                # Start follow-up thinking animation thread
                followup_thinking_thread = threading.Thread(target=show_followup_thinking_animation)
                followup_thinking_thread.daemon = True
                followup_thinking_thread.start()
                animation_manager.register_thread(followup_thinking_name, followup_thinking_thread)
            
            # APPLY RATE LIMITING before follow-up streaming API call
            context_size = sum(len(str(content)) for content in context)
            estimated_tokens = context_size // 4 + 2000  # Basic estimation with buffer
            limiter = rate_limiters.get(self.config.model)
            if limiter:
                await limiter.wait_if_needed(estimated_tokens)
            
            # Use robust streaming wrapper with automatic retry for follow-up calls
            response_stream = await self._call_gemini_api_stream(
                model=self.config.model,
                contents=context,
                config=self.generation_config
            )
            
            current_response = ""
            function_calls = []
            followup_streaming_complete = False
            followup_chunk_count = 0
            
            for chunk in response_stream:
                followup_chunk_count += 1
                # Log every followup response chunk to interactions.log
                gemini_response_logger.log_response(chunk, f"followup_stream_chunk_{followup_chunk_count}({self.config.model})")
                if (hasattr(chunk, 'candidates') and chunk.candidates and 
                    len(chunk.candidates) > 0 and chunk.candidates[0] is not None):
                    candidate = chunk.candidates[0]
                    if (hasattr(candidate, 'content') and hasattr(candidate.content, 'parts') and candidate.content.parts):
                        for part in candidate.content.parts:
                            if hasattr(part, 'thought') and part.thought and hasattr(part, 'text') and part.text:
                                # Stop follow-up thinking animation when actual thinking starts
                                if self.config.model == "gemini-2.5-pro" and 'followup_thinking_name' in locals():
                                    animation_manager.stop_animation(followup_thinking_name)
                                    if not animation_manager.should_suppress_lower_level():
                                        safe_builtin_print(f"\r" + " " * 120 + "\r", end="", flush=True)  # Clear animation line
                                # Stream thinking in real-time for main agent only
                                if self.config.model == "gemini-2.5-pro":
                                    coordinated_console.print(f"[dim bright_magenta]{part.text}[/dim bright_magenta]", end="")
                            elif hasattr(part, 'text') and part.text and not getattr(part, 'thought', False):
                                # Collect regular text (don't stream live, we'll show in Answer section)
                                current_response += part.text
                            elif hasattr(part, 'function_call') and part.function_call:
                                # Collect function calls
                                function_calls.append(part.function_call)
                response = chunk  # Keep updating with latest chunk
            
            # CRITICAL: Mark follow-up streaming as complete and ensure content is flushed
            followup_streaming_complete = True
            import sys
            sys.stdout.flush()  # Force flush all output to terminal
            time.sleep(0.1)  # Ensure terminal rendering is complete
            
            # Handle function calls recursively if present
            if function_calls:
                coordinated_console.print(f"\n[bold bright_yellow]╭─────────────────────────────────────────────────────────╮[/bold bright_yellow]")
                coordinated_console.print(f"[bold bright_yellow]│  🔄 FOLLOW-UP FUNCTION EXECUTION DETECTED              │[/bold bright_yellow]")
                coordinated_console.print(f"[bold bright_yellow]│  📊 Additional functions to execute: {len(function_calls):2d}               │[/bold bright_yellow]")
                coordinated_console.print(f"[bold bright_yellow]╰─────────────────────────────────────────────────────────╯[/bold bright_yellow]")
                
                # Show processing status with animation for follow-up calls
                followup_processing_name = "followup_processing"
                
                def show_followup_animation():
                    """Display animated processing indicator for follow-up calls"""
                    import time  # Import time in the function scope
                    # Don't start if higher-level animation is suppressing us
                    if animation_manager.should_suppress_lower_level():
                        return
                    
                    if not animation_manager.start_animation(followup_processing_name, priority=1):
                        return  # Another animation is already active
                    
                    chars = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
                    i = 0
                    try:
                        while animation_manager.is_animation_active(followup_processing_name):
                            safe_builtin_print(f"\r{chars[i % len(chars)]} Processing follow-up function calls...", end="", flush=True)
                            time.sleep(0.1)
                            i += 1
                    finally:
                        animation_manager.stop_animation(followup_processing_name)
                        safe_builtin_print("\r" + " " * 120 + "\r", end="", flush=True)  # Clear the line
                
                # Start follow-up animation thread
                followup_animation_thread = threading.Thread(target=show_followup_animation)
                followup_animation_thread.daemon = True
                followup_animation_thread.start()
                animation_manager.register_thread(followup_processing_name, followup_animation_thread)
                
                # Add the assistant response to conversation history
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": response.candidates[0].content.parts if hasattr(response, 'candidates') else []
                })
                
                # Execute each function call and collect results
                function_responses = []
                for func_call in function_calls:
                    func_name = func_call.name
                    func_args = func_call.args
                    
                    coordinated_console.print(f"[bold cyan]🚀 Executing: {func_name}[/bold cyan]")
                    
                    # Execute the function
                    result = await self._execute_function_call(func_name, func_args)
                    
                    function_responses.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=func_name,
                            response=result
                        )
                    ))
                
                # Get model's follow-up response after function execution
                coordinated_console.print(f"\n[bold bright_blue]🤖 Processing additional function results...[/bold bright_blue]")
                
                # Send function results back to model (recursive call)
                follow_up_context = self._build_context()
                follow_up_context.append(types.Content(
                    role="user",
                    parts=function_responses
                ))
                
                # Stop follow-up processing animation
                animation_manager.stop_animation(followup_processing_name)
                if not animation_manager.should_suppress_lower_level():
                    safe_builtin_print(f"\r✅ Follow-up function execution completed" + " " * 20)
                    safe_builtin_print()  # Add a new line after completion
                
                # Recursively handle any additional function calls
                await self._stream_response_with_function_handling(follow_up_context)
                
            else:
                # Show Answer section for follow-up response and protect the content
                if current_response:
                    # Start Answer section for follow-up response
                    coordinated_console.print("\n[bold bright_blue]💬 Answer:[/bold bright_blue]")
                    
                    # Display the complete follow-up answer content in a protected way
                    console.print(current_response, style="bright_white")
                    
                    # Ensure answer section is properly terminated
                    console.print()  # Final newline to complete the answer section
                    sys.stdout.flush()  # Ensure everything is rendered
                    time.sleep(0.1)  # Give terminal time to render completely
                
                # ONLY stop follow-up thinking animation AFTER content is rendered
                if self.config.model == "gemini-2.5-pro" and 'followup_thinking_name' in locals() and followup_streaming_complete:
                    animation_manager.stop_animation(followup_thinking_name)
                    time.sleep(0.1)  # Wait before clearing animation line
                    if not animation_manager.should_suppress_lower_level():
                        safe_builtin_print(f"\r" + " " * 120 + "\r", end="", flush=True)  # Clear animation line
                    
                # No more function calls, add final response to conversation history
                self.conversation_history.append({"role": "assistant", "content": current_response})
                console.print(f"\n[dim green]✅ Response completed[/dim green]")
                
        except Exception as e:
            err_str = str(e)
            # Check if this is a rate limit error that should be retried
            if any(tok in err_str for tok in ["RESOURCE_EXHAUSTED", "UNAVAILABLE", "rate limit", "quota"]):
                coordinated_console.print(f"\n[bold red]❌ API quota/rate limit error during follow-up streaming: {str(e)}[/bold red]")
                coordinated_console.print(f"[dim red]The system has already attempted retries. Please wait before sending another request.[/dim red]")
            else:
                coordinated_console.print(f"\n[bold red]❌ Error during follow-up streaming: {str(e)}[/bold red]")

    def _build_context(self) -> List[types.Content]:
        """Build conversation context from history."""
        context = []
        for entry in self.conversation_history[-10:]:  # Keep last 10 exchanges
            if entry["role"] == "user":
                context.append(types.Content(
                    role="user",
                    parts=[types.Part(text=entry["content"])]
                ))
            elif entry["role"] == "assistant":
                # Handle different content types
                if isinstance(entry["content"], str):
                    context.append(types.Content(
                        role="model",
                        parts=[types.Part(text=entry["content"])]
                    ))
                elif isinstance(entry["content"], list):
                    # Content is already a list of parts
                    context.append(types.Content(
                        role="model",
                        parts=entry["content"]
                    ))
        return context
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()

    async def _reload_research_materials(self):
        """Reload research materials from disk after additional research."""
        try:
            all_file_contents = []
            output_path = Path("research_results")
            
            # Load files from main research directories
            for query_dir in output_path.iterdir():
                if query_dir.is_dir() and query_dir.name.startswith("query_"):
                    for file_path in query_dir.glob("*.md"):
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read()
                            
                            # Get original URL from global mapping store
                            original_url = global_url_mapping_store.get(file_path.name, f"Unknown URL for {file_path.name}")
                            formatted_content = f"=== SOURCE: {original_url} ===\n{content}\n\n"
                            all_file_contents.append(formatted_content)
                        except Exception:
                            continue
            
            # Load additional research files
            additional_dir = output_path / "additional_research"
            if additional_dir.exists():
                for file_path in additional_dir.rglob("*.md"):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        
                        # Get original URL from global mapping store
                        original_url = global_url_mapping_store.get(file_path.name, f"Additional Research File: {file_path.name}")
                        formatted_content = f"=== SOURCE: {original_url} ===\n{content}\n\n"
                        all_file_contents.append(formatted_content)
                    except Exception:
                        continue
            
            # Update stored research materials
            self.last_research_materials = "\n".join(all_file_contents)
            coordinated_console.print(f"[dim green]📚 Reloaded {len(all_file_contents)} research files[/dim green]")
            
        except Exception as e:
            coordinated_console.print(f"[dim red]Warning: Failed to reload research materials: {e}[/dim red]")

    async def _execute_function_call(self, func_name: str, func_args: dict):
        """Execute a function call and return the result."""
        try:
            # Log the function call attempt by main agent
            caller_model = "MainAgent(gemini-2.5-pro)" if self.config.model == "gemini-2.5-pro" else f"Agent({self.config.model})"
            function_logger.log_function_call(
                caller=caller_model,
                function_name=func_name,
                **func_args
            )
            # Display detailed function information before execution
            function_descriptions = {
                "research": "🔍 RESEARCH FUNCTION - Starting multi-agent research pipeline",
                "sub_research": "🔬 SUB-RESEARCH FUNCTION - Launching additional targeted research agents", 
                "temp_report": "📝 TEMP REPORT FUNCTION - Generating intermediate analysis report",
                "write_final_comprehensive_report": "📊 FINAL REPORT FUNCTION - Creating comprehensive final analysis",
                "write_final_subreport": "📋 SUBREPORT FUNCTION - Generating specialized sub-analysis"
            }
            
            description = function_descriptions.get(func_name, f"⚙️ FUNCTION - {func_name}")
            coordinated_console.print(f"\n[bold bright_cyan]╭─ {description} ─╮[/bold bright_cyan]")
            
            if func_name == "research":
                # Show research configuration details
                queries_config = func_args.get("queries_config", [])
                num_queries = len(queries_config)
                subagent_queries = [q for q in queries_config if q.get("use_subagents", False)]
                num_subagents = len(subagent_queries)
                
                coordinated_console.print(f"[bright_cyan]│ 🔍 Research queries: {num_queries}[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 🤖 Subagents to deploy: {num_subagents}[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 📂 Output directory: {func_args.get('output_dir', 'research_results')}[/bright_cyan]")
                
                for i, query_config in enumerate(queries_config, 1):
                    query = query_config.get("query", "Unknown query")
                    max_urls = query_config.get("max_source_count", query_config.get("max_urls", 5))
                    use_subagents = query_config.get("use_subagents", False)
                    truncated_query = query[:50] + ('...' if len(query) > 50 else '')
                    agent_type = "🤖 Subagent" if use_subagents else "🔍 Simple query"
                    coordinated_console.print(f"[bright_cyan]│ {agent_type} {i}: '{truncated_query}'[/bright_cyan]")
                    coordinated_console.print(f"[bright_cyan]│    └─ Max URLs: {max_urls}[/bright_cyan]")
                
                coordinated_console.print(f"[bold bright_cyan]╰─ Starting research execution... ─╯[/bold bright_cyan]")
                
                result = await research_function(**func_args)
                # Store research materials for subsequent function calls
                if result and "formatted_research_materials" in result:
                    self.last_research_materials = result["formatted_research_materials"]
                    coordinated_console.print(f"[dim green]📚 Stored {len(result.get('file_contents', []))} research files for analysis[/dim green]")
                
                # If report was auto-generated, return it directly
                if result and result.get("final_report_generated") and result.get("final_report_content"):
                    coordinated_console.print(f"[bold green]✨ Research completed with auto-generated report[/bold green]")
                    # Add the final report content to the result for immediate display
                    result["display_report"] = result["final_report_content"]
                
            elif func_name == "sub_research":
                # Show sub-research details
                research_queries = func_args.get("research_queries", [])
                max_urls_per_query = func_args.get("max_urls_per_query", 5)
                coordinated_console.print(f"[bright_cyan]│ 🔍 Sub-queries: {len(research_queries)}[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 🌐 URLs per query: {max_urls_per_query}[/bright_cyan]")
                
                for i, query in enumerate(research_queries, 1):
                    truncated_query = query[:60] + ('...' if len(query) > 60 else '')
                    coordinated_console.print(f"[bright_cyan]│ {i}. '{truncated_query}'[/bright_cyan]")
                
                reasoning = func_args.get('reasoning', 'Additional research needed')
                truncated_reasoning = reasoning[:80] + ('...' if len(reasoning) > 80 else '')
                coordinated_console.print(f"[bright_cyan]│ 🎯 Reasoning: {truncated_reasoning}[/bright_cyan]")
                coordinated_console.print(f"[bold bright_cyan]╰─ Launching sub-research agents... ─╯[/bold bright_cyan]")
                
                result = await sub_research_function_new(**func_args)
                # If new research was conducted, reload materials
                if result and result.get("success") and result.get("new_files"):
                    coordinated_console.print(f"[dim yellow]🔄 Reloading research materials after additional research[/dim yellow]")
                    # Reload materials from disk to include new files
                    await self._reload_research_materials()
                    
            elif func_name == "temp_report":
                # Show temp report details
                iteration = func_args.get("iteration_number", 1)
                sources_count = len(func_args.get("sources_processed", []))
                coordinated_console.print(f"[bright_cyan]│ 📝 Report iteration: {iteration}[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 📁 Sources processed: {sources_count}[/bright_cyan]")
                coordinated_console.print(f"[bold bright_cyan]╰─ Generating intermediate report... ─╯[/bold bright_cyan]")
                
                # Inject research materials into function arguments
                if "research_materials" not in func_args or not func_args["research_materials"]:
                    func_args["research_materials"] = self.last_research_materials
                    coordinated_console.print(f"[dim cyan]📖 Injecting research materials into temp report generation[/dim cyan]")
                
                # Add agent context for proper file management
                agent_id = self._get_agent_id_from_config()
                if agent_id and agent_id in status_tracker.agents:
                    agent_state = status_tracker.agents[agent_id]
                    func_args["agent_id"] = agent_id
                    func_args["iteration_number"] = agent_state.iteration
                    func_args["query_dir"] = agent_state.query_directory
                    
                result = await temp_report_function_new(**func_args)
                
                # Update agent state with temp report info
                if agent_id and result.get("success") and agent_id in status_tracker.agents:
                    agent_state = status_tracker.agents[agent_id]
                    agent_state.add_temp_report(
                        result["temp_report_path"],
                        result["iteration_number"],
                        func_args.get("compressed_report_content", "")[:100]
                    )
                
            elif func_name == "write_final_comprehensive_report":
                # Show final report details
                report_length = len(func_args.get("report_content", ""))
                summary_length = len(func_args.get("executive_summary", ""))
                coordinated_console.print(f"[bright_cyan]│ 📄 Report content: {report_length} characters[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 📋 Executive summary: {summary_length} characters[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 📂 Output: {func_args.get('output_dir', 'research_results')}[/bright_cyan]")
                coordinated_console.print(f"[bold bright_cyan]╰─ Creating final comprehensive report... ─╯[/bold bright_cyan]")
                
                # Inject research materials into function arguments
                if "research_materials" not in func_args or not func_args["research_materials"]:
                    func_args["research_materials"] = self.last_research_materials
                    coordinated_console.print(f"[dim cyan]📖 Injecting research materials ({len(self.last_research_materials)} chars) into report generation[/dim cyan]")
                else:
                    coordinated_console.print(f"[dim cyan]📖 Using provided research materials ({len(func_args['research_materials'])} chars)[/dim cyan]")
                
                result = await write_final_comprehensive_report_function(**func_args)
                
            elif func_name == "write_final_subreport":
                # Show subreport details
                confidence = func_args.get("confidence_assessment", "Not specified")
                key_findings_count = len(func_args.get("key_findings", []))
                sources_used = func_args.get("sources_used", 0)
                coordinated_console.print(f"[bright_cyan]│ 🎯 Confidence level: {confidence}[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 🔍 Key findings: {key_findings_count}[/bright_cyan]")
                coordinated_console.print(f"[bright_cyan]│ 📊 Sources used: {sources_used}[/bright_cyan]")
                coordinated_console.print(f"[bold bright_cyan]╰─ Generating specialized subreport... ─╯[/bold bright_cyan]")
                
                # Inject research materials into function arguments
                if "research_materials" not in func_args or not func_args["research_materials"]:
                    func_args["research_materials"] = self.last_research_materials
                    coordinated_console.print(f"[dim cyan]📖 Injecting research materials into final subreport generation[/dim cyan]")
                result = await write_final_subreport_function_new(**func_args)
            else:
                coordinated_console.print(f"[bright_cyan]│ ⚙️ Unknown function: {func_name}[/bright_cyan]")
                coordinated_console.print(f"[bold bright_cyan]╰─ Attempting execution... ─╯[/bold bright_cyan]")
                result = {"error": f"Unknown function: {func_name}"}
            
            coordinated_console.print(f"[bold green]✅ {func_name} execution completed successfully[/bold green]")
            return {"result": result}
            
        except Exception as e:
            error_msg = f"Error executing {func_name}: {str(e)}"
            coordinated_console.print(f"[bold red]❌ {error_msg}[/bold red]")
            return {"error": error_msg}

    async def _handle_workflow_response(self, response, contents):
        """Handle response from workflow continuation to prevent infinite loops"""
        try:
            # Log for debugging AG02 workflow issues
            agent_id = self._get_agent_id_from_config()
            if agent_id and agent_id in agent_loggers:
                agent_loggers[agent_id].log_workflow_step("WORKFLOW_RESPONSE_DEBUG", f"Response type: {type(response)}, has candidates: {hasattr(response, 'candidates')}, has text: {hasattr(response, 'text')}")
            
            # Check if response contains function calls
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'function_call') and part.function_call:
                            # Recursively handle the function call
                            function_call = part.function_call
                            if function_call.name in ["temp_report", "sub_research", "write_final_subreport"]:
                                # Execute the function call with current contents as context
                                return await self._handle_function_call_with_context(function_call, contents)
                            
            # If no function call, return the text response
            with suppress_gemini_warnings():
                if hasattr(response, 'text') and response.text:
                    return response.text
                
            return "Agent completed workflow step."
            
        except Exception as e:
            coordinated_console.print(f"[red]❌ Error handling workflow response: {e}[/red]")
            return f"Error in workflow: {str(e)}"

    async def _handle_function_call_with_context(self, function_call, previous_contents):
        """Handle function call while preserving conversation context"""
        # This is similar to the existing function call handling but preserves context
        if function_call.name == "temp_report":
            # This shouldn't happen in workflow continuation, but handle it
            return "⚠️ temp_report() already called. Please use sub_research() or write_final_subreport()."
            
        elif function_call.name == "sub_research":
            try:
                args = {}
                if hasattr(function_call, 'args'):
                    args = function_call.args
                result = await sub_research_function_new(**args)
                # For sub_research, return success message (no further continuation needed)
                return "✅ Additional research completed. Agent workflow continuing..."
                
            except Exception as e:
                return f"Error in sub_research: {str(e)}"
                
        elif function_call.name == "write_final_subreport":
            try:
                args = {}
                if hasattr(function_call, 'args'):
                    args = function_call.args
                result = await write_final_subreport_function_new(**args)
                
                # Log completion
                agent_id = self._get_agent_id_from_config()
                if agent_id and agent_id in agent_loggers:
                    agent_loggers[agent_id].log_completion(f"Final subreport completed successfully")
                
                return f"✅ Agent {agent_id} workflow completed successfully. Final subreport has been generated."
                
            except Exception as e:
                return f"Error in write_final_subreport: {str(e)}"
                
        return "Unknown function call in workflow."


def create_agent(api_key: str, model: str = "gemini-2.5-flash", system_instruction: str = None) -> GeminiAgent:
    """Create a simple Gemini agent."""
    config = AgentConfig(
        api_key=api_key,
        model=model,
        system_instruction=system_instruction
    )
    return GeminiAgent(config)


async def write_final_comprehensive_report_function(
    report_content: str,
    executive_summary: str,
    research_materials: str = "",
    methodology_notes: str = "",
    output_dir: str = "research_results"
) -> Dict[str, Any]:
    """
    Create final comprehensive report from AI-generated content with access to research materials.
    The AI model provides the complete report content and has access to all research files.
    """
    # DEBUG: Log function start
    global main_agent_logger
    if main_agent_logger:
        main_agent_logger.log_workflow_step("REPORT_FUNCTION_START", f"write_final_comprehensive_report_function called with {len(report_content)} chars report")
    
    # Log function call
    function_logger.log_function_call(
        caller="MainAgent(gemini-2.5-pro)",
        function_name="write_final_comprehensive_report",
        report_content=report_content,
        executive_summary=executive_summary,
        research_materials=research_materials,
        methodology_notes=methodology_notes,
        output_dir=output_dir
    )
    
    # DEBUG: Log after function logger
    if main_agent_logger:
        main_agent_logger.log_workflow_step("FUNCTION_LOGGER_COMPLETE", "Function logger completed")
    
    try:
        # DEBUG: Log directory creation
        if main_agent_logger:
            main_agent_logger.log_workflow_step("CREATING_OUTPUT_DIR", f"Creating output directory: {output_dir}")
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # DEBUG: Log research note creation
        if main_agent_logger:
            main_agent_logger.log_workflow_step("CREATING_RESEARCH_NOTE", f"Creating research note from {len(research_materials)} chars")
        
        # Create the final comprehensive report document
        research_note = f"\n\n## Research Materials Summary\n\nThis report was generated based on analysis of {len(research_materials.split('=== FILE:')) - 1 if research_materials else 0} research files containing comprehensive web-sourced information.\n" if research_materials else ""
        
        final_document = f"""# Final Comprehensive Research Report

**Generated:** {datetime.now().isoformat()}

## Executive Summary

{executive_summary}

---

{report_content}

---

## Research Methodology

{methodology_notes if methodology_notes else "Advanced multi-agent research system with iterative analysis and source synthesis."}
{research_note}
---

*Report generated by Advanced Multi-Agent Research System*
"""
        
        # DEBUG: Log file writing
        if main_agent_logger:
            main_agent_logger.log_workflow_step("WRITING_FILE", f"Writing final document with {len(final_document)} chars")
        
        # Save the final report
        final_report_path = output_path / "final_comprehensive_report.md"
        with open(final_report_path, 'w', encoding='utf-8') as f:
            f.write(final_document)
        
        # DEBUG: Log file written
        if main_agent_logger:
            main_agent_logger.log_workflow_step("FILE_WRITTEN", f"File written to: {final_report_path}")
        
        coordinated_console.print(f"[green]📄 Final comprehensive report saved: {final_report_path}[/green]")
        
        # DEBUG: Log return preparation
        if main_agent_logger:
            main_agent_logger.log_workflow_step("PREPARING_RETURN", "Preparing return value")
        
        result = {
            "success": True,
            "final_report_path": str(final_report_path),
            "report_length": len(report_content),
            "summary_length": len(executive_summary),
            "message": f"Final comprehensive report created successfully at {final_report_path.name}"
        }
        
        # DEBUG: Log returning
        if main_agent_logger:
            main_agent_logger.log_workflow_step("RETURNING_RESULT", f"Returning result: {result['success']}")
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to create final comprehensive report: {e}"
        }


async def temp_report_function_new(
    compressed_report_content: str,
    agent_id: str = None,
    iteration_number: int = 1,
    query_dir: str = None
) -> Dict[str, Any]:
    """Create temporary report from AI-generated content and DELETE all source files as per flow.md"""
    
    # Log function call
    function_logger.log_function_call(
        caller="SubAgent(gemini-2.5-flash)",
        function_name="temp_report",
        compressed_report_content=compressed_report_content[:200] + "..." if len(compressed_report_content) > 200 else compressed_report_content,
        agent_id=agent_id,
        iteration_number=iteration_number
    )
    
    try:
        output_dir = Path("research_results")
        output_dir.mkdir(exist_ok=True)
        
        # Generate filename with agent ID and iteration
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        agent_part = f"{agent_id}_" if agent_id else ""
        temp_report_filename = f"temp_report_{agent_part}iter{iteration_number}_{timestamp}.md"
        
        # Create comprehensive temp report with iteration info
        temp_report_content = f"""# Temporary Research Analysis - Iteration {iteration_number}

**Agent:** {agent_id or 'Unknown'}
**Generated:** {datetime.now().isoformat()}
**Iteration:** {iteration_number}

## Compressed Analysis

{compressed_report_content}

---

*Temporary analysis report created by research agent - Iteration {iteration_number}*
*Original source files have been deleted and converted to this compressed report*
"""
        
        # Save temp report
        temp_report_path = output_dir / temp_report_filename
        with open(temp_report_path, 'w', encoding='utf-8') as f:
            f.write(temp_report_content)
        
        # CRITICAL: Delete all files in the query directory as per flow.md (including previous temp reports)
        files_deleted = []
        if query_dir:
            query_path = Path(query_dir)
            if query_path.exists():
                # Delete ALL .md files EXCEPT final subreports (per flow.md: "delete the previous temp report and files")
                for file_path in query_path.glob("*.md"):
                    if not file_path.name.startswith("final_subreport_"):
                        try:
                            file_path.unlink()
                            files_deleted.append(file_path.name)
                        except Exception as e:
                            coordinated_console.print(f"[yellow]Warning: Could not delete {file_path.name}: {e}[/yellow]")
                
                coordinated_console.print(f"[yellow]🗑️ Deleted {len(files_deleted)} files including previous temp reports (as per flow.md)[/yellow]")
        
        # Update global status tracker
        if agent_id and agent_id in status_tracker.agents:
            status_tracker.update_agent(agent_id, 
                                      state=f"📝 Temp Report {iteration_number}",
                                      iteration=iteration_number,
                                      file_count=1)  # Only temp report remains
        
        return {
            "success": True,
            "temp_report_path": str(temp_report_path),
            "content_length": len(compressed_report_content),
            "iteration_number": iteration_number,
            "files_deleted": files_deleted,
            "agent_id": agent_id,
            "message": f"Temporary report iteration {iteration_number} created successfully, {len(files_deleted)} source files deleted"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "iteration_number": iteration_number,
            "agent_id": agent_id,
            "message": f"Failed to create temporary report: {e}"
        }


async def sub_research_function_new(
    research_queries: List[str],
    reasoning: str,
    max_urls_per_query: int = 5
) -> Dict[str, Any]:
    """
    Perform additional research based on AI-identified gaps.
    """
    # Log function call
    function_logger.log_function_call(
        caller="SubAgent(gemini-2.5-flash)",
        function_name="sub_research",
        research_queries=research_queries,
        reasoning=reasoning,
        max_urls_per_query=max_urls_per_query
    )
    try:
        if not research_queries:
            return {"success": False, "error": "No research queries provided"}
        
        # Ensure we do not exceed the global remaining URL allowance
        global_remaining = research_limits.get_remaining_urls()
        if global_remaining <= 0:
            return {"success": False, "error": "Global URL quota exhausted", "new_files": []}

        max_urls_per_query = min(max_urls_per_query, global_remaining)
        
        # Create subdirectory for additional research
        output_dir = Path("research_results") / "additional_research"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_new_files = []
        total_urls_found = 0
        
        coordinated_console.print(f"[cyan]Starting additional research: {reasoning}[/cyan]")
        
        for i, query in enumerate(research_queries):
            try:
                coordinated_console.print(f"[cyan]  Query {i+1}/{len(research_queries)}: {query[:60]}...[/cyan]")
                
                # Get URLs
                urls = query2url(query, num_results=max_urls_per_query)
                
                if urls:
                    # Filter out already processed URLs
                    new_urls = research_limits.filter_new_urls(urls)
                    
                    if new_urls:
                        # Create URL-to-filename mapping for new URLs
                        new_url_mapping = {}
                        for url in new_urls:
                            safe_name = (
                                url
                                .replace("https://", "")
                                .replace("http://", "")
                                .replace("/", "_")
                                .replace("?", "_")
                                .replace("&", "_")
                            )[:100]  # Same logic as url2file
                            filename = f"{safe_name}.md"
                            new_url_mapping[filename] = url
                        
                        # Process URLs to files
                        query_subdir = output_dir / f"query_{i+1}"
                        await url2file(new_urls, str(query_subdir))
                        
                        # Add to processed URLs
                        research_limits.add_processed_urls(new_urls)
                        
                        # Track new files
                        new_files = list(query_subdir.glob("*.md"))
                        all_new_files.extend([str(f) for f in new_files])
                        
                        # Update global URL mapping store for future reference
                        global_url_mapping_store.update(new_url_mapping)
                        
                        total_urls_found += len(new_urls)
                        coordinated_console.print(f"[green]    Found {len(new_urls)} new URLs[/green]")
                    else:
                        coordinated_console.print(f"[yellow]    All URLs already processed[/yellow]")
                        
            except Exception as e:
                coordinated_console.print(f"[red]    Error with query '{query}': {e}[/red]")
                continue
        
        return {
            "success": True,
            "new_files": all_new_files,
            "urls_found": total_urls_found,
            "queries_processed": len(research_queries),
            "reasoning": reasoning,
            "message": f"Additional research completed: {total_urls_found} new URLs, {len(all_new_files)} files"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Additional research failed: {e}"
        }


async def write_final_subreport_function_new(
    subreport_content: str,
    key_findings: List[str] = None
) -> Dict[str, Any]:
    """
    Create final subreport from AI-generated content. Saves ONLY the agent's analysis.
    """
    # Log function call
    function_logger.log_function_call(
        caller="SubAgent(gemini-2.5-flash)",
        function_name="write_final_subreport",
        subreport_content=subreport_content[:200] + "..." if len(subreport_content) > 200 else subreport_content,
        key_findings=key_findings
    )
    
    try:
        output_dir = Path("research_results")
        output_dir.mkdir(exist_ok=True)
        
        # Generate simple unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create simple final subreport - ONLY the agent's analysis  
        final_subreport_content = f"""# Final Research Subreport

**Generated:** {datetime.now().isoformat()}

## Key Findings

{chr(10).join([f"- {finding}" for finding in (key_findings or ["No key findings specified"])])}

## Comprehensive Analysis

{subreport_content}

---

*Final subreport completed by research agent*
"""
        
        # Save final subreport
        final_subreport_path = output_dir / f"final_subreport_{timestamp}.md"
        with open(final_subreport_path, 'w', encoding='utf-8') as f:
            f.write(final_subreport_content)
        
        # Update global status tracker
        status_tracker.add_research_update(f"Subreport created: {final_subreport_path.name}")
        
        return {
            "success": True,
            "final_subreport_path": str(final_subreport_path),
            "key_findings_count": len(key_findings or []),
            "content_length": len(subreport_content),
            "message": f"Final subreport completed successfully"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Failed to create final subreport: {e}"
        }

# Function call logging system
class FunctionCallLogger:
    """Centralized logging for all function calls with parameter truncation."""
    
    def __init__(self, log_file: str = "research_results/function_calls.log"):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        # Initialize log file with header
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== FUNCTION CALL LOG - Session Started: {datetime.now().isoformat()} ===\n\n")
    
    def log_function_call(self, caller: str, function_name: str, **kwargs):
        """Log a function call with truncated parameters."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        # Truncate parameters to 200 characters each
        truncated_params = {}
        for key, value in kwargs.items():
            if value is None:
                truncated_params[key] = "None"
            elif isinstance(value, (str, int, float, bool)):
                str_value = str(value)
                if len(str_value) > 200:
                    truncated_params[key] = str_value[:200] + "..."
                else:
                    truncated_params[key] = str_value
            elif isinstance(value, (list, tuple)):
                list_str = str(value)
                if len(list_str) > 200:
                    truncated_params[key] = list_str[:200] + "..."
                else:
                    truncated_params[key] = list_str
            elif isinstance(value, dict):
                dict_str = str(value)
                if len(dict_str) > 200:
                    truncated_params[key] = dict_str[:200] + "..."
                else:
                    truncated_params[key] = dict_str
            else:
                str_repr = repr(value)
                if len(str_repr) > 200:
                    truncated_params[key] = str_repr[:200] + "..."
                else:
                    truncated_params[key] = str_repr
        
        # Format log entry
        log_entry = f"[{timestamp}] {caller} -> {function_name}(\n"
        for key, value in truncated_params.items():
            log_entry += f"    {key}={value}\n"
        log_entry += ")\n\n"
        
        # Write to log file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            # Fallback: print to console if logging fails
            safe_builtin_print(f"[WARNING] Function call logging failed: {e}")

# Individual Agent Logger System
class AgentLogger:
    """Individual logging system for each agent with detailed workflow tracking."""
    
    def __init__(self, agent_id: str, log_dir: str = "research_results/agent_logs"):
        self.agent_id = agent_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{agent_id}.log"
        
        # Initialize agent log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== AGENT {agent_id} LOG - Session Started: {datetime.now().isoformat()} ===\n\n")
    
    def log_workflow_step(self, step: str, details: str = "", status: str = "INFO"):
        """Log a workflow step with timestamp."""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
        log_entry = f"[{timestamp}] [{status}] {step}"
        if details:
            log_entry += f" - {details}"
        log_entry += "\n"
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
        except Exception as e:
            safe_builtin_print(f"[WARNING] Agent {self.agent_id} logging failed: {e}")
    
    def log_function_call(self, function_name: str, params: dict = None, result: str = ""):
        """Log function calls with parameters and results."""
        self.log_workflow_step(f"FUNCTION_CALL: {function_name}", f"Params: {params}, Result: {result[:100]}...")
    
    def log_decision(self, decision: str, reasoning: str = ""):
        """Log agent decisions."""
        self.log_workflow_step(f"DECISION: {decision}", reasoning)
    
    def log_analysis(self, analysis: str):
        """Log analysis results."""
        self.log_workflow_step("ANALYSIS", analysis[:500] + "..." if len(analysis) > 500 else analysis)
    
    def log_error(self, error: str):
        """Log errors."""
        self.log_workflow_step("ERROR", error, "ERROR")
    
    def log_completion(self, final_status: str):
        """Log agent completion."""
        self.log_workflow_step("COMPLETION", final_status, "SUCCESS")

# Gemini Response Logger - Silent logging to interactions.log
class GeminiResponseLogger:
    """Silent logging for all Gemini API responses to interactions.log file."""
    
    def __init__(self, log_file: str = "interactions.log"):
        self.log_file = Path(log_file)
        # Initialize log file
        with open(self.log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== GEMINI INTERACTIONS LOG - {datetime.now().isoformat()} ===\n\n")
    
    def log_response(self, response_obj, context: str = ""):
        """Log Gemini response object with all parts - no console output."""
        timestamp = datetime.now().isoformat()
        
        try:
            log_entry = f"\n--- RESPONSE {timestamp} ---\n"
            if context:
                log_entry += f"CONTEXT: {context}\n"
            
            # Log response type and basic info
            log_entry += f"TYPE: {type(response_obj)}\n"
            
            # Log all response attributes
            if hasattr(response_obj, '__dict__'):
                for attr, value in response_obj.__dict__.items():
                    log_entry += f"{attr}: {repr(value)}\n"
            
            # Log candidates if present
            if hasattr(response_obj, 'candidates') and response_obj.candidates:
                for i, candidate in enumerate(response_obj.candidates):
                    log_entry += f"\nCANDIDATE_{i}: {repr(candidate)}\n"
                    if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                        for j, part in enumerate(candidate.content.parts or []):
                            log_entry += f"  PART_{j}: {repr(part)}\n"
            
            # Log text if present
            if hasattr(response_obj, 'text') and response_obj.text:
                log_entry += f"TEXT: {response_obj.text}\n"
            
            log_entry += "--- END RESPONSE ---\n\n"
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_entry)
                
        except Exception:
            # Silent failure - no console output
            pass

# Prompt and History Logger - Silent logging to separate files per API call
class PromptHistoryLogger:
    """Silent logging for AI prompts and history to separate files (call_1.log, call_2.log, etc.)."""
    
    def __init__(self, log_dir: str = "prompt_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.call_counter = 0
        
    def log_api_call(self, contents: list, context: str = "", config=None, model: str = ""):
        """Log full prompt and history for a single API call - no console output."""
        self.call_counter += 1
        timestamp = datetime.now().isoformat()
        
        # Force create the log file regardless of errors
        log_file = self.log_dir / f"call_{self.call_counter}.log"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"=== AI API CALL LOG - CALL {self.call_counter} ===\n")
                f.write(f"TIMESTAMP: {timestamp}\n")
                f.write(f"MODEL: {model}\n")
                if context:
                    f.write(f"CONTEXT: {context}\n")
                f.write(f"{'=' * 60}\n\n")
                
                # Section 0: Full API Configuration
                f.write("SECTION 0: FULL API CONFIGURATION\n")
                f.write("-" * 40 + "\n")
                if config:
                    f.write(f"Config Type: {type(config)}\n")
                    f.write(f"Config Raw: {repr(config)}\n")
                    
                    # Extract key configuration details
                    if hasattr(config, 'system_instruction') and config.system_instruction:
                        f.write(f"\nSYSTEM INSTRUCTION:\n{config.system_instruction}\n")
                    
                    if hasattr(config, 'tools') and config.tools:
                        f.write(f"\nTOOLS ({len(config.tools)}):\n")
                        for i, tool in enumerate(config.tools):
                            f.write(f"  Tool {i+1}: {repr(tool)}\n")
                    
                    if hasattr(config, 'safety_settings') and config.safety_settings:
                        f.write(f"\nSAFETY SETTINGS:\n{repr(config.safety_settings)}\n")
                    
                    if hasattr(config, 'thinking_config') and config.thinking_config:
                        f.write(f"\nTHINKING CONFIG:\n{repr(config.thinking_config)}\n")
                    
                    if hasattr(config, '__dict__'):
                        f.write(f"\nALL CONFIG ATTRIBUTES:\n")
                        for attr, value in config.__dict__.items():
                            f.write(f"  {attr}: {repr(value)}\n")
                else:
                    f.write("No config provided\n")
                f.write("\n")
                
                # Section 1: Full Conversation History
                f.write("SECTION 1: FULL CONVERSATION HISTORY\n")
                f.write("-" * 40 + "\n")
                if contents:
                    for i, content in enumerate(contents):
                        f.write(f"\n--- MESSAGE {i+1} ---\n")
                        
                        # Handle different content types
                        if isinstance(content, str):
                            f.write(f"Type: String (Direct prompt)\n")
                            f.write(f"Content:\n{content}\n")
                        else:
                            f.write(f"Type: {type(content)}\n")
                            f.write(f"Raw Content: {repr(content)}\n")
                            
                            # Try to extract readable content
                            if hasattr(content, 'text'):
                                f.write(f"Text Content:\n{content.text}\n")
                            elif hasattr(content, 'parts'):
                                f.write("Parts:\n")
                                for j, part in enumerate(content.parts):
                                    f.write(f"  Part {j+1}: {repr(part)}\n")
                            elif hasattr(content, '__dict__'):
                                f.write("Attributes:\n")
                                for attr, value in content.__dict__.items():
                                    f.write(f"  {attr}: {repr(value)}\n")
                        
                        f.write(f"--- END MESSAGE {i+1} ---\n")
                else:
                    f.write("No contents found\n")
                
                # Section 2: Current Prompt (Last Message)
                f.write(f"\n\nSECTION 2: CURRENT PROMPT (LAST MESSAGE)\n")
                f.write("-" * 40 + "\n")
                if contents:
                    current_prompt = contents[-1]
                    if isinstance(current_prompt, str):
                        f.write(f"Prompt Length: {len(current_prompt)} characters\n")
                        f.write(f"Prompt Word Count: {len(current_prompt.split())} words\n")
                        f.write("Prompt Content:\n")
                        f.write(current_prompt)
                    else:
                        f.write(f"Prompt Type: {type(current_prompt)}\n")
                        f.write(f"Prompt Content: {repr(current_prompt)}\n")
                else:
                    f.write("No current prompt found\n")
                
                # Section 3: Metadata
                f.write(f"\n\nSECTION 3: METADATA\n")
                f.write("-" * 40 + "\n")
                f.write(f"API Call Number: {self.call_counter}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Context: {context}\n")
                f.write(f"Total Messages in History: {len(contents) if contents else 0}\n")
                total_chars = sum(len(str(c)) for c in contents) if contents else 0
                f.write(f"Total Characters: {total_chars}\n")
                
                f.write(f"\n{'=' * 60}\n")
                f.write("=== END LOG ===\n")
                
        except Exception as e:
            # Force create error log file
            try:
                with open(log_file, 'w', encoding='utf-8') as f:
                    f.write(f"=== ERROR LOG - CALL {self.call_counter} ===\n")
                    f.write(f"TIMESTAMP: {timestamp}\n")
                    f.write(f"ERROR: {str(e)}\n")
                    f.write(f"ERROR TYPE: {type(e)}\n")
                    f.write(f"CONTEXT: {context}\n")
                    f.write(f"CONTENTS TYPE: {type(contents)}\n")
                    f.write(f"CONTENTS: {repr(contents)}\n")
                    f.write("=== END ERROR LOG ===\n")
            except:
                # Last resort - create minimal error file
                try:
                    with open(self.log_dir / f"error_{self.call_counter}.txt", 'w') as f:
                        f.write(f"Error occurred at {timestamp}\n")
                except:
                    pass

# Global function call logger
function_logger = FunctionCallLogger()

# Global Gemini response logger
gemini_response_logger = GeminiResponseLogger()

# Global prompt and history logger
prompt_history_logger = PromptHistoryLogger()

# Force create a startup log to test logging is working
try:
    startup_file = Path("prompt_logs") / "startup_test.txt"
    startup_file.parent.mkdir(exist_ok=True)
    with open(startup_file, 'w') as f:
        f.write(f"Agents module loaded at {datetime.now().isoformat()}\n")
        f.write("Logging system initialized\n")
except Exception as e:
    try:
        with open("logging_error.txt", 'w') as f:
            f.write(f"Failed to create startup log: {e}\n")
    except:
        pass

# Global agent loggers store
agent_loggers = {}

# Main agent logger (for MainAgent debugging)
main_agent_logger = None

# Helper functions for enhanced subagent workflow

async def load_agent_research_materials(
    agent_id: str,
    initial_files_dir: Path,
    url_mapping: Dict[str, str],
    token_limit: int
) -> str:
    """Load current research materials for agent with proper token management."""
    content_chunks = []
    total_chars = 0
    max_chars = min(token_limit * 3, 400000)  # Approximate token-to-char conversion, max 400K
    
    # Get agent state to check for temp reports
    agent_state = status_tracker.agents.get(agent_id)
    
    # Load temp reports first (highest priority)
    if agent_state and agent_state.temp_reports:
        for temp_report in agent_state.temp_reports:
            try:
                with open(temp_report["path"], 'r', encoding='utf-8') as f:
                    temp_content = f.read()
                    
                formatted_chunk = f"\n=== TEMP REPORT (Iteration {temp_report['iteration']}) ===\n{temp_content}\n"
                
                if total_chars + len(formatted_chunk) < max_chars:
                    content_chunks.append(formatted_chunk)
                    total_chars += len(formatted_chunk)
                else:
                    break
            except Exception as e:
                coordinated_console.print(f"[yellow]Warning: Could not load temp report {temp_report['path']}: {e}[/yellow]")
    
    # Load initial research files (if no temp reports exist or space remains)
    if initial_files_dir.exists() and total_chars < max_chars:
        initial_files = list(initial_files_dir.glob("*.md"))
        
        # Filter out temp reports and final subreports
        source_files = [f for f in initial_files if not f.name.startswith(("temp_report_", "final_subreport_"))]
        
        for file_path in source_files:
            if total_chars >= max_chars:
                break
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    filename = file_path.name

                    # Get the original URL for this file
                    original_url = url_mapping.get(filename, f"Unknown URL for {filename}")
                    
                    # Truncate individual files if they're too large
                    remaining_space = max_chars - total_chars
                    if len(content) > remaining_space // 2:  # Leave space for formatting
                        content = content[:remaining_space // 2] + f"\n\n[TRUNCATED - Original file was {len(content)} characters]"

                    formatted_chunk = f"\n=== SOURCE: {original_url} ===\n{content}\n"
                    
                    if total_chars + len(formatted_chunk) < max_chars:
                        content_chunks.append(formatted_chunk)
                        total_chars += len(formatted_chunk)
                    else:
                        # Add truncation note and break
                        remaining_files = len(source_files) - len([c for c in content_chunks if "=== SOURCE:" in c])
                        if remaining_files > 0:
                            content_chunks.append(f"\n[NOTE: Additional {remaining_files} files truncated due to token limits]\n")
                        break
                        
            except Exception as e:
                coordinated_console.print(f"[yellow]Warning: Could not load {file_path.name}: {e}[/yellow]")
                continue

    research_materials = "".join(content_chunks)
    
    # Update agent state
    if agent_state:
        source_file_count = len([c for c in content_chunks if "=== SOURCE:" in c])
        temp_report_count = len([c for c in content_chunks if "=== TEMP REPORT" in c])
        status_tracker.update_agent(agent_id, file_count=source_file_count + temp_report_count)
    
    return research_materials


def create_iteration_prompt(
    agent_id: str,
    query: str,
    global_task_context: str,
    current_iteration: int,
    research_materials: str,
    urls_remaining: int,
    token_limit: int,
    temp_reports: List[Dict[str, str]]
) -> str:
    """Create iteration-specific prompt following flow.md guidelines (informative, not instructional)."""
    
    # Iteration context
    iteration_context = f"This is iteration {current_iteration} for agent {agent_id}."
    
    if current_iteration == 1:
        iteration_context += " You are analyzing the initial research files."
    else:
        iteration_context += f" You have completed {current_iteration - 1} previous iterations."
        if temp_reports:
            latest_temp = temp_reports[-1]
            iteration_context += f" Your most recent temp report was from iteration {latest_temp['iteration']}."
    
    # URL budget context
    if urls_remaining <= 0:
        budget_context = f"⚠️ CRITICAL: Your URL budget is EXHAUSTED (0 URLs remaining). You cannot perform additional research and should immediately create your final subreport."
    elif urls_remaining <= 2:
        budget_context = f"⚠️ WARNING: You have only {urls_remaining} URLs remaining. Consider if current information is sufficient to create final subreport."
    else:
        budget_context = f"You have {urls_remaining} URLs remaining in your budget for additional research."
    
    # Token limit context
    token_context = f"Your final subreport token limit is {token_limit:,} tokens."
    
    # Decision context (following flow.md pattern)
    if urls_remaining <= 0:
        decision_context = """
⚠️ CRITICAL DECISION REQUIRED:
Your URL budget is EXHAUSTED. You MUST call write_final_subreport() immediately with whatever research materials you have.
Additional research is NOT possible. Your response will be analyzed for completion.
"""
    else:
        decision_context = """
You must analyze the research materials and make a decision:

- If the information is SUFFICIENT for your query → Call write_final_subreport()  
- If the information is INSUFFICIENT → Call temp_report() first, then sub_research() for more information

Your response will be analyzed for DECISION: SUFFICIENT or DECISION: INSUFFICIENT patterns.
"""
    
    # Create the complete prompt (informative, not instructional as per flow.md)
    prompt = f"""**CURRENT ITERATION CONTEXT:**
{iteration_context}

**GLOBAL TASK CONTEXT:**
{global_task_context}

**YOUR SPECIFIC QUERY:**
{query}

**RESOURCE STATUS:**
{budget_context}
{token_context}

**DECISION FRAMEWORK:**
{decision_context}

**RESEARCH MATERIALS:**
{research_materials}

**CURRENT DATE:** {datetime.now().strftime("%Y-%m-%d")}
"""
    
    return prompt


def parse_agent_decision(response: str, agent_id: str, iteration: int) -> str:
    """Parse agent decision from response looking for SUFFICIENT/INSUFFICIENT patterns as per flow.md."""
    
    response_lower = response.lower()
    
    # Check for completion indicators
    completion_patterns = [
        "final subreport has been generated",
        "write_final_subreport",
        "decision: sufficient",
        "sufficient information",
        "completed successfully",
        "✅",
        "workflow completed"
    ]
    
    if any(pattern in response_lower for pattern in completion_patterns):
        agent_loggers[agent_id].log_decision("SUFFICIENT", f"Iteration {iteration}: Sufficient information detected")
        return "FINAL_REPORT_COMPLETED"
    
    # Check for insufficient/continuing indicators
    continue_patterns = [
        "decision: insufficient",
        "insufficient information",
        "temp_report",
        "sub_research",
        "need more",
        "additional research",
        "continue",
        "not enough"
    ]
    
    if any(pattern in response_lower for pattern in continue_patterns):
        agent_loggers[agent_id].log_decision("INSUFFICIENT", f"Iteration {iteration}: Insufficient information, continuing")
        return "INSUFFICIENT_CONTINUING"
    
    # Check for error indicators
    error_patterns = [
        "error",
        "failed",
        "exception",
        "could not",
        "unable to"
    ]
    
    if any(pattern in response_lower for pattern in error_patterns):
        agent_loggers[agent_id].log_decision("ERROR", f"Iteration {iteration}: Error detected in response")
        return "ERROR"
    
    # Default: assume continuing (insufficient)
    agent_loggers[agent_id].log_decision("UNCLEAR", f"Iteration {iteration}: Unclear response, assuming insufficient")
    return "INSUFFICIENT_CONTINUING"


