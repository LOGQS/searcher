from datetime import datetime
from pathlib import Path
import os

# --- Rich UI imports ---------------------------------------------------------
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from prompt_toolkit import PromptSession
from prompt_toolkit.styles import Style as PromptStyle

# --- Agent imports -------------------------------------------------------------
from agents import create_agent
from prompts import main_agent_prompt


gemini_api_key = os.getenv("GEMINI_API_KEY")

# Initialize Rich console and prompt session
console = Console()
prompt_session = PromptSession(
    style=PromptStyle.from_dict({
        'prompt': '#9333ea bold',  # Purple prompt for research theme
        'completion-menu.completion': 'bg:#7c3aed fg:#ffffff',
        'completion-menu.completion.current': 'bg:#a855f7 fg:#ffffff bold',
    })
)

# Global session state
session_stats = {
    'total_chat_messages': 0,
    'session_start_time': datetime.now(),
}

# Global agent instance
chat_agent = None

def get_chat_agent():
    """Get or create the chat agent instance."""
    global chat_agent
    if chat_agent is None and gemini_api_key:
        chat_agent = create_agent(
            api_key=gemini_api_key,
            model="gemini-2.5-pro",
            system_instruction=main_agent_prompt
        )
    return chat_agent

async def handle_chat_message(message: str):
    """Handle a chat message and stream the response."""
    agent = get_chat_agent()
    if not agent:
        console.print("[bold red]❌ No API key found. Set GEMINI_API_KEY environment variable.[/bold red]")
        return
    
    console.print(f"\n[bold bright_blue]🤖 Assistant:[/bold bright_blue]")
    
    try:
        # chat_stream handles all output internally, just await it
        await agent.chat_stream(message)
        
        session_stats['total_chat_messages'] += 1
        
    except Exception as e:
        console.print(f"\n[bold red]❌ Chat error: {e}[/bold red]")
    
    # Ensure there's a newline before the next prompt appears
    console.print()

def show_help():
    """Show help information about commands and usage."""
    help_text = """[bold bright_magenta]🔬 Researcher Commands:[/bold bright_magenta]

[bold bright_cyan]Chat Commands:[/bold bright_cyan]
  [yellow]Just type your message[/yellow]               # Chat with the AI assistant
  [yellow]chat "your message"[/yellow]                  # Alternative chat format

[bold bright_cyan]Session Commands:[/bold bright_cyan]
  [yellow]/stats[/yellow] or [yellow]/status[/yellow]               # Show session statistics
  [yellow]/help[/yellow] or [yellow]/h[/yellow]                    # Show this help
  [yellow]/clear[/yellow]                               # Clear chat history
  [yellow]/output[/yellow] or [yellow]/results[/yellow]            # Show results directory
  [yellow]exit[/yellow] or [yellow]quit[/yellow]                   # Exit the application

[bold bright_cyan]Features:[/bold bright_cyan]
  • Conversational AI with streaming responses
  • Persistent conversation history
  • Rich terminal interface with live updates
  • Session statistics tracking

[bold bright_cyan]Usage Examples:[/bold bright_cyan]
  [yellow]Hello, how can you help me today?[/yellow]
  [yellow]What do you know about quantum computing?[/yellow]
  [yellow]chat "Tell me about AI research"[/yellow]"""
    
    console.print(Panel(
        help_text,
        border_style="bright_magenta",
        padding=(1, 2),
        title="[bold bright_cyan]📚 Help & Usage[/bold bright_cyan]",
        title_align="left"
    ))

def show_stats():
    """Show session statistics."""
    runtime = datetime.now() - session_stats['session_start_time']
    runtime_str = str(runtime).split('.')[0]  # Remove microseconds
    
    stats_table = Table(
        title="📊 Research Session Statistics",
        show_header=True,
        header_style="bold bright_magenta",
        border_style="bright_magenta"
    )
    stats_table.add_column("Metric", style="bold white", width=25)
    stats_table.add_column("Value", style="bright_cyan", width=15)
    stats_table.add_column("Details", style="dim white", width=30)
    
    stats_table.add_row("Session Runtime", runtime_str, "Time since start")
    stats_table.add_row("Chat Messages", str(session_stats['total_chat_messages']), "Messages exchanged")
    
    console.print()
    console.print(stats_table)
    console.print()

def try_handle_stats_command(user_input: str) -> bool:
    """Handle /stats or /status commands."""
    if user_input.strip().lower() in ["/stats", "/status", "/statistics"]:
        show_stats()
        return True
    return False

def try_handle_help_command(user_input: str) -> bool:
    """Handle /help command."""
    if user_input.strip().lower() in ["/help", "/h", "help"]:
        show_help()
        return True
    return False

def try_handle_clear_command(user_input: str) -> bool:
    """Handle /clear command."""
    if user_input.strip().lower() in ["/clear", "/reset"]:
        agent = get_chat_agent()
        if agent:
            agent.clear_history()
            console.print("[bold bright_green]✅ Chat history cleared![/bold bright_green]")
        else:
            console.print("[bold red]❌ No chat agent available[/bold red]")
        return True
    return False

def try_handle_output_command(user_input: str) -> bool:
    """Handle /output or /results commands."""
    if user_input.strip().lower() in ["/output", "/results", "/reports"]:
        # Check both old results directory and new research_results directory
        directories_to_check = [Path("research_results")]
        
        found_results = False
        for results_dir in directories_to_check:
            if results_dir.exists():
                found_results = True
                console.print(f"\n[bold bright_magenta]📁 {results_dir.name.title()} Directory:[/bold bright_magenta]")
                console.print(f"[bright_cyan]{results_dir.absolute()}[/bright_cyan]")
                
                # Show directory contents
                try:
                    md_files = list(results_dir.glob("**/*.md"))  # Recursive search
                    if md_files:
                        console.print(f"[dim]Contains {len(md_files)} report files:[/dim]")
                        
                        # Show recent files
                        recent_files = sorted(md_files, key=lambda f: f.stat().st_mtime, reverse=True)[:5]
                        for file in recent_files:
                            size_kb = file.stat().st_size / 1024
                            mtime = datetime.fromtimestamp(file.stat().st_mtime)
                            relative_path = file.relative_to(results_dir)
                            console.print(f"[dim]  • {relative_path} ({size_kb:.1f} KB, {mtime.strftime('%Y-%m-%d %H:%M')})[/dim]")
                        
                        if len(md_files) > 5:
                            console.print(f"[dim]  ... and {len(md_files) - 5} more files[/dim]")
                        
                        # Check for final comprehensive report
                        final_report = results_dir / "final_comprehensive_report.md"
                        if final_report.exists():
                            console.print(f"\n[bold green]📄 Final Comprehensive Report Available:[/bold green]")
                            console.print(f"[bright_green]{final_report.name}[/bright_green]")
                            
                            # Show preview of final report
                            try:
                                with open(final_report, 'r', encoding='utf-8') as f:
                                    content = f.read()
                                    preview = content[:2000] + "..." if len(content) > 2000 else content
                                    
                                console.print(Panel(
                                    preview,
                                    title="📄 Final Report Preview",
                                    border_style="green",
                                    padding=(1, 2)
                                ))
                            except Exception as e:
                                console.print(f"[dim red]Error reading final report: {e}[/dim red]")
                    else:
                        console.print(f"[dim yellow]Directory is empty - no reports generated yet[/dim yellow]")
                except Exception as e:
                    console.print(f"[dim red]Error reading directory: {e}[/dim red]")
        
        if not found_results:
            console.print(f"\n[dim yellow]No results directories found yet. Run a research session first.[/dim yellow]")
        
        console.print()
        return True
    return False

async def handle_chat_command(user_input: str) -> bool:
    """Handle explicit chat commands."""
    if user_input.lower().startswith("chat "):
        message = user_input[5:].strip()
        if message:
            # Remove quotes if present
            if message.startswith('"') and message.endswith('"'):
                message = message[1:-1]
            elif message.startswith("'") and message.endswith("'"):
                message = message[1:-1]
            
            await handle_chat_message(message)
        else:
            console.print("[bold red]❌ Please provide a message[/bold red]")
            console.print("[dim]Usage: chat \"your message\"[/dim]")
        return True
    return False



def main():
    """Main function for interactive mode."""
    # Interactive mode
    welcome_text = """[bold bright_magenta]🔬 Researcher[/bold bright_magenta] [bright_cyan]Interactive Pipeline[/bright_cyan]
[dim blue]Multi-agent research orchestration for comprehensive analysis[/dim blue]"""
    
    console.print(Panel.fit(
        welcome_text,
        border_style="bright_magenta",
        padding=(1, 2),
        title="[bold bright_cyan]🔬 Research Orchestration Pipeline[/bold bright_cyan]",
        title_align="center"
    ))
    
    # Show features panel
    features_text = """[bold bright_magenta]🚀 Advanced Research Features:[/bold bright_magenta]
  • [bright_cyan]Multi-Agent Research[/bright_cyan] - Advanced subagent orchestration with live UI tracking
  • [bright_cyan]Function Calling[/bright_cyan] - AI can execute research, temp_report, sub_research functions
  • [bright_cyan]Rate Limiting[/bright_cyan] - Automatic compliance with Gemini API limits
  • [bright_cyan]URL Deduplication[/bright_cyan] - Global tracking prevents duplicate URL processing
  • [bright_cyan]Live Status Dashboard[/bright_cyan] - Real-time agent progress and file count tracking
  • [bright_cyan]Final Report Synthesis[/bright_cyan] - Comprehensive integration of all research
  • [bright_cyan]Streaming responses[/bright_cyan] - Real-time response generation
  • [bright_cyan]Persistent history[/bright_cyan] - Maintains conversation context
  • [bright_cyan]Rich terminal interface[/bright_cyan] - Beautiful chat display and formatting

[bold bright_magenta]🎯 Quick Start:[/bold bright_magenta]
  • [bright_cyan]Just type your research question[/bright_cyan] - AI will determine if research is needed
  • [bright_cyan]/help[/bright_cyan] - Show all available commands
  • [bright_cyan]/results[/bright_cyan] - View research reports and final comprehensive report
  • [bright_cyan]/stats[/bright_cyan] - View session statistics
  • [bright_cyan]/clear[/bright_cyan] - Clear conversation history

[bold bright_magenta]🔬 Research Examples:[/bold bright_magenta]
  • [bright_cyan]"Analyze the current state of quantum computing"[/bright_cyan]
  • [bright_cyan]"Compare renewable energy adoption in Germany vs France"[/bright_cyan]
  • [bright_cyan]"Research the economic impact of AI on healthcare"[/bright_cyan]"""
    
    console.print(Panel(
        features_text,
        border_style="magenta",
        padding=(1, 2),
        title="[bold magenta]💡 Getting Started[/bold magenta]",
        title_align="left"
    ))
    
    # Check API key
    if not gemini_api_key:
        console.print("\n[bold red]⚠️  Warning: GEMINI_API_KEY not found in environment variables[/bold red]")
        console.print("[dim]Set your API key: export GEMINI_API_KEY=your_key_here[/dim]")
    else:
        console.print("\n[bold bright_green]✅ Ready to chat! Type your message below.[/bold bright_green]")
    
    console.print()
    
    # Interactive loop
    while True:
        try:
            user_input = prompt_session.prompt("🔬 Researcher> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold yellow]👋 Exiting gracefully...[/bold yellow]")
            break
        
        if not user_input:
            continue
        
        if user_input.lower() in ["exit", "quit"]:
            console.print("[bold bright_magenta]👋 Goodbye! Happy researching![/bold bright_magenta]")
            break
        
        # Handle commands
        if try_handle_help_command(user_input):
            continue
        
        if try_handle_stats_command(user_input):
            continue
        
        if try_handle_clear_command(user_input):
            continue
        
        if try_handle_output_command(user_input):
            continue
        
        # Handle explicit chat commands
        import asyncio
        if asyncio.run(handle_chat_command(user_input)):
            continue
        
        # Default: treat as chat message
        if gemini_api_key:
            asyncio.run(handle_chat_message(user_input))
        else:
            console.print("[bold red]❌ Cannot chat without GEMINI_API_KEY. Please set the environment variable.[/bold red]")
    
    # Show final session summary
    if session_stats['total_chat_messages'] > 0:
        console.print("\n[bold bright_green]📊 Final Session Summary:[/bold bright_green]")
        show_stats()
    
    console.print("[bold magenta]✨ Session finished. Thank you for using Researcher![/bold magenta]")


if __name__ == "__main__":
    main()
