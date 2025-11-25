#!/usr/bin/env python3
"""
IQ - Intelligent Question answering CLI for real-time web search.
Combines live web search with LLM synthesis for cited, real-time answers.
"""

import argparse
import sys
import os
from typing import Optional
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from search import WebSearcher
from llm import LLMSynthesizer
from formatting import OutputFormatter
from cache import QACache, CachedQA


class IQCLI:
    """Main CLI application for IQ."""

    def __init__(self, use_colors: bool = True, num_results: int = 8, use_cache: bool = True, streaming: bool = True, verbose: bool = False, model: str = None):
        self.formatter = OutputFormatter(use_colors=use_colors, verbose=verbose)
        self.num_results = num_results
        self.use_cache = use_cache
        self.streaming = streaming

        try:
            self.searcher = WebSearcher()
            # Use model from env or default to gpt-3.5-turbo
            llm_model = model or os.getenv("IQ_MODEL", "gpt-3.5-turbo")
            self.synthesizer = LLMSynthesizer(model=llm_model)
            self.cache = QACache()
        except ValueError as e:
            self.formatter.print_error(str(e))
            sys.exit(1)
    
    def process_query(self, query: str, force_refresh: bool = False) -> bool:
        """
        Process a single query through the full pipeline.
        
        Args:
            query: User's search query
            force_refresh: If True, skip cache and force fresh search
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if hasattr(self, 'verbose') and self.verbose:
                print(f"DEBUG: process_query called with verbose={self.verbose}")
            
            # Step 1: Check cache (only if explicitly requested and not force refresh)
            if hasattr(self, 'use_cache_retrieval') and self.use_cache_retrieval and self.use_cache and not force_refresh:
                cached_answer = self.cache.find_exact_match(query)
                if cached_answer:
                    self.formatter.print_success("Found exact match in cache")
                    print()  # Add spacing
                    
                    # Convert cached search result dictionaries back to SearchResult objects
                    from search import SearchResult
                    search_results = []
                    for result_dict in cached_answer.search_results:
                        search_results.append(SearchResult(
                            title=result_dict.get('title', 'Unknown'),
                            url=result_dict.get('url', ''),
                            snippet=result_dict.get('snippet', '')
                        ))
                    
                    self.formatter.print_response(cached_answer.answer, search_results)
                    return True
                
                # Check for similar questions with higher threshold (95% similarity)
                similar_questions = self.cache.find_similar_questions(query, min_similarity=0.95, max_results=1)
                if similar_questions:
                    qa, similarity = similar_questions[0]  # Get the best match
                    self.formatter.print_success(f"Found similar question in cache (similarity: {similarity:.1%})")
                    print(f"Original question: {qa.question}")
                    print()  # Add spacing
                    
                    # Convert cached search result dictionaries back to SearchResult objects
                    from search import SearchResult
                    search_results = []
                    for result_dict in qa.search_results:
                        search_results.append(SearchResult(
                            title=result_dict.get('title', 'Unknown'),
                            url=result_dict.get('url', ''),
                            snippet=result_dict.get('snippet', '')
                        ))
                    
                    self.formatter.print_response(qa.answer, search_results)
                    return True
            
            # Step 2: Search
            self.formatter.print_status("Searching the web")
            search_results = self.searcher.search(query, num_results=self.num_results)
            
            if not search_results:
                self.formatter.print_error("No search results found")
                return False
            
            self.formatter.print_success(f"Found {len(search_results)} results")
            
            # Step 3: Synthesize
            if self.streaming:
                self.formatter.print_status("Generating answer")
                print()  # Add spacing before streaming output
                answer = self.synthesizer.synthesize_answer(query, search_results, streaming=True)
                print()  # Add spacing after streaming output
            else:
                self.formatter.print_status("Synthesizing answer")
                answer = self.synthesizer.synthesize_answer(query, search_results, streaming=False)
                self.formatter.print_success("Answer generated")
                print()  # Add spacing
                # Display the answer
                self.formatter.print_response(answer, search_results)
            
            # Step 4: Cache the result
            if self.use_cache:
                self.cache.store_qa(query, answer, search_results)
            
            # Step 5: Display citations (for streaming mode, we already printed the answer)
            if self.streaming:
                print()  # Add spacing before citations
                self.formatter.print_citations(search_results)
            
            return True
            
        except Exception as e:
            self.formatter.print_error(f"Failed to process query: {e}")
            return False
    
    def interactive_mode(self):
        """Run IQ in interactive mode."""
        console = Console()

        # Welcome banner
        welcome_panel = Panel(
            "[bold cyan]Welcome to IQ Interactive Mode![/bold cyan]\n\n"
            "[dim]Ask any question and get real-time answers from the web.[/dim]\n\n"
            "[yellow]Commands:[/yellow]\n"
            "  [cyan]/cache list[/cyan]           List recent entries\n"
            "  [cyan]/cache search[/cyan] TERM    Search cache\n"
            "  [cyan]/cache stats[/cyan]          Show statistics\n"
            "  [cyan]quit[/cyan] or [cyan]exit[/cyan]          Exit\n\n"
            "[dim]Press Ctrl+C to exit anytime[/dim]",
            border_style="bright_cyan",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(welcome_panel)
        print()
        
        while True:
            try:
                query = input("❓ Your question: ").strip()
                
                if not query:
                    continue
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print(self.formatter.format_success("Goodbye!"))
                    break
                
                # Handle cache commands in interactive mode
                if query.startswith('/cache'):
                    if not self.use_cache:
                        self.formatter.print_error("Cache is disabled")
                        continue
                    
                    parts = query.split(' ', 2)
                    if len(parts) < 2:
                        print(self.formatter.format_info("Cache commands: list, search <term>, stats, clear"))
                        continue
                    
                    command = parts[1].lower()
                    
                    if command == 'list':
                        self.list_cache_command()
                    elif command == 'search' and len(parts) >= 3:
                        self.search_cache_command(parts[2])
                    elif command == 'search':
                        print(self.formatter.format_info("Usage: /cache search <search term>"))
                    elif command == 'stats':
                        self.cache_stats_command()
                    elif command == 'clear':
                        self.clear_cache_command()
                    else:
                        print(self.formatter.format_info("Cache commands: list, search <term>, stats, clear"))
                    
                    print()
                    continue
                
                print()  # Add spacing before processing
                success = self.process_query(query)
                
                if success:
                    print("\n" + "─" * 50 + "\n")  # Separator between queries
                
            except KeyboardInterrupt:
                print(f"\n{self.formatter.format_success('Goodbye!')}")
                break
            except EOFError:
                print(f"\n{self.formatter.format_success('Goodbye!')}")
                break

    def list_cache_command(self, count: int = 10):
        """List recent cached Q&As."""
        if not self.use_cache:
            self.formatter.print_error("Cache is disabled")
            return
        
        recent_entries = self.cache.get_recent_entries(count)
        
        if not recent_entries:
            print(self.formatter.format_info("No cached entries found"))
            return
        
        print(self.formatter.format_success(f"Recent {len(recent_entries)} cached Q&As:"))
        print()
        
        for i, qa in enumerate(recent_entries, 1):
            cache_time = datetime.fromisoformat(qa.timestamp)
            print(f"{self.formatter.format_info(f'{i}.')} {qa.question}")
            print(f"   {cache_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   {qa.answer[:100]}{'...' if len(qa.answer) > 100 else ''}")
            print()

    def search_cache_command(self, search_term: str):
        """Search through cached Q&As."""
        if not self.use_cache:
            self.formatter.print_error("Cache is disabled")
            return
        
        matches = self.cache.find_similar_questions(search_term, min_similarity=0.8, max_results=10)
        
        if not matches:
            print(self.formatter.format_info(f"No cached entries found matching '{search_term}'"))
            return
        
        print(self.formatter.format_success(f"Found {len(matches)} cached entries matching '{search_term}':"))
        print()
        
        for i, (qa, similarity) in enumerate(matches, 1):
            cache_time = datetime.fromisoformat(qa.timestamp)
            print(f"{self.formatter.format_info(f'{i}.')} {qa.question} (similarity: {similarity:.1%})")
            print(f"   {cache_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   {qa.answer[:100]}{'...' if len(qa.answer) > 100 else ''}")
            print()

    def cache_stats_command(self):
        """Show cache statistics."""
        if not self.use_cache:
            self.formatter.print_error("Cache is disabled")
            return
        
        stats = self.cache.get_cache_stats()
        
        print(self.formatter.format_success("Cache Statistics:"))
        print(f"Total entries: {stats['total_entries']}")
        print(f"Cache size: {stats['cache_size_mb']} MB")
        
        if stats['newest_entry']:
            newest = datetime.fromisoformat(stats['newest_entry'])
            print(f"Newest entry: {newest.strftime('%Y-%m-%d %H:%M')}")
        
        if stats['oldest_entry']:
            oldest = datetime.fromisoformat(stats['oldest_entry'])
            print(f"Oldest entry: {oldest.strftime('%Y-%m-%d %H:%M')}")

    def clear_cache_command(self):
        """Clear all cached entries."""
        if not self.use_cache:
            self.formatter.print_error("Cache is disabled")
            return
        
        self.cache.clear_cache()
        print(self.formatter.format_success("Cache cleared successfully"))
    
    def config_list_command(self):
        """List current configuration from .env file."""
        env_path = os.path.join(os.getcwd(), '.env')

        if not os.path.exists(env_path):
            self.formatter.print_error(".env file not found. Copy .env.example to .env to get started.")
            return

        self.formatter.print_success("Current Configuration:")
        print()

        # Read and display .env contents
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue
                # Mask API keys for security
                if '=' in line:
                    key, value = line.split('=', 1)
                    if 'KEY' in key.upper() and value and value != 'your_' + key.lower():
                        masked_value = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                        print(f"{self.formatter.format_info(key)}: {masked_value}")
                    else:
                        print(f"{self.formatter.format_info(key)}: {value}")

    def config_set_command(self, key: str, value: str):
        """Set a configuration value in .env file."""
        env_path = os.path.join(os.getcwd(), '.env')

        if not os.path.exists(env_path):
            self.formatter.print_error(".env file not found. Copy .env.example to .env first.")
            return

        # Read current .env contents
        with open(env_path, 'r') as f:
            lines = f.readlines()

        # Update or add the key
        key_found = False
        for i, line in enumerate(lines):
            if line.strip().startswith(key + '='):
                lines[i] = f"{key}={value}\n"
                key_found = True
                break

        if not key_found:
            lines.append(f"{key}={value}\n")

        # Write back to .env
        with open(env_path, 'w') as f:
            f.writelines(lines)

        self.formatter.print_success(f"Configuration updated: {key}={value}")
        print(self.formatter.format_info("Restart IQ for changes to take effect."))

    def config_get_command(self, key: str):
        """Get a configuration value from .env file."""
        env_path = os.path.join(os.getcwd(), '.env')

        if not os.path.exists(env_path):
            self.formatter.print_error(".env file not found. Copy .env.example to .env first.")
            return

        # Read .env and find the key
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith(key + '='):
                    _, value = line.split('=', 1)
                    # Mask API keys for security
                    if 'KEY' in key.upper() and value and value != 'your_' + key.lower():
                        masked_value = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                        print(f"{self.formatter.format_info(key)}: {masked_value}")
                    else:
                        print(f"{self.formatter.format_info(key)}: {value}")
                    return

        self.formatter.print_error(f"Configuration key '{key}' not found.")

    def config_models_command(self):
        """List all available models by provider."""
        console = Console()

        self.formatter.print_success("Available Models by Provider:")
        print()

        models_data = {
            "OpenAI": {
                "description": "Get API key at: https://platform.openai.com/api-keys",
                "env_var": "OPENAI_API_KEY",
                "models": [
                    ("gpt-3.5-turbo", "Default, fast & cost-effective"),
                    ("gpt-4", "Most capable"),
                    ("gpt-4-turbo", "Fast GPT-4 variant"),
                    ("gpt-4o", "Latest GPT-4 variant with vision")
                ]
            },
            "Anthropic Claude": {
                "description": "Get API key at: https://console.anthropic.com/",
                "env_var": "ANTHROPIC_API_KEY",
                "models": [
                    ("claude-3-haiku-20240307", "Fastest"),
                    ("claude-3-sonnet-20240229", "Balanced"),
                    ("claude-3-opus-20240229", "Most capable"),
                    ("claude-3-5-sonnet-20241022", "Latest, most capable")
                ]
            },
            "Google Gemini": {
                "description": "Get API key at: https://makersuite.google.com/app/apikey",
                "env_var": "GOOGLE_API_KEY",
                "models": [
                    ("gemini-pro", "Fast & capable"),
                    ("gemini-1.5-pro", "Most capable"),
                    ("gemini-1.5-flash", "Fastest")
                ]
            },
            "OpenRouter": {
                "description": "Get API key at: https://openrouter.ai/keys",
                "env_var": "OPENROUTER_API_KEY",
                "models": [
                    ("anthropic/claude-3-sonnet", "Claude via OpenRouter"),
                    ("meta-llama/llama-3-70b", "Llama 3 70B"),
                    ("google/gemini-pro", "Gemini via OpenRouter")
                ]
            },
            "Perplexity": {
                "description": "Get API key at: https://www.perplexity.ai/settings/api",
                "env_var": "PERPLEXITY_API_KEY",
                "models": [
                    ("pplx-7b-online", "Fast with web access"),
                    ("pplx-70b-online", "Most capable with web access")
                ]
            },
            "Ollama (Local)": {
                "description": "Install from: https://ollama.ai (no API key needed)",
                "env_var": "OLLAMA_BASE_URL (optional)",
                "models": [
                    ("llama2", "Meta's Llama 2"),
                    ("mistral", "Mistral 7B"),
                    ("mixtral", "Mixtral 8x7B"),
                    ("codellama", "Code Llama"),
                    ("phi", "Microsoft Phi")
                ]
            }
        }

        for provider, data in models_data.items():
            console.print(f"\n[bold cyan]{provider}[/bold cyan]")
            console.print(f"[dim]{data['description']}[/dim]")
            console.print(f"[yellow]Env var:[/yellow] {data['env_var']}")

            models_table = Table(show_header=False, box=None, padding=(0, 2))
            models_table.add_column(style="green", no_wrap=True)
            models_table.add_column(style="dim")

            for model, description in data['models']:
                models_table.add_row(model, description)

            console.print(models_table)

        print()
        print(self.formatter.format_info("Usage: iq --model <model_name> \"your question\""))
        print(self.formatter.format_info("Or set IQ_MODEL in .env: iq --config-set IQ_MODEL gpt-4"))

    def random_question_command(self):
        """Generate and answer a random interesting question."""
        import random
        
        # Categories and question templates for generating interesting questions
        question_templates = [
            # History & Culture
            "What were the long-term consequences of {historical_event} on modern society?",
            "How did {ancient_civilization} influence contemporary {field}?",
            "What can we learn from the rise and fall of {empire} that applies to today's world?",
            "How has the concept of {cultural_concept} evolved from ancient times to now?",
            
            # Science & Technology
            "What are the latest breakthroughs in {scientific_field} and their potential impact?",
            "How is {emerging_technology} changing the way we approach {domain}?",
            "What ethical considerations arise from advances in {tech_field}?",
            "How might {scientific_discovery} reshape our understanding of {concept}?",
            
            # Society & Philosophy
            "How is {social_phenomenon} affecting {demographic} in the 21st century?",
            "What role does {institution} play in addressing modern {challenge}?",
            "How has {philosophical_concept} been reinterpreted in contemporary {context}?",
            "What are the sociological implications of {trend} on future generations?",
            
            # Economics & Politics
            "How are {economic_factor} and {political_system} interconnected in today's world?",
            "What lessons can {country} teach us about {policy_area}?",
            "How is {global_trend} reshaping international {domain}?",
            "What are the unintended consequences of {policy_type} in modern societies?",
            
            # Environment & Future
            "How are communities adapting to {environmental_challenge} around the world?",
            "What innovative solutions are emerging to address {sustainability_issue}?",
            "How might {environmental_factor} influence {aspect} in the next decade?",
            "What can indigenous knowledge teach us about {environmental_topic}?"
        ]
        
        # Replacement values for templates
        replacements = {
            'historical_event': ['the Industrial Revolution', 'the fall of the Berlin Wall', 'the invention of the printing press', 'the Silk Road trade routes', 'the Renaissance'],
            'ancient_civilization': ['Ancient Rome', 'the Maya', 'Ancient Egypt', 'the Indus Valley civilization', 'Ancient Greece'],
            'empire': ['the Ottoman Empire', 'the British Empire', 'the Mongol Empire', 'the Roman Empire', 'the Spanish Empire'],
            'cultural_concept': ['democracy', 'justice', 'art', 'education', 'family structures'],
            'field': ['architecture', 'governance', 'philosophy', 'medicine', 'urban planning'],
            'scientific_field': ['quantum computing', 'CRISPR gene editing', 'artificial intelligence', 'neuroscience', 'climate science'],
            'emerging_technology': ['artificial intelligence', 'blockchain', 'quantum computing', 'biotechnology', 'renewable energy'],
            'tech_field': ['artificial intelligence', 'genetic engineering', 'surveillance technology', 'social media algorithms', 'automation'],
            'scientific_discovery': ['CRISPR', 'gravitational waves', 'exoplanets', 'the human microbiome', 'quantum entanglement'],
            'domain': ['education', 'healthcare', 'transportation', 'communication', 'entertainment'],
            'concept': ['consciousness', 'time', 'reality', 'human behavior', 'the universe'],
            'social_phenomenon': ['remote work', 'social media', 'urbanization', 'digital nomadism', 'the gig economy'],
            'demographic': ['Gen Z', 'millennials', 'rural communities', 'urban populations', 'elderly populations'],
            'institution': ['education', 'healthcare systems', 'democratic institutions', 'the family unit', 'religious organizations'],
            'challenge': ['climate change', 'inequality', 'mental health', 'technological disruption', 'political polarization'],
            'philosophical_concept': ['freedom', 'justice', 'truth', 'beauty', 'moral responsibility'],
            'context': ['digital age', 'post-pandemic world', 'globalized society', 'multicultural societies', 'technological era'],
            'trend': ['artificial intelligence', 'climate change', 'globalization', 'demographic shifts', 'technological advancement'],
            'economic_factor': ['inflation', 'cryptocurrency', 'automation', 'globalization', 'income inequality'],
            'political_system': ['democracy', 'authoritarianism', 'federalism', 'international cooperation', 'populism'],
            'country': ['Denmark', 'Singapore', 'Costa Rica', 'Rwanda', 'South Korea'],
            'policy_area': ['education', 'healthcare', 'environmental protection', 'social welfare', 'innovation'],
            'global_trend': ['climate change', 'digitalization', 'demographic transition', 'urbanization', 'economic inequality'],
            'policy_type': ['universal basic income', 'carbon taxes', 'digital privacy laws', 'immigration policies', 'education reform'],
            'environmental_challenge': ['rising sea levels', 'extreme weather', 'water scarcity', 'biodiversity loss', 'air pollution'],
            'sustainability_issue': ['plastic pollution', 'food waste', 'renewable energy', 'sustainable agriculture', 'circular economy'],
            'environmental_factor': ['climate change', 'deforestation', 'ocean acidification', 'urban heat islands', 'biodiversity loss'],
            'aspect': ['urban planning', 'agriculture', 'migration patterns', 'economic systems', 'social structures'],
            'environmental_topic': ['sustainable agriculture', 'forest management', 'water conservation', 'climate adaptation', 'biodiversity preservation']
        }
        
        # Generate a random question
        template = random.choice(question_templates)
        
        # Replace placeholders with random values
        question = template
        for placeholder, values in replacements.items():
            if '{' + placeholder + '}' in question:
                question = question.replace('{' + placeholder + '}', random.choice(values))
        
        # Display the generated question
        self.formatter.print_status("Generated random question:")
        print(f"\n{self.formatter.format_info('Q:')} {question}\n")
        
        # Process the question through the normal pipeline
        success = self.process_query(question, force_refresh=False)
        
        if not success:
            self.formatter.print_error("Failed to generate answer for the random question")
            return
        
        # Add a note about the random question feature
        print(f"\n{self.formatter.format_info('💡 Tip:')} Use 'iq --random' anytime for another interesting question!")


def print_beautiful_help():
    """Print a beautiful, custom help message using Rich."""
    console = Console()

    # Banner
    banner = Text()
    banner.append("█ ██▀▀█\n", style="bold cyan")
    banner.append("█ █ ▄▄█\n", style="bold cyan")
    banner.append("█ ▀▀▀█▄", style="bold cyan")

    title = Text("\nIQ - Intelligent Question Answering\n", style="bold bright_white")
    subtitle = Text("Real-time web search powered by AI", style="dim cyan")

    header_content = Text()
    header_content.append(banner)
    header_content.append(title)
    header_content.append(subtitle)

    console.print(Panel(
        header_content,
        border_style="bright_cyan",
        box=box.DOUBLE,
        padding=(1, 2)
    ))

    # Usage section
    console.print("\n[bold bright_white]USAGE[/bold bright_white]")
    console.print("  [cyan]iq[/cyan] [dim]<query>[/dim]              Ask a question")
    console.print("  [cyan]iq[/cyan]                       Interactive mode")
    console.print("  [cyan]iq[/cyan] [yellow]--help[/yellow]               Show this help")

    # Quick Start Examples
    console.print("\n[bold bright_white]QUICK START[/bold bright_white]")
    examples_table = Table(show_header=False, box=None, padding=(0, 2))
    examples_table.add_column(style="cyan", no_wrap=True)
    examples_table.add_column(style="dim")

    examples_table.add_row("iq", "Start interactive mode")
    examples_table.add_row('iq "Latest AI news"', "Ask a single question")
    examples_table.add_row("iq --random", "Get a random interesting question")

    console.print(examples_table)

    # Options grouped by category
    console.print("\n[bold bright_white]SEARCH OPTIONS[/bold bright_white]")
    search_table = Table(show_header=False, box=None, padding=(0, 2))
    search_table.add_column(style="yellow", no_wrap=True, width=25)
    search_table.add_column(style="white")

    search_table.add_row("--results N", "Number of search results (default: 8)")
    search_table.add_row("--no-streaming", "Show complete answer at once")
    search_table.add_row("--random", "Generate random question")

    console.print(search_table)

    # Cache options
    console.print("\n[bold bright_white]CACHE OPTIONS[/bold bright_white]")
    cache_table = Table(show_header=False, box=None, padding=(0, 2))
    cache_table.add_column(style="yellow", no_wrap=True, width=25)
    cache_table.add_column(style="white")

    cache_table.add_row("--use-cache", "Use cached answers if available")
    cache_table.add_row("--no-cache", "Disable caching for this query")
    cache_table.add_row("--force-refresh", "Skip cache, force fresh search")
    cache_table.add_row("--cache-list", "List recent cached Q&As")
    cache_table.add_row("--cache-search TERM", "Search cached Q&As")
    cache_table.add_row("--cache-stats", "Show cache statistics")
    cache_table.add_row("--cache-clear", "Clear all cached entries")

    console.print(cache_table)

    # Config options
    console.print("\n[bold bright_white]CONFIG OPTIONS[/bold bright_white]")
    config_table = Table(show_header=False, box=None, padding=(0, 2))
    config_table.add_column(style="yellow", no_wrap=True, width=25)
    config_table.add_column(style="white")

    config_table.add_row("--config-list", "List current configuration")
    config_table.add_row("--config-set KEY VALUE", "Set config value")
    config_table.add_row("--config-get KEY", "Get config value")
    config_table.add_row("--config-models", "List all available models")

    console.print(config_table)

    # Display options
    console.print("\n[bold bright_white]DISPLAY OPTIONS[/bold bright_white]")
    display_table = Table(show_header=False, box=None, padding=(0, 2))
    display_table.add_column(style="yellow", no_wrap=True, width=25)
    display_table.add_column(style="white")

    display_table.add_row("--no-color", "Disable colored output")
    display_table.add_row("--verbose", "Enable debug output")
    display_table.add_row("--version", "Show version number")

    console.print(display_table)

    # Advanced Examples
    console.print("\n[bold bright_white]EXAMPLES[/bold bright_white]")

    example_panel_1 = Panel(
        "[cyan]iq[/cyan] [dim]\"What are the latest developments in quantum computing?\"[/dim]",
        title="[yellow]Basic Search[/yellow]",
        border_style="dim",
        box=box.ROUNDED
    )
    console.print(example_panel_1)

    example_panel_2 = Panel(
        "[cyan]iq[/cyan] [yellow]--use-cache[/yellow] [yellow]--results[/yellow] [green]5[/green] [dim]\"Python best practices\"[/dim]",
        title="[yellow]With Options[/yellow]",
        border_style="dim",
        box=box.ROUNDED
    )
    console.print(example_panel_2)

    example_panel_3 = Panel(
        "[cyan]iq[/cyan] [yellow]--cache-search[/yellow] [dim]\"machine learning\"[/dim]",
        title="[yellow]Search Cache[/yellow]",
        border_style="dim",
        box=box.ROUNDED
    )
    console.print(example_panel_3)

    # Interactive mode tips
    tips_panel = Panel(
        "[bold]Interactive Mode Commands:[/bold]\n\n"
        "  [cyan]/cache list[/cyan]           List recent entries\n"
        "  [cyan]/cache search[/cyan] TERM    Search cache\n"
        "  [cyan]/cache stats[/cyan]          Show statistics\n"
        "  [cyan]/cache clear[/cyan]          Clear cache\n"
        "  [cyan]quit[/cyan] or [cyan]exit[/cyan]          Exit interactive mode",
        title="[bright_cyan]💡 Tips[/bright_cyan]",
        border_style="bright_cyan",
        box=box.ROUNDED
    )
    console.print("\n")
    console.print(tips_panel)

    # Footer
    console.print("\n[dim]Cache Location:[/dim] [cyan]~/.iq_cache/qa_cache.json[/cyan]")
    console.print("[dim]Documentation:[/dim] [cyan]https://github.com/vowalsh/iq-cli[/cyan]")
    console.print("[dim]Made by[/dim] [bright_cyan]@vowalsh[/bright_cyan]\n")


def main():
    """Main entry point for the CLI."""
    # Check for help flag first
    if '--help' in sys.argv or '-h' in sys.argv:
        print_beautiful_help()
        sys.exit(0)

    parser = argparse.ArgumentParser(
        description="IQ - Intelligent Question answering with real-time web search",
        add_help=False  # We'll use custom help
    )

    parser.add_argument(
        "query",
        nargs="?",
        help="Search query (if not provided, enters interactive mode)"
    )

    parser.add_argument(
        "--help", "-h",
        action="store_true",
        help="Show help message"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    parser.add_argument(
        "--results",
        type=int,
        default=8,
        help="Number of search results to fetch (default: 8)"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching"
    )
    
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force fresh search, skip cache"
    )
    
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached answers if available (exact or similar matches)"
    )
    
    parser.add_argument(
        "--cache-search",
        type=str,
        help="Search through cached Q&As"
    )
    
    parser.add_argument(
        "--cache-list",
        action="store_true",
        help="List recent cached Q&As"
    )
    
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics"
    )
    
    parser.add_argument(
        "--cache-clear",
        action="store_true",
        help="Clear all cached entries"
    )
    
    parser.add_argument(
        "--random",
        action="store_true",
        help="Generate and answer a random interesting question"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="IQ 1.0.0"
    )
    
    parser.add_argument(
        "--no-streaming",
        action="store_true",
        help="Disable streaming output (show complete answer at once)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug output"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="LLM model to use (gpt-3.5-turbo, gpt-4, claude-3-sonnet-20240229, etc.)"
    )

    parser.add_argument(
        "--config-list",
        action="store_true",
        help="List current configuration"
    )

    parser.add_argument(
        "--config-set",
        nargs=2,
        metavar=("KEY", "VALUE"),
        help="Set configuration value (e.g., --config-set IQ_MODEL gpt-4)"
    )

    parser.add_argument(
        "--config-get",
        type=str,
        metavar="KEY",
        help="Get configuration value"
    )

    parser.add_argument(
        "--config-models",
        action="store_true",
        help="List all available models by provider"
    )

    # Check for unparsed arguments that might indicate missing quotes
    args, unknown = parser.parse_known_args()

    if unknown:
        console = Console()
        console.print("[red]✗ Error:[/red] Unrecognized arguments detected.", style="bold")
        console.print("\n[yellow]Did you forget to use quotes around your question?[/yellow]")
        console.print("\n[dim]Examples of correct usage:[/dim]")
        console.print('  [cyan]iq[/cyan] [green]"What is the weather today?"[/green]')
        console.print('  [cyan]iq[/cyan] [green]"How does quantum computing work?"[/green]')
        console.print("\n[dim]Run[/dim] [cyan]iq --help[/cyan] [dim]for more information.[/dim]")
        sys.exit(1)

    # Initialize CLI
    use_colors = not args.no_color
    enable_cache = not args.no_cache  # Cache storage is always enabled unless --no-cache
    use_cache_retrieval = args.use_cache  # Cache retrieval only if --use-cache flag is used
    streaming = not args.no_streaming  # Default to True, disable with --no-streaming
    verbose = args.verbose
    model = args.model if hasattr(args, 'model') else None
    cli = IQCLI(use_colors=use_colors, num_results=args.results, use_cache=enable_cache, streaming=streaming, verbose=verbose, model=model)
    cli.use_cache_retrieval = use_cache_retrieval

    # Handle config commands
    if args.config_list:
        cli.config_list_command()
        return

    if args.config_set:
        cli.config_set_command(args.config_set[0], args.config_set[1])
        return

    if args.config_get:
        cli.config_get_command(args.config_get)
        return

    if args.config_models:
        cli.config_models_command()
        return

    # Handle cache commands
    if args.cache_search:
        cli.search_cache_command(args.cache_search)
        return
    
    if args.cache_list:
        cli.list_cache_command()
        return
    
    if args.cache_stats:
        cli.cache_stats_command()
        return
    
    if args.cache_clear:
        cli.clear_cache_command()
        return
    
    if args.random:
        cli.random_question_command()
        return
    
    if args.query:
        # Single query mode
        success = cli.process_query(args.query, force_refresh=args.force_refresh)
        sys.exit(0 if success else 1)
    else:
        # Interactive mode
        cli.interactive_mode()


if __name__ == "__main__":
    main()
