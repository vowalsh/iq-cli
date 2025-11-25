"""
Output formatting functionality for IQ CLI.
Handles displaying answers with citations and source mapping.
"""

from typing import List
from colorama import Fore, Style, init
from search import SearchResult
from charts import ChartGenerator

# Initialize colorama for cross-platform color support
init(autoreset=True)


class OutputFormatter:
    """Handles formatting and displaying IQ output."""
    
    def __init__(self, use_colors: bool = True, verbose: bool = False):
        self.use_colors = use_colors
        self.verbose = verbose
        self.chart_generator = ChartGenerator(use_colors=use_colors, verbose=verbose)
    
    def format_response(self, answer: str, search_results: List[SearchResult]) -> str:
        """
        Format the complete response with answer and citations.
        
        Args:
            answer: LLM-generated answer with inline citations
            search_results: List of SearchResult objects for source mapping
            
        Returns:
            Formatted response string
        """
        output = []
        
        # Add header
        output.append(self._format_header("IQ Answer"))
        output.append("")
        
        # Add the synthesized answer
        output.append(self._format_answer(answer))
        output.append("")
        
        # Generate and add charts if applicable
        if self.verbose:
            print("DEBUG: Calling chart_generator.generate_charts()")
        chart_output = self.chart_generator.generate_charts(answer, search_results)
        if self.verbose:
            print(f"DEBUG: chart_output = {chart_output is not None}")
        if chart_output:
            output.append(chart_output)
            output.append("")
        
        # Add sources section
        if search_results:
            output.append(self._format_sources_header())
            output.append("")
            
            for i, result in enumerate(search_results, 1):
                output.append(self._format_source(i, result))
                output.append("")
        
        return "\n".join(output)
    
    def format_error(self, error_message: str) -> str:
        """Format error messages."""
        if self.use_colors:
            return f"{Fore.RED}Error: {error_message}{Style.RESET_ALL}"
        return f"Error: {error_message}"
    
    def format_status(self, message: str) -> str:
        """Format status messages."""
        if self.use_colors:
            return f"{Fore.YELLOW}⏳ {message}...{Style.RESET_ALL}"
        return f"⏳ {message}..."
    
    def format_success(self, message: str) -> str:
        """Format success messages."""
        if self.use_colors:
            return f"{Fore.GREEN}✓ {message}{Style.RESET_ALL}"
        return f"✓ {message}"
    
    def format_info(self, message: str) -> str:
        """Format info messages."""
        if self.use_colors:
            return f"{Fore.CYAN}ℹ {message}{Style.RESET_ALL}"
        return f"ℹ {message}"
    
    def _format_header(self, title: str) -> str:
        """Format the main header."""
        if self.use_colors:
            return f"{Fore.CYAN}{Style.BRIGHT}{'=' * 50}\n{title.center(50)}\n{'=' * 50}{Style.RESET_ALL}"
        return f"{'=' * 50}\n{title.center(50)}\n{'=' * 50}"
    
    def _format_answer(self, answer: str) -> str:
        """Format the main answer text."""
        if self.use_colors:
            # Highlight citations in the answer
            formatted_answer = answer
            # Simple highlighting of [1], [2], etc.
            import re
            citation_pattern = r'\[(\d+)\]'
            formatted_answer = re.sub(
                citation_pattern, 
                f'{Fore.BLUE}[\\1]{Style.RESET_ALL}', 
                formatted_answer
            )
            return formatted_answer
        return answer
    
    def _format_sources_header(self) -> str:
        """Format the sources section header."""
        if self.use_colors:
            return f"{Fore.MAGENTA}{Style.BRIGHT}Sources:{Style.RESET_ALL}"
        return "Sources:"
    
    def _format_source(self, number: int, result: SearchResult) -> str:
        """Format a single source entry."""
        if self.use_colors:
            return (f"{Fore.BLUE}[{number}]{Style.RESET_ALL} "
                   f"{Fore.WHITE}{Style.BRIGHT}{result.title}{Style.RESET_ALL}\n"
                   f"    {Fore.CYAN}{result.url}{Style.RESET_ALL}\n"
                   f"    {Fore.WHITE}{result.snippet}{Style.RESET_ALL}")
        else:
            return (f"[{number}] {result.title}\n"
                   f"    {result.url}\n"
                   f"    {result.snippet}")
    
    def print_response(self, answer: str, search_results: List[SearchResult]):
        """Print the complete formatted response."""
        print(self.format_response(answer, search_results))
    
    def print_citations(self, search_results: List[SearchResult]):
        """Print just the citations/sources section."""
        if search_results:
            print(self._format_sources_header())
            print()
            
            for i, result in enumerate(search_results, 1):
                print(self._format_source(i, result))
                print()

    def print_error(self, error_message: str):
        """Print formatted error message."""
        print(self.format_error(error_message))
    
    def print_status(self, message: str):
        """Print formatted status message."""
        print(self.format_status(message))
    
    def print_success(self, message: str):
        """Print formatted success message."""
        print(self.format_success(message))
