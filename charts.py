"""
Chart generation functionality for IQ CLI.
Detects numeric/statistical data and generates terminal charts using rich.
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.columns import Columns
from search import SearchResult


class ChartGenerator:

    """Generates terminal charts and visualizations for numeric data."""
    
    def __init__(self, use_colors: bool = True, verbose: bool = False):
        self.console = Console(color_system="auto" if use_colors else None)
        self.use_colors = use_colors
        self.verbose = verbose
    
    def should_generate_chart(self, answer: str, search_results: List[SearchResult]) -> bool:
        """
        Determine if the content contains numeric/statistical data suitable for charting.
        
        Args:
            answer: LLM-generated answer
            search_results: Search results that informed the answer
            
        Returns:
            True if charts should be generated
        """
        # Keywords that suggest statistical/numeric content
        chart_keywords = [
            'percentage', 'percent', '%', 'statistics', 'data', 'numbers',
            'growth', 'increase', 'decrease', 'trend', 'rate', 'ratio',
            'comparison', 'versus', 'vs', 'higher', 'lower', 'average',
            'median', 'mean', 'total', 'sum', 'count', 'distribution',
            'market share', 'revenue', 'profit', 'sales', 'price',
            'temperature', 'population', 'gdp', 'inflation', 'unemployment',
            'trillion', 'billion', 'million', 'compare', 'comparison'
        ]
        
        # Check if answer contains chart-worthy keywords
        answer_lower = answer.lower()
        has_keywords = any(keyword in answer_lower for keyword in chart_keywords)
        
        # Simplified numeric pattern checks
        has_percentages = '%' in answer or 'percent' in answer_lower
        has_dollar_amounts = '$' in answer
        has_scale_words = any(word in answer_lower for word in ['trillion', 'billion', 'million'])
        has_numbers = bool(re.search(r'\d+', answer))
        
        # If we have GDP-related keywords and numbers, generate charts
        has_gdp_content = 'gdp' in answer_lower and has_numbers
        
        result = (has_keywords and (has_percentages or has_dollar_amounts or has_scale_words)) or has_gdp_content
        
        return result
    
    def extract_numeric_data(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract numeric data points from text.
        
        Args:
            text: Text to extract data from
            
        Returns:
            List of dictionaries containing extracted data points
        """
        data_points = []
        
        # Common country names for better extraction
        countries = [
            'United States', 'USA', 'US', 'Japan', 'Germany', 'United Kingdom', 'UK', 
            'France', 'Italy', 'Canada', 'China', 'India', 'Brazil', 'Russia',
            'Australia', 'South Korea', 'Mexico', 'Spain', 'Netherlands', 'Switzerland'
        ]
        
        def clean_label(label: str) -> str:
            """Clean up extracted labels to get proper country/entity names."""
            # Remove common prefixes and suffixes
            label = label.strip()
            
            # Remove leading articles and prepositions
            prefixes_to_remove = ['the ', 'and ', 'in ', 'of ', 'for ', 'with ', 'by ']
            for prefix in prefixes_to_remove:
                if label.lower().startswith(prefix):
                    label = label[len(prefix):]
            
            # Remove trailing words that indicate actions or descriptions
            suffixes_to_remove = [' had', ' has', ' was', ' were', ' is', ' are', ' with', ' at', ' of']
            for suffix in suffixes_to_remove:
                if label.lower().endswith(suffix):
                    label = label[:-len(suffix)]
            
            # Check if the cleaned label contains a known country name
            label_lower = label.lower()
            for country in countries:
                if country.lower() in label_lower:
                    return country
            
            # Capitalize first letter of each word for consistency
            return ' '.join(word.capitalize() for word in label.split())
        
        
        # Pattern for percentages
        percentage_pattern = r'(\w+(?:\s+\w+)*)\s*:?\s*(\d+(?:\.\d+)?)\s*%'
        for match in re.finditer(percentage_pattern, text, re.IGNORECASE):
            label = clean_label(match.group(1))
            value = float(match.group(2))
            data_points.append({
                'label': label,
                'value': value,
                'type': 'percentage',
                'unit': '%'
            })
        
        # Pattern for currency values with scale (trillion, billion, million)
        currency_scale_pattern = r'(\w+(?:\s+\w+)*)\s*:?\s*\$?([\d,]+(?:\.\d+)?)\s*(trillion|billion|million)'
        for match in re.finditer(currency_scale_pattern, text, re.IGNORECASE):
            label = clean_label(match.group(1))
            value_str = match.group(2).replace(',', '')
            scale = match.group(3).lower()
            
            try:
                value = float(value_str)
                # Convert to actual values
                if scale == 'trillion':
                    value *= 1_000_000_000_000
                elif scale == 'billion':
                    value *= 1_000_000_000
                elif scale == 'million':
                    value *= 1_000_000
                
                data_points.append({
                    'label': label,
                    'value': value,
                    'type': 'currency',
                    'unit': '$',
                    'display_scale': scale
                })
            except ValueError:
                continue
        
        # Pattern for regular currency values
        currency_pattern = r'(\w+(?:\s+\w+)*)\s*:?\s*\$?([\d,]+(?:\.\d+)?)\s*(?!trillion|billion|million)'
        for match in re.finditer(currency_pattern, text, re.IGNORECASE):
            label = clean_label(match.group(1))
            value_str = match.group(2).replace(',', '')
            try:
                value = float(value_str)
                # Skip if already captured with scale
                if not any(dp['label'].lower() == label.lower() for dp in data_points):
                    data_points.append({
                        'label': label,
                        'value': value,
                        'type': 'currency',
                        'unit': '$'
                    })
            except ValueError:
                continue
        
        # Pattern for general numbers with labels
        number_pattern = r'(\w+(?:\s+\w+)*)\s*:?\s*([\d,]+(?:\.\d+)?)'
        for match in re.finditer(number_pattern, text, re.IGNORECASE):
            label = clean_label(match.group(1))
            value_str = match.group(2).replace(',', '')
            try:
                value = float(value_str)
                # Skip if already captured as percentage or currency
                if not any(dp['label'].lower() == label.lower() for dp in data_points):
                    data_points.append({
                        'label': label,
                        'value': value,
                        'type': 'number',
                        'unit': ''
                    })
            except ValueError:
                continue
        
        return data_points
    
    def create_bar_chart(self, data_points: List[Dict[str, Any]], title: str = "Data Visualization") -> str:
        """
        Create a simple horizontal bar chart using text characters.
        
        Args:
            data_points: List of data points to chart
            title: Chart title
            
        Returns:
            Formatted chart as string
        """
        if not data_points:
            return ""
        
        # Sort by value for better visualization
        sorted_data = sorted(data_points, key=lambda x: x['value'], reverse=True)
        
        # Take top 10 items to avoid cluttering
        chart_data = sorted_data[:10]
        
        if not chart_data:
            return ""
        
        max_value = max(item['value'] for item in chart_data)
        max_label_length = max(len(item['label']) for item in chart_data)
        
        # Create table for the chart
        table = Table(title=title, show_header=False, box=None, padding=(0, 1))
        table.add_column("Label", style="cyan", width=max_label_length + 2)
        table.add_column("Bar", style="green")
        table.add_column("Value", style="yellow", justify="right")
        
        for item in chart_data:
            label = item['label']
            value = item['value']
            unit = item.get('unit', '')
            display_scale = item.get('display_scale', '')
            
            # Calculate bar length (max 30 characters)
            bar_length = int((value / max_value) * 30) if max_value > 0 else 0
            bar = "█" * bar_length + "░" * (30 - bar_length)
            
            # Format value
            if unit == '%':
                value_str = f"{value:.1f}%"
            elif unit == '$' and display_scale:
                # Convert back to original scale for display
                if display_scale == 'trillion':
                    display_value = value / 1_000_000_000_000
                    value_str = f"${display_value:.2f}T"
                elif display_scale == 'billion':
                    display_value = value / 1_000_000_000
                    value_str = f"${display_value:.2f}B"
                elif display_scale == 'million':
                    display_value = value / 1_000_000
                    value_str = f"${display_value:.2f}M"
                else:
                    value_str = f"${value:,.0f}"
            elif unit == '$':
                value_str = f"${value:,.0f}"
            else:
                value_str = f"{value:,.0f}"
            
            table.add_row(label, bar, value_str)
        
        # Capture the table output
        with self.console.capture() as capture:
            self.console.print(table)
        
        return capture.get()
    
    def create_comparison_table(self, data_points: List[Dict[str, Any]], title: str = "Comparison") -> str:
        """
        Create a comparison table for numeric data.
        
        Args:
            data_points: List of data points to display
            title: Table title
            
        Returns:
            Formatted table as string
        """
        if not data_points:
            return ""
        
        table = Table(title=title, show_header=True, header_style="bold magenta")
        table.add_column("Item", style="cyan")
        table.add_column("Value", style="yellow", justify="right")
        table.add_column("Type", style="green")
        
        for item in data_points:
            label = item['label']
            value = item['value']
            unit = item.get('unit', '')
            data_type = item.get('type', 'number')
            
            # Format value
            if unit == '%':
                value_str = f"{value:.1f}%"
            elif unit == '$' and item.get('display_scale', ''):
                # Convert back to original scale for display
                display_scale = item['display_scale']
                if display_scale == 'trillion':
                    display_value = value / 1_000_000_000_000
                    value_str = f"${display_value:.2f}T"
                elif display_scale == 'billion':
                    display_value = value / 1_000_000_000
                    value_str = f"${display_value:.2f}B"
                elif display_scale == 'million':
                    display_value = value / 1_000_000
                    value_str = f"${display_value:.2f}M"
                else:
                    value_str = f"${value:,.0f}"
            elif unit == '$':
                value_str = f"${value:,.0f}"
            else:
                value_str = f"{value:,.0f}"
            
            table.add_row(label, value_str, str(data_type).title())   

        # Capture the table output
        with self.console.capture() as capture:
            self.console.print(table)
        
        return capture.get()
    
    def generate_charts(self, answer: str, search_results) -> Optional[str]:
        """
        Generate appropriate charts/visualizations for the given content.
        
        Args:
            answer: LLM-generated answer
            search_results: Search results that informed the answer (can be SearchResult objects or dicts from cache)
            
        Returns:
            Formatted chart output or None if no charts generated
        """
        # Completely isolate chart generation to prevent any errors from affecting the main query
        try:
            # Check if we should generate charts
            should_generate = self.should_generate_chart(answer, search_results)
            if self.verbose:
                print(f"DEBUG: should_generate_chart = {should_generate}")
            if not should_generate:
                return None
            
            # Extract data only from answer text to avoid cache compatibility issues
            all_data = self.extract_numeric_data(answer)
            if self.verbose:
                print(f"DEBUG: extracted {len(all_data) if all_data else 0} data points")
                if all_data:
                    for i, item in enumerate(all_data[:3]):  # Show first 3 items
                        print(f"DEBUG: data[{i}] = {item}")
            
            if not all_data or len(all_data) < 2:
                if self.verbose:
                    print("DEBUG: Not enough data points for chart generation")
                return None
            
            # Filter and validate data
            valid_data = []
            for item in all_data:
                try:
                    if (isinstance(item, dict) and 
                        'label' in item and 
                        'value' in item and 
                        isinstance(item['label'], str) and 
                        len(item['label']) > 2 and 
                        float(item['value']) > 0):
                        valid_data.append(item)
                except:
                    continue
            
            if len(valid_data) < 2:
                return None
            
            # Create a simple chart using only the most reliable data
            try:
                # Sort by value for better visualization
                sorted_data = sorted(valid_data, key=lambda x: float(x['value']), reverse=True)[:5]
                
                # Create a simple table
                table = Table(title="📊 GDP Comparison", show_header=True, header_style="bold magenta")
                table.add_column("Country", style="cyan")
                table.add_column("GDP", style="yellow", justify="right")
                
                for item in sorted_data:
                    try:
                        label = str(item['label'])
                        value = float(item['value'])
                        unit = item.get('unit', '')
                        
                        # Format value safely
                        if unit == '$' and item.get('display_scale'):
                            scale = item['display_scale']
                            if scale == 'trillion':
                                display_value = value / 1_000_000_000_000
                                value_str = f"${display_value:.2f}T"
                            elif scale == 'billion':
                                display_value = value / 1_000_000_000
                                value_str = f"${display_value:.2f}B"
                            else:
                                value_str = f"${value:,.0f}"
                        elif unit == '$':
                            value_str = f"${value:,.0f}"
                        else:
                            value_str = f"{value:,.0f}"
                        
                        table.add_row(label, value_str)
                    except:
                        continue
                
                # Capture the table output
                with self.console.capture() as capture:
                    self.console.print(Panel(
                        table,
                        title="[bold blue]Visual Data Summary[/bold blue]",
                        border_style="blue",
                        padding=(1, 2)
                    ))
                return capture.get()
                
            except Exception as chart_error:
                # Even if chart creation fails, don't crash
                return None
            
        except Exception as e:
            # Completely silent failure - no chart generation errors should ever affect the main query
            return None
