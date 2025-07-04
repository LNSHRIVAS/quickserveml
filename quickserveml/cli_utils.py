# quickserveml/cli_utils.py

"""
CLI utilities for consistent, rich-formatted output across QuickServeML commands.
Provides colored output with fallback to plain text for environments without color support.
"""

import os
import sys
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.tree import Tree
from rich.markup import escape


class CLIFormatter:
    """
    Centralized CLI formatting utility with rich output and plain text fallback.
    """
    
    def __init__(self, force_terminal=None):
        """
        Initialize the CLI formatter.
        
        Args:
            force_terminal: Override terminal detection for testing
        """
        # Detect if we're in a terminal that supports colors
        self.supports_color = self._detect_color_support(force_terminal)
        
        # Initialize rich console with appropriate settings
        self.console = Console(
            force_terminal=self.supports_color,
            no_color=not self.supports_color,
            width=None,  # Auto-detect width
        )
    
    def _detect_color_support(self, force_terminal=None):
        """Detect if the current environment supports colored output."""
        if force_terminal is not None:
            return force_terminal
            
        # Check common environment variables that indicate no color support
        no_color_vars = ['NO_COLOR', 'TERM_PROGRAM']
        if any(os.getenv(var) for var in no_color_vars if os.getenv(var) == 'dumb'):
            return False
            
        # Check if we're in a CI environment
        ci_vars = ['CI', 'GITHUB_ACTIONS', 'GITLAB_CI', 'JENKINS_URL']
        if any(os.getenv(var) for var in ci_vars):
            return False
            
        # Check if stdout is a TTY
        return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()
    
    def success(self, message, title=None):
        """Display a success message."""
        if self.supports_color:
            if title:
                self.console.print(Panel(message, title=f"✅ {title}", style="green"))
            else:
                self.console.print(f"✅ {message}", style="green")
        else:
            prefix = f"[SUCCESS] {title}: " if title else "[SUCCESS] "
            self.console.print(f"{prefix}{message}")
    
    def info(self, message, title=None):
        """Display an info message."""
        if self.supports_color:
            if title:
                self.console.print(Panel(message, title=f"ℹ️  {title}", style="blue"))
            else:
                self.console.print(f"ℹ️  {message}", style="blue")
        else:
            prefix = f"[INFO] {title}: " if title else "[INFO] "
            self.console.print(f"{prefix}{message}")
    
    def warning(self, message, title=None):
        """Display a warning message."""
        if self.supports_color:
            if title:
                self.console.print(Panel(message, title=f"⚠️  {title}", style="yellow"))
            else:
                self.console.print(f"⚠️  {message}", style="yellow")
        else:
            prefix = f"[WARNING] {title}: " if title else "[WARNING] "
            self.console.print(f"{prefix}{message}")
    
    def error(self, message, title=None):
        """Display an error message."""
        if self.supports_color:
            if title:
                self.console.print(Panel(message, title=f"❌ {title}", style="red"))
            else:
                self.console.print(f"❌ {message}", style="red")
        else:
            prefix = f"[ERROR] {title}: " if title else "[ERROR] "
            self.console.print(f"{prefix}{message}")
    
    def section_header(self, title, subtitle=None):
        """Display a section header."""
        if self.supports_color:
            if subtitle:
                self.console.print(f"\n[bold blue]{title}[/bold blue]")
                self.console.print(f"[dim]{subtitle}[/dim]\n")
            else:
                self.console.print(f"\n[bold blue]{title}[/bold blue]\n")
        else:
            self.console.print(f"\n=== {title} ===")
            if subtitle:
                self.console.print(f"{subtitle}")
            self.console.print()
    
    def create_table(self, title=None, show_header=True, header_style="bold magenta"):
        """Create a rich table for structured data display."""
        if self.supports_color:
            return Table(title=title, show_header=show_header, header_style=header_style)
        else:
            # For plain text, we'll just return a simple table structure
            return PlainTable(title=title, show_header=show_header)
    
    def print_table(self, table):
        """Print a table (rich or plain)."""
        if self.supports_color and hasattr(table, '_columns'):
            self.console.print(table)
        elif hasattr(table, 'print_plain'):
            # Handle plain table
            table.print_plain(self.console)
        else:
            # It's a rich table, print it normally
            self.console.print(table)
    
    def create_progress(self, description="Processing..."):
        """Create a progress bar for long-running operations."""
        if self.supports_color:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
        else:
            return PlainProgress(description, self.console)
    
    def progress_bar(self, total, description="Processing..."):
        """Create a progress bar with total count and description."""
        if self.supports_color:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=self.console,
            )
            task_id = progress.add_task(description, total=total)
            return ProgressBar(progress, task_id)
        else:
            return PlainProgressBar(description, total, self.console)
    
    def print_json(self, data, title=None):
        """Print JSON data with nice formatting."""
        if self.supports_color:
            if title:
                self.console.print(f"\n[bold]{title}[/bold]")
            self.console.print_json(data=data)
        else:
            if title:
                self.console.print(f"\n{title}:")
            import json
            self.console.print(json.dumps(data, indent=2))
    
    def create_tree(self, label):
        """Create a tree structure for hierarchical data."""
        if self.supports_color:
            return Tree(label)
        else:
            return PlainTree(label)
    
    def print_tree(self, tree):
        """Print a tree structure."""
        if self.supports_color and hasattr(tree, '_rich_console'):
            self.console.print(tree)
        else:
            tree.print_plain(self.console)
    
    def print_columns(self, items, title=None):
        """Print items in columns."""
        if self.supports_color:
            if title:
                self.console.print(f"[bold]{title}[/bold]")
            columns = Columns(items, equal=True, expand=True)
            self.console.print(columns)
        else:
            if title:
                self.console.print(f"{title}:")
            for item in items:
                self.console.print(f"  • {item}")


class PlainTable:
    """Simple table implementation for environments without rich support."""
    
    def __init__(self, title=None, show_header=True):
        self.title = title
        self.show_header = show_header
        self.columns = []
        self.rows = []
        self.headers = []
    
    def add_column(self, header, style=None, justify="left"):
        """Add a column to the table."""
        self.headers.append(header)
        self.columns.append({"header": header, "style": style, "justify": justify})
    
    def add_row(self, *values):
        """Add a row to the table."""
        self.rows.append(list(values))
    
    def print_plain(self, console):
        """Print the table in plain text format."""
        if self.title:
            console.print(f"\n{self.title}")
            console.print("=" * len(self.title))
        
        if self.show_header and self.headers:
            header_line = " | ".join(str(h) for h in self.headers)
            console.print(header_line)
            console.print("-" * len(header_line))
        
        for row in self.rows:
            row_line = " | ".join(str(cell) for cell in row)
            console.print(row_line)


class PlainProgress:
    """Simple progress indicator for environments without rich support."""
    
    def __init__(self, description, console):
        self.description = description
        self.console = console
        self.current = 0
        self.total = 0
    
    def add_task(self, description, total=100):
        """Add a task to track."""
        self.total = total
        self.console.print(f"{description} (0/{total})")
        return "task_id"
    
    def update(self, task_id, advance=1, description=None):
        """Update progress."""
        self.current += advance
        if description:
            self.console.print(f"{description} ({self.current}/{self.total})")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.total > 0:
            self.console.print(f"✓ {self.description} completed ({self.current}/{self.total})")


class PlainTree:
    """Simple tree structure for environments without rich support."""
    
    def __init__(self, label):
        self.label = label
        self.children = []
        self.level = 0
    
    def add(self, label):
        """Add a child node."""
        child = PlainTree(label)
        child.level = self.level + 1
        self.children.append(child)
        return child
    
    def print_plain(self, console, prefix=""):
        """Print the tree in plain text format."""
        indent = "  " * self.level
        console.print(f"{indent}{prefix}{self.label}")
        
        for i, child in enumerate(self.children):
            is_last = i == len(self.children) - 1
            child_prefix = "└── " if is_last else "├── "
            child.print_plain(console, child_prefix)


# Global formatter instance
_formatter = None

def get_formatter():
    """Get the global CLI formatter instance."""
    global _formatter
    if _formatter is None:
        _formatter = CLIFormatter()
    return _formatter

# Convenience functions
def success(message, title=None):
    """Display a success message."""
    get_formatter().success(message, title)

def info(message, title=None):
    """Display an info message."""
    get_formatter().info(message, title)

def warning(message, title=None):
    """Display a warning message."""
    get_formatter().warning(message, title)

def error(message, title=None):
    """Display an error message."""
    get_formatter().error(message, title)

def section_header(title, subtitle=None):
    """Display a section header."""
    return get_formatter().section_header(title, subtitle)


class ProgressBar:
    """Rich progress bar wrapper for context manager usage."""
    
    def __init__(self, progress, task_id):
        self.progress = progress
        self.task_id = task_id
    
    def __enter__(self):
        self.progress.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress.stop()
    
    def advance(self, amount=1):
        """Advance the progress bar by the specified amount."""
        self.progress.update(self.task_id, advance=amount)


class PlainProgressBar:
    """Plain text progress bar for environments without rich support."""
    
    def __init__(self, description, total, console):
        self.description = description
        self.total = total
        self.console = console
        self.current = 0
    
    def __enter__(self):
        self.console.print(f"{self.description} (0/{self.total})")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.console.print(f"{self.description} completed ({self.total}/{self.total})")
    
    def advance(self, amount=1):
        """Advance the progress bar by the specified amount."""
        self.current += amount
        if self.current % max(1, self.total // 10) == 0:  # Update every 10%
            self.console.print(f"{self.description} ({self.current}/{self.total})")
