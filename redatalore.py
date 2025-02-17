import os
from datetime import datetime
import json
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.traceback import install
from rich.prompt import Prompt
from rich.markdown import Markdown
from rich.text import Text
from rich.live import Live
from rich.layout import Layout
from rich import print as rprint
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import TerminalFormatter
import pygments.util
from dotenv import load_dotenv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO, StringIO
import base64
from sklearn.linear_model import LinearRegression
import numpy as np
import traceback
import ast
import sys
from contextlib import redirect_stdout, redirect_stderr
import threading
import _thread
import time

# Use OpenRouter's OpenAI client instead of Anthropic.
from openai import OpenAI

load_dotenv()
install()  # Install rich traceback handler

console = Console()

# Create the OpenAI client with OpenRouter's API endpoint and your API key.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
#openai/gpt-4o-2024-11-20
#anthropic/claude-3-5-haiku
conversation_history = []
current_df = None
figure_counter = 0


system_prompt = """
You are Claude, an AI data analyst for Datalore, integrated with a data analysis system. Your capabilities include:
1. Reading and displaying various data file formats (CSV, Excel, JSON)
2. Data preprocessing and cleaning
3. Exploratory Data Analysis (EDA)
4. Statistical analysis
5. Data visualization
6. Machine learning model building and evaluation
7. Executing custom Python code

When interacting with the user:
- Help them analyze their data efficiently
- Use available tools to perform data analysis tasks when needed
- Provide clear, accurate and detailed responses

If you are unsure, ask for clarification.
"""

WELCOME_ART = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║                      🔮 DATALORE 🔮                         ║
║                                                              ║
║              Your AI-powered Data Analysis Tool              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""

def print_colored(text, style=""):
    """Enhanced print function using rich"""
    console.print(text, style=style)

def print_code(code, language):
    """Enhanced code printing with syntax highlighting"""
    syntax = Syntax(code, language, theme="monokai", line_numbers=True)
    console.print(syntax)

def display_dataframe(df, title="DataFrame Preview"):
    """Enhanced DataFrame display using rich tables"""
    table = Table(title=title, show_header=True, header_style="bold magenta")
    
    # Add columns
    for column in df.columns:
        table.add_column(str(column), style="cyan")
    
    # Add rows
    for _, row in df.head().iterrows():
        table.add_row(*[str(val) for val in row])
    
    console.print(table)

def show_progress(description="Processing"):
    """Create a progress context for long operations"""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True
    )

def read_data(file_path, file_type):
    global current_df
    with show_progress(f"Reading {file_type} file") as progress:
        task = progress.add_task(description="Reading...", total=None)
        try:
            if file_type == "csv":
                current_df = pd.read_csv(file_path)
            elif file_type == "excel":
                current_df = pd.read_excel(file_path)
            elif file_type == "json":
                current_df = pd.read_json(file_path)
            else:
                return "Unsupported file type"
            progress.update(task, completed=True)
            display_dataframe(current_df, f"Data from {file_path}")
            return f"Data read successfully. Shape: {current_df.shape}"
        except Exception as e:
            console.print(f"[red]Error reading file:[/red] {str(e)}")
            return f"Error reading file: {str(e)}"

def preprocess_data(operations):
    global current_df
    if current_df is None:
        return "No data loaded. Please read a data file first."
    try:
        for operation in operations:
            if operation == "drop_na":
                current_df = current_df.dropna()
            elif operation == "fill_na_mean":
                current_df = current_df.fillna(current_df.mean())
            elif operation == "normalize":
                current_df = (current_df - current_df.mean()) / current_df.std()
        return f"Preprocessing completed. New shape: {current_df.shape}"
    except Exception as e:
        return f"Error during preprocessing: {str(e)}"

def analyze_data(analysis_type):
    global current_df
    if current_df is None:
        return "No data loaded. Please read a data file first."
    try:
        if analysis_type == "summary":
            return current_df.describe().to_string()
        elif analysis_type == "correlation":
            return current_df.corr().to_string()
        elif analysis_type == "regression":
            X = current_df.iloc[:, :-1]
            y = current_df.iloc[:, -1]
            model = LinearRegression().fit(X, y)
            return f"Regression coefficients: {model.coef_}"
        else:
            return f"Unsupported analysis type: {analysis_type}"
    except Exception as e:
        return f"Error during analysis: {str(e)}"

def visualize_data(plot_type, x_column, y_column=None):
    global current_df, figure_counter
    if current_df is None:
        return "No data loaded. Please read a data file first."
    try:
        # Clear any existing plots
        plt.clf()
        plt.close('all')
        
        # Create new figure
        plt.figure(figsize=(10, 6))
        
        if plot_type == "scatter":
            sns.scatterplot(data=current_df, x=x_column, y=y_column)
        elif plot_type == "bar":
            sns.barplot(data=current_df, x=x_column, y=y_column)
        elif plot_type == "histogram":
            sns.histplot(data=current_df, x=x_column)
        elif plot_type == "line":
            sns.lineplot(data=current_df, x=x_column, y=y_column)
        
        plt.title(f"{plot_type.capitalize()} plot")
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_counter += 1
        filename = f"plot_{timestamp}_{figure_counter}.png"
        
        # Save and close
        plt.savefig(filename)
        plt.close('all')
        
        return f"Visualization saved as {filename}"
    except Exception as e:
        plt.close('all')  # Ensure cleanup on error
        return f"Error during visualization: {str(e)}"

def execute_code(code, timeout=30, max_output_length=10000):
    global current_df, figure_counter
    
    def analyze_code_safety(code):
        """Analyze the code for potentially unsafe operations."""
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    if any(name.name == 'os' for name in node.names):
                        return False, "Importing 'os' module is not allowed for security reasons."
                if isinstance(node, (ast.Call, ast.Attribute)):
                    func_name = ''
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node, ast.Attribute):
                        func_name = node.attr
                    if func_name in ['eval', 'exec', 'compile']:
                        return False, f"Use of '{func_name}' is not allowed for security reasons."
            return True, "Code analysis passed."
        except SyntaxError as e:
            return False, f"Syntax error in code: {str(e)}"

    def run_code_in_namespace(code, global_ns, local_ns):
        """Execute the code in a specific namespace and capture its output."""
        output_buffer = StringIO()
        error_buffer = StringIO()
        
        with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
            exec(code, global_ns, local_ns)
        
        return output_buffer.getvalue(), error_buffer.getvalue()

    def execute_with_timeout(code, global_ns, local_ns, timeout):
        """Execute the code with a timeout."""
        result = {"output": "", "error": "", "timed_out": False}
        
        def target():
            try:
                result["output"], result["error"] = run_code_in_namespace(code, global_ns, local_ns)
            except Exception as e:
                result["error"] = f"Error: {str(e)}\n{traceback.format_exc()}"

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            _thread.interrupt_main()
            thread.join()
            result["timed_out"] = True
            result["error"] = "Execution timed out"
        
        return result

    # Clear any existing plots before execution
    plt.clf()
    plt.close('all')

    # Step 1: Analyze code safety
    is_safe, safety_message = analyze_code_safety(code)
    if not is_safe:
        return {"output": "", "error": safety_message, "variables": {}}

    # Step 2: Prepare the execution environment
    global_ns = {
        '__builtins__': __builtins__,
        'pd': pd,
        'np': np,
        'plt': plt,
        'sns': sns,
        'datetime': datetime,
    }
    local_ns = {'current_df': current_df}

    # Step 3: Replace 'df' with 'current_df' in the code
    modified_code = code.replace('df', 'current_df')

    # Step 4: Execute the modified code with timeout
    result = execute_with_timeout(modified_code, global_ns, local_ns, timeout)

    # Step 5: Process the results
    output = result["output"]
    error = result["error"]

    if len(output) > max_output_length:
        output = output[:max_output_length] + "\n... (output truncated)"
    
    if 'current_df' in local_ns and not local_ns['current_df'].equals(current_df):
        current_df = local_ns['current_df']
        output += f"\n\nDataFrame modified. New shape: {current_df.shape}"
        output += f"\nFirst few rows:\n{current_df.head().to_string()}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"preprocessed_data_{timestamp}.csv"
        current_df.to_csv(csv_filename, index=False)
        output += f"\n\nPreprocessed data saved as: {csv_filename}"
    
    created_vars = {k: v for k, v in local_ns.items() if k not in global_ns and not k.startswith('_')}
    
    if plt.get_fignums():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_counter += 1
        filename = f"plot_{timestamp}_{figure_counter}.png"
        try:
            plt.savefig(filename)
            output += f"\nPlot saved as {filename}"
        except Exception as e:
            output += f"\nPlot saving error: {str(e)}"
        finally:
            plt.close('all')  # Ensure all figures are closed

    return {
        "output": output,
        "error": error,
        "variables": created_vars,
        "timed_out": result["timed_out"]
    }

# Update the tools definitions to match OpenRouter's function calling schema.
tools = [
    {
        "type": "function",
        "function": {
            "name": "read_data",
            "description": "Read a data file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to read"
                    },
                    "file_type": {
                        "type": "string",
                        "description": "The type of the file (csv, excel, json)"
                    }
                },
                "required": ["file_path", "file_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "preprocess_data",
            "description": "Preprocess and clean the data",
            "parameters": {
                "type": "object",
                "properties": {
                    "operations": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of preprocessing operations to perform"
                    }
                },
                "required": ["operations"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_data",
            "description": "Perform statistical analysis on the data",
            "parameters": {
                "type": "object",
                "properties": {
                    "analysis_type": {
                        "type": "string",
                        "description": "Type of analysis to perform (e.g., 'summary', 'correlation', 'regression')"
                    }
                },
                "required": ["analysis_type"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visualize_data",
            "description": "Create data visualizations",
            "parameters": {
                "type": "object",
                "properties": {
                    "plot_type": {
                        "type": "string",
                        "description": "Type of plot to create (e.g., 'scatter', 'bar', 'histogram', 'line')"
                    },
                    "x_column": {
                        "type": "string",
                        "description": "Column to use for x-axis"
                    },
                    "y_column": {
                        "type": "string",
                        "description": "Column to use for y-axis (if applicable)"
                    }
                },
                "required": ["plot_type", "x_column"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute custom Python code",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute"
                    }
                },
                "required": ["code"]
            }
        }
    }
]

def execute_tool(tool_name, tool_input):
    if tool_name == "read_data":
        return read_data(**tool_input)
    elif tool_name == "preprocess_data":
        return preprocess_data(**tool_input)
    elif tool_name == "analyze_data":
        return analyze_data(**tool_input)
    elif tool_name == "visualize_data":
        return visualize_data(**tool_input)
    elif tool_name == "execute_code":
        result = execute_code(**tool_input)
        return f"Output: {result['output']}\nError: {result['error']}\nVariables: {result['variables']}\nTimed out: {result['timed_out']}"
    else:
        return f"Unknown tool: {tool_name}"

def chat_with_claude(user_input):
    global conversation_history
    conversation_history.append({"role": "user", "content": user_input})
    if not any(msg.get("role") == "system" for msg in conversation_history):
        conversation_history.insert(0, {"role": "system", "content": system_prompt})
    
    try:
        response = client.chat.completions.create(
            model="openai/o3-mini-high",
            messages=conversation_history,
            tools=tools,
            tool_choice="auto"
        )
    except Exception as e:
        console.print(f"[red]API request error:[/red] {str(e)}")
        return f"Error: {str(e)}"
    
    if not response or not hasattr(response, "choices") or not response.choices:
        console.print("[red]Error:[/red] Received no response from the API.")
        return "Error: Received no response from the API."
    
    assistant_message = response.choices[0].message
    
    if assistant_message.tool_calls:
        tool_results = []
        for tool_call in assistant_message.tool_calls:
            tool_name = tool_call.function.name
            arguments_str = tool_call.function.arguments
            tool_call_id = tool_call.id
            
            console.print(f"\n[yellow]Tool Used:[/yellow] {tool_name}")
            console.print(f"[yellow]Tool Input:[/yellow] {arguments_str}")
            
            try:
                tool_input = json.loads(arguments_str)
            except Exception as e:
                tool_input = arguments_str
            
            result = execute_tool(tool_name, tool_input)
            tool_results.append(result)
            
            console.print(f"[green]Tool Result:[/green]")
            if isinstance(result, pd.DataFrame):
                display_dataframe(result)
            else:
                console.print(result)
            
            conversation_history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": tool_call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": arguments_str
                    }
                }]
            })
            conversation_history.append({
                "role": "tool",
                "name": tool_name,
                "tool_call_id": tool_call_id,
                "content": str(result)  # Convert result to string to ensure it's serializable
            })
        
        # Get a follow-up response after tool execution
        try:
            follow_up = client.chat.completions.create(
                model="openai/o3-mini-high",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"I've analyzed the data and here are the results: {', '.join(str(r) for r in tool_results)}. Let me explain what this means."}
                ]
            )
            
            if follow_up and follow_up.choices and follow_up.choices[0].message:
                follow_up_message = follow_up.choices[0].message.content
                console.print(f"\n[blue]Analysis:[/blue] {follow_up_message}")
                conversation_history.append({"role": "assistant", "content": follow_up_message})
                return follow_up_message
            else:
                console.print("[red]No follow-up analysis available[/red]")
                return tool_results[-1]
        except Exception as e:
            console.print(f"[red]Error getting follow-up analysis:[/red] {str(e)}")
            return tool_results[-1]  # Return the last tool result if follow-up fails
    else:
        assistant_response = assistant_message.content or ""
        console.print(f"\n[blue]Claude:[/blue] {assistant_response}")
        conversation_history.append({"role": "assistant", "content": assistant_response})
        return assistant_response

def main():
    console.print(Panel.fit(WELCOME_ART, border_style="blue"))
    console.print("\n[bold blue]Welcome to your AI-powered Data Analyst![/bold blue]")
    console.print("[cyan]I can help you analyze data from various file formats.[/cyan]")
    console.print("[green]Just chat naturally about what you'd like to do![/green]")
    console.print("[yellow]Type 'exit' to end the conversation.[/yellow]\n")
    
    while True:
        user_input = Prompt.ask("[bold blue]You")
        if user_input.lower() == 'exit':
            console.print("\n[bold green]Thank you for using the AI Data Analyst. Goodbye![/bold green]")
            break
        
        with console.status("[bold blue]Processing...") as status:
            response = chat_with_claude(user_input)
            
            # Only try to process code blocks if response is a string and contains code blocks
            if isinstance(response, str) and "```" in response:
                parts = response.split("```")
                for i, part in enumerate(parts):
                    if i % 2 == 0:
                        console.print(Markdown(part))
                    else:
                        lines = part.split('\n')
                        language = lines[0].strip() if lines else ""
                        code = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                        
                        if language and code:
                            print_code(code, language)
                        elif code:
                            print_code(code, "python")
                        else:
                            console.print(part)

if __name__ == "__main__":
    main()
