import os
from datetime import datetime
import json
from colorama import init, Fore, Style
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
init()

USER_COLOR = Fore.WHITE
CLAUDE_COLOR = Fore.BLUE
TOOL_COLOR = Fore.YELLOW
RESULT_COLOR = Fore.GREEN

# Create the OpenAI client with OpenRouter's API endpoint and your API key.
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

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

def print_colored(text, color):
    print(f"{color}{text}{Style.RESET_ALL}")

def print_code(code, language):
    try:
        lexer = get_lexer_by_name(language, stripall=True)
        formatted_code = highlight(code, lexer, TerminalFormatter())
        print(formatted_code)
    except pygments.util.ClassNotFound:
        print_colored(f"Code (language: {language}):\n{code}", CLAUDE_COLOR)

def read_data(file_path, file_type):
    global current_df
    try:
        if file_type == "csv":
            current_df = pd.read_csv(file_path)
        elif file_type == "excel":
            current_df = pd.read_excel(file_path)
        elif file_type == "json":
            current_df = pd.read_json(file_path)
        else:
            return "Unsupported file type"
        return f"Data read successfully. Shape: {current_df.shape}\n\nFirst few rows:\n{current_df.head().to_string()}"
    except Exception as e:
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
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        figure_counter += 1
        filename = f"plot_{timestamp}_{figure_counter}.png"
        
        plt.savefig(filename)
        plt.close()
        
        return f"Visualization saved as {filename}"
    except Exception as e:
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
            plt.close()

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
    
    # Extract only function definitions from our tools
    functions_list = [tool["function"] for tool in tools if "function" in tool]

    while True:
        try:
            response = client.chat.completions.create(
                model="openai/gpt-4o-2024-11-20",
                messages=conversation_history,
                tools=tools,              # Should use tools parameter
                tool_choice="auto"        # Should use tool_choice parameter
            )
        except Exception as e:
            print_colored(f"API request error: {e}", CLAUDE_COLOR)
            return f"Error: {str(e)}"
        
        if not response or not hasattr(response, "choices") or not response.choices:
            print_colored("Error: Received no response from the API.", CLAUDE_COLOR)
            return "Error: Received no response from the API."
        
        assistant_message = response.choices[0].message
        
        # Fix: Check the tool_calls attribute instead of using dict.get() 
        if assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                tool_name = tool_call.function.name
                arguments_str = tool_call.function.arguments
                tool_call_id = tool_call.id
                
                print_colored(f"\nTool Used: {tool_name}", TOOL_COLOR)
                print_colored(f"Tool Input: {arguments_str}", TOOL_COLOR)
                try:
                    tool_input = json.loads(arguments_str)
                except Exception as e:
                    tool_input = arguments_str  # fallback if not valid JSON
                
                result = execute_tool(tool_name, tool_input)
                print_colored(f"Tool Result: {result}", RESULT_COLOR)
                
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
                    "content": result
                })
        else:
            assistant_response = assistant_message.content or ""
            print_colored(f"\nClaude: {assistant_response}", CLAUDE_COLOR)
            conversation_history.append({"role": "assistant", "content": assistant_response})
            break
    return assistant_response

def main():
    print_colored("Welcome to your AI-powered Data Analyst!\n", CLAUDE_COLOR)
    print_colored("I am Claude, and I can help you analyze data from various file formats.", CLAUDE_COLOR)
    print_colored("Just chat with me naturally about what you'd like to do and I'll do my best to assist you.", CLAUDE_COLOR)
    print_colored("You can ask me to read data files, preprocess data, perform statistical analysis, visualize data, and more.", CLAUDE_COLOR)
    print_colored("Type 'exit' to end the conversation.", CLAUDE_COLOR)
    
    while True:
        user_input = input(f"\n{USER_COLOR}You: {Style.RESET_ALL}")
        if user_input.lower() == 'exit':
            print_colored("Thank you for using the AI Data Analyst. Goodbye!", CLAUDE_COLOR)
            break
        
        response = chat_with_claude(user_input)
        
        if "```" in response:
            parts = response.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    print_colored(part, CLAUDE_COLOR)
                else:
                    lines = part.split('\n')
                    language = lines[0].strip() if lines else ""
                    code = '\n'.join(lines[1:]) if len(lines) > 1 else ""
                    
                    if language and code:
                        print_code(code, language)
                    elif code:
                        print_colored(f"Code:\n{code}", CLAUDE_COLOR)
                    else:
                        print_colored(part, CLAUDE_COLOR)

if __name__ == "__main__":
    main()
