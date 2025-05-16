# Datalore 📊🔍

Datalore is an AI-powered Data Analysis tool that integrates Anthropic's Claude API with various data analysis libraries and custom functions. It provides an interactive interface for users to perform data analysis tasks using natural language commands.

## ✨ Features

- 🗣️ Natural language interaction for data analysis tasks
- 🧠 Integration with Anthropic's Claude API for advanced language processing
- 📁 Data loading from various file formats (CSV, Excel, JSON)
- 🧹 Data preprocessing and cleaning
- 🔬 Exploratory Data Analysis (EDA)
- 📈 Statistical analysis
- 📊 Data visualization
- 🐍 Custom Python code execution for advanced operations
- 💬 Conversation history management
- 🎨 Colorized terminal output for enhanced readability

## 📋 Requirements

- Python 3.7+
- Anthropic API key

## 🚀 Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/datalore.git
   cd datalore
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your Anthropic API key:
   - Create a `.env` file in the project root
   - Add your API key: `ANTHROPIC_API_KEY=your_api_key_here`

## 🎮 Usage

Run the main script:

```
python datalore.py
```

Follow the prompts to interact with Claude, the AI data analyst. You can ask questions, request data analysis tasks, and even execute custom Python code.

Example commands:
- "Load the sales_data.csv file"
- "Show me a summary of the data"
- "Create a scatter plot of price vs. quantity"
- "Run a linear regression on the data"

## 💻 Custom Code Execution

You can execute custom Python code using the `execute_code` tool. This allows for more complex operations and data manipulations. The code is executed in a sandboxed environment for safety.

Example:
```python
# Assuming 'current_df' is already loaded with your data
current_df = current_df.dropna()  # Remove rows with missing values
current_df['new_column'] = current_df['existing_column'] * 2  # Create a new column
current_df = current_df[current_df['some_column'] > 0]  # Filter rows
```

## 🛡️ Safety and Limitations

- The tool includes safety checks for code execution to prevent malicious operations.
- Large datasets may impact performance. Consider using sample data for initial analysis.
- The tool relies on the Anthropic API, so an internet connection is required.

## 🤝 Contributing

Contributions to Datalore are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Anthropic for the Claude API
- The open-source community for the various data analysis libraries used in this project
- Test
