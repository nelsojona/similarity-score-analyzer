To activate the virtual environment created during the setup process, follow these platform-specific instructions:

### **Activating the Virtual Environment:**

- **On macOS/Linux:**

  ```bash
  source venv/bin/activate
  ```

- **On Windows:**

  ```bash
  venv\Scripts\activate
  ```

Once the virtual environment is activated, your terminal prompt should change to indicate that you are now working within the virtual environment. You can then proceed with the installation of dependencies and running the application.

### **Updated README Instructions:**

# Similarity Score Analyzer

Similarity Score Analyzer is a Python application that analyzes the similarity of webpage content to a given query. The application leverages modern web scraping techniques, natural language processing, and machine learning models to provide detailed similarity analysis and optimization suggestions.

## Features

- **Web Scraping**: Extracts webpage content using the `crawlee` library.
- **Text Preprocessing**: Cleans and preprocesses text data for analysis.
- **Embedding Generation**: Generates text embeddings using TensorFlow Hub models.
- **Similarity Scoring**: Computes cosine similarity scores between query and webpage sections.
- **Google Cloud NLP Integration**: Analyzes sentiment and entity recognition using Google Cloud Natural Language API.
- **Heatmap Visualization**: Displays similarity scores in a heatmap for easy visualization.
- **Optimization Suggestions**: Provides suggestions to improve content relevance based on similarity scores.

## Installation

To install and set up the Similarity Score Analyzer, follow these steps:

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- A Google Cloud account with access to the Natural Language API

### Setup Instructions

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/similarity-score-analyzer.git
   cd similarity-score-analyzer
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python3 -m venv venv
   ```

3. **Activate the Virtual Environment:**

   - On **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

   - On **Windows**:
     ```bash
     venv\Scripts\activate
     ```

4. **Install the Package and Dependencies:**

   ```bash
   pip install .
   ```

   This will install the Similarity Score Analyzer package along with all required dependencies listed in `setup.py`.

5. **Set Up Environment Variables:**

   Create a `.env` file in the root directory of the project and add your Google Cloud NLP API key:

   ```bash
   echo "GOOGLE_CLOUD_NLP_API_KEY=your_actual_api_key_here" > .env
   ```

6. **Run the Application:**

   Start the application using Streamlit:

   ```bash
   streamlit run similarity_analyzer/main.py
   ```

## Usage

Once the application is running, you can use the web interface to:

1. **Enter the Target URL**: Provide the URL of the webpage you want to analyze.
2. **Enter the Query**: Input the query you want to optimize the webpage content for.
3. **Select an Embedding Model**: Choose from available embedding models like Universal Sentence Encoder.
4. **Analyze**: Click the "Analyze" button to start the analysis.
5. **View Results**: The application will display the overall similarity score, section-wise heatmap, optimization suggestions, and Google Cloud NLP analysis (sentiment and entities).

## Contributing

Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

### Running Tests

To run the unit tests included with the project:

```bash
python -m unittest discover tests
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contact

For any inquiries or feedback, please open an issue.
