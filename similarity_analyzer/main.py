import os

# Disable gRPC fork support
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "0"

import asyncio
import streamlit as st
import logging
from web_scraper import scrape_webpage
from text_preprocessor import preprocess_text
from embedding_generator import generate_embeddings, load_model, EMBEDDING_MODELS
from similarity_scorer import calculate_similarity
from heatmap_generator import generate_heatmap
from google_nlp import analyze_all_sections
from similarity_analyzer.config import CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_optimization_suggestions(sections: list, scores: list, query: str) -> list:
    """
    Generates optimization suggestions for webpage sections based on similarity scores.

    Args:
        sections (list): A list of text sections from the webpage.
        scores (list): A list of similarity scores for each section.
        query (str): The query term that the webpage sections are being optimized for.

    Returns:
        list: A list of suggestions for improving the relevance of sections with low similarity scores.
    """
    suggestions = []
    for i, (section, score) in enumerate(zip(sections, scores)):
        if score < 5:
            missing_keywords = [word for word in query.split() if word not in section.lower()]
            suggestions.append(f"Section {i+1} (Score: {score:.2f}):")
            suggestions.append(f"  - Consider adding these keywords: {', '.join(missing_keywords)}")
            suggestions.append(f"  - Expand on topics related to: {query}")
    return suggestions

async def analyze_webpage(url: str, query: str, model_name: str):
    """
    Asynchronous function to perform all analysis operations.

    Args:
        url (str): The URL of the webpage to analyze.
        query (str): The query to optimize for.
        model_name (str): The name of the embedding model to use.

    Returns:
        tuple: Processed sections, similarity scores, sentiments, and entities.
    """
    logger.info(f"Analyzing URL: {url} with query: {query}")

    # Scraping the webpage content
    webpage_data = scrape_webpage(url)
    if not webpage_data:
        return None, None, None, None

    sections = webpage_data["sections"]

    # Preprocess the text sections and the query
    processed_sections = [preprocess_text(section) for section in sections]
    processed_query = preprocess_text(query)

    # Load the selected embedding model
    model = load_model(model_name)
    if not model:
        return None, None, None, None

    # Generate embeddings for the sections and the query
    section_embeddings = generate_embeddings(model, processed_sections)
    query_embedding = generate_embeddings(model, [processed_query])

    # Calculate similarity scores
    scores = calculate_similarity(query_embedding, section_embeddings)

    # Perform sentiment and entity analysis
    sentiments, entities = await analyze_all_sections(sections)

    return sections, scores, sentiments, entities

def main():
    """
    Main function to run the Similarity Score Analyzer Streamlit app.
    """
    st.title("Similarity Score Analyzer")

    # Input fields for the target URL and the query term
    url = st.text_input("Enter Target URL")
    query = st.text_input("Enter Query to Optimize For")

    # Dropdown to select the embedding model
    model_options = list(EMBEDDING_MODELS.keys())
    selected_model = st.selectbox("Choose Embedding Model", model_options)

    if st.button("Analyze"):
        if url and query:
            with st.spinner('Analyzing webpage...'):
                if selected_model == "Gemini Text Embedding":
                    st.info("Using Gemini API for embedding generation. This may take a moment...")
                
                sections, scores, sentiments, entities = asyncio.run(analyze_webpage(url, query, selected_model))

                if sections is None:
                    st.error("Failed to process the webpage. Please check the URL or embedding model.")
                    return

                # Display the overall similarity score
                st.subheader("Overall Similarity Score (Average):")
                st.write(sum(scores) / len(scores))

                # Display a heatmap of similarity scores
                st.subheader("Section Similarity Scores Heatmap:")
                st.plotly_chart(generate_heatmap(scores, [f"Section {i+1}" for i in range(len(scores))]))

                # Generate and display optimization suggestions
                suggestions = generate_optimization_suggestions(sections, scores, query)
                st.subheader("Optimization Suggestions:")
                for suggestion in suggestions:
                    st.write(suggestion)

                # Perform and display sentiment and entity analysis
                st.subheader("Google Cloud Natural Language API Analysis:")
                for i, (section, sentiment, section_entities) in enumerate(zip(sections, sentiments, entities)):
                    st.write(f"**Section {i+1}:**")
                    if sentiment:
                        st.write(f"Sentiment: {sentiment.score:.2f} ({sentiment.magnitude:.2f})")
                    else:
                        st.write("Sentiment analysis failed.")
                    st.write("Entities:")
                    for entity in section_entities:
                        st.write(f"  - {entity.name}: {entity.type_}")

if __name__ == "__main__":
    main()