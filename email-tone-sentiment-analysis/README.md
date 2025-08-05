# Email Tone and Sentiment Analysis

## Project Overview

This project implements a comprehensive email analysis system that combines multiple AI-powered tools to analyze emails for sentiment, tone, and potential grammatical issues. The system provides dynamic tone feedback, visualizes sentiment trends, and leverages various pre-trained models to deliver insights for professional communication.

## Features

### Core Analysis Capabilities

- **Sentiment Analysis**: Multi-model sentiment detection using VADER, TextBlob, and transformer-based models
- **Emotion Detection**: Identifies emotions like joy, sadness, anger, fear, surprise, and disgust
- **Tone Analysis**: Evaluates professional tone, formality, and communication style
- **Grammar & Spelling**: Advanced grammar checking and spelling correction
- **Readability Assessment**: Computes Flesch Readability Ease Score for text complexity
- **Sarcasm Detection**: Identifies sarcastic content that might be misinterpreted

### Visualization & Reporting

- **Sentiment Trends**: Time-series visualization of sentiment changes
- **Emotion Distribution**: Pie charts and bar graphs showing emotion breakdown
- **Tone Feedback**: Real-time suggestions for improving professional tone
- **Grammar Highlights**: Visual indicators for grammatical issues and suggestions

## Technologies Used

| Technology                      | Purpose                                                          |
| ------------------------------- | ---------------------------------------------------------------- |
| **Transformers (Hugging Face)** | Pre-trained models for sentiment, emotion, and sarcasm detection |
| **NLTK**                        | VADER lexicon-based sentiment analysis                           |
| **TextBlob**                    | Text processing and word correction                              |
| **SpaCy**                       | Advanced grammar checking and NLP tasks                          |
| **Matplotlib & Seaborn**        | Data visualization and trend analysis                            |
| **textstat**                    | Readability score computation                                    |
| **language_tool_python**        | Grammar and spelling correction                                  |
| **Sentence Transformers**       | Advanced text embeddings and similarity                          |

## Project Structure

```
email-tone-sentiment-analysis/
├── CSC5240_Group12_Email_Tone_and_Sentiment_Analysis.ipynb  # Main analysis notebook
├── CSC5240_Email_Tone_and_Sentiment_Analysis.ipynb          # Alternative implementation
└── README.md                                                 # This file
```

## Setup and Installation

### Prerequisites

- Python 3.8+
- Google Colab (recommended) or Jupyter Notebook
- Internet connection for model downloads

### Installation

1. **Clone the repository**:

   ```bash
   git clone <repository-url>
   cd email-tone-sentiment-analysis
   ```

2. **Install dependencies** (run in notebook):

   ```python
   # The notebook includes automatic installation of:
   # - transformers
   # - nltk
   # - textblob
   # - spacy
   # - matplotlib
   # - seaborn
   # - textstat
   # - language-tool-python
   # - sentence-transformers
   ```

3. **Download required models**:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   ```

## Usage

### Basic Email Analysis

1. **Open the main notebook**: `CSC5240_Group12_Email_Tone_and_Sentiment_Analysis.ipynb`

2. **Install dependencies**: Run the "Installing Required Dependencies" section

3. **Input your email text**: Use the provided text input areas or modify the sample emails

4. **Run analysis**: Execute all cells to get comprehensive results

### Analysis Output

The system provides:

- **Sentiment Scores**: Positive, negative, and neutral sentiment percentages
- **Emotion Breakdown**: Detailed emotion classification with confidence scores
- **Tone Suggestions**: Professional tone improvement recommendations
- **Grammar Feedback**: Spelling and grammatical error identification
- **Visualizations**: Charts showing sentiment trends and emotion distribution

## Key Features in Detail

### Multi-Model Sentiment Analysis

- **VADER**: Rule-based sentiment analysis optimized for social media
- **TextBlob**: Simple polarity and subjectivity scoring
- **Transformer Models**: State-of-the-art BERT-based sentiment classification

### Advanced NLP Capabilities

- **Emotion Detection**: 6-class emotion classification (joy, sadness, anger, fear, surprise, disgust)
- **Sarcasm Detection**: Identifies potentially sarcastic content
- **Grammar Checking**: Comprehensive grammar and style suggestions
- **Readability Scoring**: Flesch Readability Ease Score for text complexity

### Professional Communication Focus

- **Tone Analysis**: Evaluates formality and professionalism
- **Style Suggestions**: Recommendations for improving communication clarity
- **Error Highlighting**: Visual indicators for potential issues

## Results and Applications

This system is particularly useful for:

- **Professional Communication**: Improving email tone and clarity
- **Content Review**: Pre-sending analysis of important communications
- **Training**: Educational tool for improving writing skills
- **Quality Assurance**: Automated review of customer-facing communications

## Technical Architecture

The project employs a modular approach with:

- **Multiple Analysis Pipelines**: Parallel processing of different analysis types
- **Model Ensemble**: Combining results from multiple models for robust predictions
- **Real-time Processing**: Immediate feedback for user input
- **Visualization Dashboard**: Comprehensive reporting and trend analysis

## Future Enhancements

- **Custom Model Training**: Domain-specific sentiment models
- **Real-time API**: Web service for integration with email clients
- **Batch Processing**: Analysis of email archives and historical data
- **Multi-language Support**: Extension to other languages
- **Advanced Analytics**: Trend analysis and predictive insights

## Authors

- **Thomas D. Robertson**
- **Arefin Niam**
- **Edward Gannod**

## License

This project is part of academic coursework and is intended for educational and portfolio purposes.

---

_This project demonstrates advanced NLP techniques and AI integration for practical business applications, showcasing skills in machine learning, data visualization, and software development._
