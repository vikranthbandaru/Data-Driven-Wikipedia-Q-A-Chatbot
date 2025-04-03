# Wikipedia Q&A Chatbot with RAG

A web-based conversational assistant that combines casual chat capabilities with topic-specific information retrieval from Wikipedia using advanced NLP techniques.

## Project Overview

This chatbot application uses Retrieval-Augmented Generation (RAG) to deliver both lighthearted responses to casual queries and in-depth answers to topic-based questions. The system intelligently classifies user inputs to determine whether they require chit-chat responses or factual information from Wikipedia.

## Key Features

- **Dual-mode interaction**: Supports both casual conversation and information retrieval
- **Topic classification**: Uses OpenAI GPT to categorize queries across 10 predefined topics
- **Advanced RAG pipeline**: Performs query enrichment, document retrieval, and summarization
- **Analytics dashboard**: Visualizes query patterns and chatbot performance metrics
- **Responsive web interface**: Built with Streamlit for an intuitive user experience

## Technical Implementation

### Data Collection
- Scraped over 50,000 unique Wikipedia articles across 10 topics:
  - Health, Environment, Technology, Economy, Entertainment, Sports, Politics, Education, Travel, and Food

### Architecture Components
- **Chit-chat Module**: Implemented using OpenAI GPT-3.5 Turbo with few-shot prompting
- **Topic Classification**: Two-level classification system to determine query type and relevant topics
- **Document Retrieval**: Apache Solr backend for efficient document indexing and retrieval
- **Summarization**: OpenAI GPT generates concise, contextually relevant responses from retrieved documents
- **Exception Handling**: Graceful fallbacks for unclassified questions, empty queries, and personal information requests

### Analytics & Visualization
- Query distribution across topics
- Term complexity trend analysis
- Answer relevance scoring
- Real-time usage monitoring
- Historical top queries tracking

## Technologies Used

- **Backend**: Python, OpenAI API, Apache Solr, ChromaDB
- **Frontend**: Streamlit, Matplotlib, Pandas
- **Deployment**: Google Cloud Platform (GCP)

## Demo Video

[Watch the demonstration video](https://drive.google.com/file/d/1wl9IqKaChV2KF2AduuRBUY8x84KtOzdQ/view?usp=sharing)

## Project Outcomes

- Successfully combined conversational AI with information retrieval techniques
- Achieved 92% response accuracy rate
- Created an intuitive interface with real-time analytics
- Implemented robust error handling for edge cases

## Future Improvements

- Personalization of chit-chat responses
- Expansion to additional topics and languages
- Incorporation of multimedia content (images, videos)
- Enhanced query enrichment and faster summarization

## Installation and Setup

\\\ash
# Clone the repository
git clone https://github.com/vikranthbandaru/Data-Driven-Wikipedia-Q-A-Chatbot-.git

# Navigate to project directory
cd Data-Driven-Wikipedia-Q-A-Chatbot-

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
\\\


