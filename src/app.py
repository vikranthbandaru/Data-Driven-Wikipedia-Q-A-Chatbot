import streamlit as st
import time
import pandas as pd
import os
import matplotlib.pyplot as plt
from openai_utils import answer_question_with_rag, generate_chitchat_response, classify_query, classify_topics, evaluate_answer_relevance, classify_topics1
import openai
import traceback
import openai
from datasets import Dataset 
from ragas.metrics import answer_relevancy
from ragas import evaluate

if 'history' not in st.session_state:
    st.session_state.history = []
if 'query_data' not in st.session_state:
    st.session_state.query_data = pd.DataFrame(columns=["timestamp", "query", "terms", "response_terms", "classification", "topics", "relevancy_score"])
if 'relevancy_scores' not in st.session_state:
    st.session_state.relevancy_scores = []  # For storing relevancy scores
    
# Define the topic categories
TOPICS = [
    "Health", "Environment", "Technology", "Economy", "Entertainment",
    "Sports", "Politics", "Education", "Travel", "Food"
]

# Streamlit App
st.set_page_config(page_title="Q&A with RAG", layout="wide")
st.title("Q&A with RAG (Information Retrieval Project 3)")

# Custom CSS for enhanced visual styling
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    body {
        font-family: 'Poppins', sans-serif;
        background-color: #f4f7fb;
    }
    .title {
        color: #3f72af;
        text-align: center;
        font-size: 2.5em;
        font-weight: 600;
    }
    
    .sidebar .sidebar-content {
        background-color: #3f72af;
        color: white;
        padding: 20px;
        border-radius: 10px;
    }
    
    .sidebar .sidebar-header {
        color: #ffffff;
        font-weight: 600;
    }
    
    .stButton>button {
        background-color: #6c63ff;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }

    .stButton>button:hover {
        background-color: #5a54e7;
    }
    
    .query-card {
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
        margin-bottom: 10px;
    }

    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        align-self: flex-start;
        max-width: 70%;
        text-align: left;
    }
    .assistant-bubble {
        background-color: #EAEAEA;
        padding: 10px;
        border-radius: 10px;
        margin: 5px;
        align-self: flex-end;
        max-width: 70%;
        text-align: right;
    }
    .chat-container {
        display: flex;
        flex-direction: column;
        height: 400px;
        overflow-y: auto;
        background-color: #F0F0F0;
        padding: 10px;
        border: 1px solid #E0E0E0;
    }
    .input-container {
        position: fixed;
        bottom: 0;
        width: 100%;
        background-color: #FFFFFF;
        padding: 10px;
        border-top: 1px solid #E0E0E0;
    }

    /* Add more colors */
    .query-stats {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }

    /* Sidebar components */
    .sidebar .multiselect .multiselect-checkbox {
        border-radius: 8px;
        color: blue;
    }

    .stTextInput>div>div>input {
        font-size: 16px;
        padding: 10px;
        
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for visualizations and category filters
st.sidebar.title("Navigation")
menu = st.sidebar.radio(
    "Go to",
    ("Home", "About Our Team", "Products", "Contact Us", "Terms and Conditions")
)

if menu == "Home":
    # Checkbox section for selecting categories
    st.sidebar.subheader("Filter by Categories")
    selected_categories = st.sidebar.multiselect("Choose topics:", TOPICS)

    # Visualization section
    st.sidebar.subheader("Query Statistics")
    query_data = st.session_state.query_data

    # Queries over time
    if not query_data.empty:
        with st.sidebar:
            # Assuming `query_data` is a DataFrame with columns: "timestamp", "terms", "topics", "query", "responses", "errors"
             # Queries by Topics
            st.markdown("### **Queries by Topics**")
            topic_counts = query_data["topics"].explode().value_counts()  # Count multiple topics
            st.bar_chart(topic_counts, color="#6c63ff")
            
            # Number of Terms per Query Over Time
            st.markdown("### **Number of Terms per Query Over Time**")
            fig, ax = plt.subplots()
            query_data["terms"].plot(ax=ax, kind="line", title="Terms per Query Over Time", color="#3f72af")
            st.pyplot(fig)

            # Number of Terms per Query Over Time
            st.markdown("### **Number of response_terms per response Over Time**")
            fig, ax = plt.subplots()
            query_data["response_terms"].plot(ax=ax, kind="line", title="Terms per response Over Time", color="#3f72af")
            st.pyplot(fig)
            
           
            
            # Historical Top 5 Queries
            st.markdown("### **Historical Top 5 Queries**")
            top_queries = query_data["query"].value_counts().head(5)  # Get top 5 queries
            st.table(top_queries)

            st.markdown("### **Real-Time Answer Relevancy Scores**")
            # Plotting answer relevancy scores
            fig, ax = plt.subplots()
            if len(st.session_state.relevancy_scores) > 0:
                pd.Series(st.session_state.relevancy_scores).plot(ax=ax, kind="line", color="#FF6347", title="Answer Relevancy Over Time")
            else:
                ax.set_title("No data yet")
            st.pyplot(fig)
            
            # Assuming the timestamp and other data is already populated in session_state
            st.markdown("### **Queries in the Last 5 Minutes**")
            now = time.time()
            
            # Filter queries that happened in the last 5 minutes (300 seconds)
            last_5_minutes = query_data[query_data["timestamp"] > (now - 300)]
            
            # Create a new column for rounded timestamps (to 10-second intervals)
            last_5_minutes['timestamp_rounded'] = (last_5_minutes['timestamp'] // 10) * 10  # Round to the nearest 10 seconds
            
            # Group by the rounded timestamp and count the number of queries in each bin
            time_series = last_5_minutes.groupby("timestamp_rounded").size()
            
            # Create a plot
            fig, ax = plt.subplots(figsize=(10, 6))  # Bigger plot for better readability
            time_series.plot(ax=ax, kind="line", color="#6c63ff", marker='o', linewidth=2)
            
            # Add title and labels
            ax.set_title("Number of Queries in the Last 5 Minutes", fontsize=16)
            ax.set_xlabel("Time (seconds since epoch)", fontsize=12)
            ax.set_ylabel("Number of Queries", fontsize=12)
            
            # Add gridlines for better visibility
            ax.grid(True)
            
            # Optionally, you can format the x-axis to show the actual time rather than raw seconds
            # ax.set_xticks(time_series.index)
            # ax.set_xticklabels([time.strftime('%H:%M:%S', time.localtime(ts)) for ts in time_series.index])
            
            # Display the plot
            st.pyplot(fig)
            
            # Grouping Queries by Length
            st.markdown("### **Grouping Similar Queries**")
            query_data["query_length"] = query_data["query"].apply(len)
            query_groups = query_data.groupby("query_length")["query"].apply(list)
            st.write("Query Groups by Length:")
            st.table(query_groups)
            
        

    # Main Question and Answer Section
    st.header("Ask Your Questions")

    # Chat history display
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input for user question
    question = st.text_input("Enter your question:", key="question_input", placeholder="Type your question here...")

    # Button with advanced styling
    if st.button("Ask", key="ask_button"):
        if question:
            with st.spinner("Processing your question..."):
                try:
                    # Step 1: Classify the query
                    classification = classify_query(question)
                    print(classification)

                    # Step 2: Handle chit-chat or topic-related queries
                    if classification == "chit-chat":
                        response = generate_chitchat_response(question)
                        filter_topics = ["chit chat"]
                    elif classification == "topic-related":
                        topics = classify_topics(question, TOPICS)
                        
                        # Filter metadata using classified topics and selected categories
                        filter_topics = topics
                        if selected_categories:
                            filter_topics = list(set(topics) | set(selected_categories))
                        print("FILTERED TOPICS----------------------->", filter_topics)
                        response = answer_question_with_rag(None, question, st.session_state.history, filter_topics)
                    else:
                        response = "Unable to classify the query. Please try again."
                        topics = []

                    # Update session history
                    st.session_state.history.append({"role": "user", "content": question})
                    st.session_state.history.append({"role": "assistant", "content": response})

        
                    relevancy_score = evaluate_answer_relevance(question, response)

                    # Record the query with metadata
                    num_terms = len(question.split(" "))
                    num_terms_res = len(response.split(" "))
                    st.session_state.query_data = pd.concat([st.session_state.query_data, pd.DataFrame([{
                        "timestamp": time.time(),
                        "query": question,
                        "terms": num_terms,
                        "response_terms": num_terms_res,
                        "classification": classification,
                        "topics": filter_topics,
                        "relevancy_score": relevancy_score
                    }])], ignore_index=True)

                    # Update relevancy scores for visualization
                    st.session_state.relevancy_scores.append(relevancy_score)

                    # Refresh to display updated chat and visualizations
                    st.rerun()
                except Exception as e:
                    traceback.print_exc()
                    st.error(f"An error occurred: {e}")
        else:
            traceback.print_exc()
            st.warning("Please enter a question to proceed.")

# Handle "Contact Us" page
elif menu == "About Our Team":
    st.subheader("We'd love to hear from you!")

    # Contact information (modify these with your actual details)
    st.write("### Our Team")
    st.write("1. **Karan Ramchandani(50615003)**")
    st.write("2. **Vikranth Bandaru(50607347)** ")
 
    
# Footer
st.markdown("---")
st.markdown("© 2024 Q&A Chatbot. All rights reserved.")
