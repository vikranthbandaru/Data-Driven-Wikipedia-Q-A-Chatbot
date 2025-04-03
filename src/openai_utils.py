from openai import OpenAI
import traceback
import openai
# Set your OpenAI API key
client = OpenAI(api_key='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')
solr_url = "http://34.44.184.192:8983/solr/IRF24P2"

# Function to generate a summary using OpenAI API
def evaluate_answer_relevance(question, answer):
    # Updated prompt to ask for a relevance score between 1 and 10
    prompt = (
        f"Given the following question and answer, please rate the relevance of the answer to the question on a scale of 1 to 10, "
        f"where 1 is completely irrelevant and 10 is highly relevant:\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Please **ONLY provide a score between 1 and 10 based on the relevance of the answer to the question. ONLY GIVE A NUMBER"
    )
    
    # Send request to OpenAI API for relevance score
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",  # Use the model you want to use
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=50,  # Limiting to a small response since it's just a score
        n=1,
        stop=None,
        temperature=0,  # Set temperature to 0 for deterministic output
    )

    # Extract and return the relevance score
    print(response)
    relevance_score = response.choices[0].message.content.strip()  # Extract the score from the response content
    print(relevance_score)
    try:
        relevance_score = int(relevance_score)  # Ensure it's an integer
    except ValueError:
        relevance_score = 5  # If it can't be parsed, return None

    return relevance_score

def query_enrichment(query):
    prompt = (
        f"Given the following query, extract and list all the proper noun keywowds with corrected speliing:\n\n"
        f"{query}\n\n"
        f"NOTE: give only proper noun(case lowered) seperated by space and nothing else:\n"
       
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content

def classify_topics1(query, topics):
    """
    Classifies a query into one or more relevant topics using keyword extraction 
    and topic mapping examples in the prompt.

    Args:
        query (str): The user's query.
        topics (list): A list of topic descriptions.

    Returns:
        list: A list of classified topics.
    """
    # Updated few-shot examples for multi-topic classification
    few_shot_examples = """
    Classify the query into one or more relevant topics from the following:
    1. Health
    2. Environment
    3. Technology
    4. Economy
    5. Entertainment
    6. Sports
    7. Politics
    8. Education
    9. Travel
    10. Food

    First, extract the keywords from the query.
    Then, classify the query into relevant topics based on the keywords.

    Examples:
    Sentence: "What are the latest advancements in AI and its impact on mental health?"
    Keywords: AI, mental health
    Topics: Technology, Health

    Sentence: "How are economy and technology related?"
    Keywords: economy, technology
    Topics: Economy, Technology

    Sentence: "Which team won the last FIFA World Cup, and how is the sports industry evolving?"
    Keywords: FIFA World Cup, sports industry
    Topics: Sports, Economy

    Sentence: "What are the latest food security trends in the world?"
    Keywords: food security
    Topics: Food

    Sentence: "What are the health impacts of swimming?"
    Keywords: health, swimming
    Topics: Health, Sports

    For the following query:
    Sentence: "{query}"
    **NOTE: the reponse should only be topics seperated by ,.
    """
    # Integrate the prompt into the few-shot examples
    full_prompt = few_shot_examples.format(query=query)

    # Use an NLP model or service to classify the topics
    # Replace this with the desired NLP service or API call
    # e.g., `response = openai.Completion.create(...)`
    # Here, we'll simulate the response for demonstration
    response = simulate_response(full_prompt, topics)  # Replace with actual API call

    # Parse and return the topics from the response
    return [topic.strip() for topic in response.split(",")]

def simulate_response(prompt, topics):
    """
    Simulates the response for demonstration purposes.

    Args:
        prompt (str): The input prompt.
        topics (list): A list of topic descriptions.

    Returns:
        str: A simulated response with relevant topics.
    """
    # This is where you'd interact with an NLP service. This is a mock example.
    if "AI" in prompt or "artificial intelligence" in prompt:
        return "Technology"
    elif "mental health" in prompt:
        return "Health, Technology"
    elif "deforestation" in prompt:
        return "Environment"
    elif "FIFA World Cup" in prompt:
        return "Sports, Economy"
    elif "food security" in prompt:
        return "Food"
    elif "swimming" in prompt:
        return "Health, Sports"
    else:
        return "General"


def classify_topics(query, topics):
    """
    Classifies a query into one or more relevant topics using a few-shot prompt.

    Args:
        query (str): The user's query.
        topics (list): A list of topic descriptions.

    Returns:
        list: A list of classified topics.
    """
    # Few-shot examples for multi-topic classification
    few_shot_examples = """
    Classify the query into one or more relevant topics from the below 10 topics:
    1. Health
    2. Environment
    3. Technology
    4. Economy
    5. Entertainment
    6. Sports
    7. Politics
    8. Education
    9. Travel
    10. Food

    Examples:
    Query: "What are the latest advancements in AI and its impact on mental health?"
    Topics: Technology, Health

    Query: "What is deforestation?"
    Topics: Environment

    Query: "How are economy and technology related?"
    Topics: Economy, Technology

    Query: "what is the health impact of swimmings and politics?"
    Topics: Health, Sports, Politics

    NOTE: if there are multi topics. Your answer must contain more than one topic.
    the reponse should only be topics seperated by ,. 
    """

    # Construct the prompt
    prompt = f"{few_shot_examples}\nQuery: \"{query}\"\nTopics:"

    # Use OpenAI's GPT model for classification
    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
        topics_response = response.choices[0].message.content.strip()
        print("topics_response",topics_response)
        # Split topics into a list, ensure they match predefined topics
        classified_topics = [
            t.strip() for t in topics_response.split(",") if t.strip() in topics
        ]
        print("classified_topics",classified_topics)
        return classified_topics if classified_topics else ["Uncategorized"]
    except Exception as e:
        return [f"Error in topic classification: {e}"]


def classify_query(query):
    """
    Classify the query as either 'chit-chat' or 'topic-related'.
    Uses a few-shot learning prompt for classification.
    """
    prompt = """
    Classify the following queries as either 'chit-chat' or 'topic-related'. Respond with exactly one of the two labels.

    Examples:
    1. "How's the weather today?" => chit-chat
    2. "Can you tell me more about deep learning?" => topic-related
    3. "What's your favorite movie?" => chit-chat
    4. "Explain the benefits of federated learning." => topic-related

    Query: "{}"
    respond only either chit-chat ot topic-related
    """.format(query)

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].message.content

def call_openai_api(prompt, model="gpt-3.5-turbo-0125", max_tokens=500, temperature=0.7):
    """
    Calls OpenAI's API to generate a response based on the given prompt.

    Args:
        prompt (str): The input prompt for the model.
        model (str): OpenAI model to use (default is "gpt-4").
        max_tokens (int): Maximum number of tokens for the response.
        temperature (float): Sampling temperature for creativity control.

    Returns:
        str: Generated response.
    """
    try:
        response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"OpenAI API error: {e}"

def generate_chitchat_response(query):
    """
    Generate a conversational response to a chit-chat query using OpenAI's GPT API.

    Args:
        query (str): The user's chit-chat query.

    Returns:
        str: The assistant's response to the chit-chat query.
    """
    # Few-shot prompt to guide the assistant's conversational style
    prompt = f"""
    You are a friendly, conversational assistant. Respond to casual, informal questions or comments in a natural and engaging manner.

    Examples:
    - User: "What's your favorite food?"
      Assistant: "I don't eat, but if I could, I'd love to try pizza!"
    - User: "How's the weather there?"
      Assistant: "It’s always perfect here in the digital world. How about you?"
    - User: "Tell me a joke."
      Assistant: "Why don't skeletons fight each other? Because they don’t have the guts!"

    Note: if you donot have enough information. politely decline to answer.

    Now respond to this query:
    User: "{query}"
    Assistant:
    """

    try:
        response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0,
    )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Oops! Something went wrong while generating a response: {e}"

def query_completion(query, combined_history):
    # Define a few-shot prompt with examples of how to correct or expand a query based on conversation history
    prompt = (
        f"Given the following query, and previous conversation history, your task is to complete, fix, or expand the query "
        f"based on the provided context and prior conversations:\n\n"
        
        f"EXAMPLE 1:\n"
        f"QUERY: What is the its capital?\n"
        f"HISTORY: User: Tell me about France. \nBot: France is a country in Western Europe, famous for its art, culture, and history.\n"
        f"Bot's completion: What is the capital of France?\n\n"
        
        f"EXAMPLE 2:\n"
        f"QUERY: Tell me about its weather.\n"
        f"HISTORY: User: What is the best time to visit Paris? \nBot: Paris is best visited in the spring and fall due to mild temperatures.\n"
        f"Bot's completion: Tell me about the weather in Paris?\n\n"
        
        f"EXAMPLE 3:\n"
        f"QUERY: What is symptoms diabetes?\n"
        f"HISTORY: User: What are the causes of diabetes? \nBot: Diabetes can be caused by genetic factors, lifestyle, and other health conditions.\n"
        f"Bot's completion: What are the symptoms of diabetes?\n\n"
        
        f"QUERY: {query}\n\n"
        f"HISTORY: {combined_history}\n\n"
        f"NOTE: Complete, correct, or expand the query based on the above conversation history and only return the completed query:\n"
    )

    # Call the OpenAI API with the few-shot prompt
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=200,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return response.choices[0].message.content

import pysolr

def get_summaries_through_solr(solr_url, query, topics, k):
    """
    Queries the Solr index and retrieves top-k summaries filtered by topics.

    Parameters:
        solr_url (str): URL of the Solr core.
        query (str): Search query to match in summaries.
        topics (list): List of topics to filter results by.
        k (int): Number of top results to retrieve.

    Returns:
        list: List of summaries from the retrieved results.
    """
    # Initialize the Solr client
    solr = pysolr.Solr(solr_url, always_commit=True, timeout=10)

    # Build the filter query for topics
    fq_topics = " OR ".join([f'topic:"{topic}"' for topic in topics])
    search_query = query_enrichment(query)
    print("SEARCH QUERY",search_query)
    # Define the Solr query parameters
    solr_query = {
        "q": f'summary:"{search_query}"',  # Query in the summary field
        # "fq": fq_topics,  # Filter query by topics
        "op":"OR",
        "fl": "summary",  # Retrieve only the summary field
        "rows": k,
        "sort": "score desc" 
    }

    # Execute the query
    results = solr.search(**solr_query)

    # Extract summaries from the results
    summaries = [result['summary'] for result in results.docs if 'summary' in result]

    return summaries

def answer_question_with_rag(chroma_client, question, history, filter_topics):
    """
    Use RAG with ChromaDB for question answering, incorporating history and topic filtering for better context.

    Args:
        chroma_client: The ChromaDB client for vector database queries.
        question (str): The user's question.
        history (list): Chat history for context.
        top_k (int): Number of top documents to retrieve for context.
        filter_topics (list): List of topics to filter metadata during retrieval.

    Returns:
        str: Generated answer from the language model.
    """
    try:
        # Step 1: Incorporate chat history into the query
        combined_history = "\n".join([f"User: {msg['content']}" if msg["role"] == "user" 
                                       else f"Assistant: {msg['content']}" 
                                       for msg in history])

        # Combine question and history for query enrichment
        #enriched_query = f"Question: {question}\n\nConversation History:\n{combined_history}\n\n"
        enriched_query = query_completion(question, combined_history)
        print("ENRIHED QUERY",enriched_query)
        print("FILTER TOPICS REACHED", filter_topics)
        retrieved_documents = get_summaries_through_solr(solr_url, enriched_query, filter_topics, 8)

        # Combine retrieved documents into a single context
        context = "\n".join([doc for doc in retrieved_documents])
    
        print("CONTEXT:",context)
        # Step 3: Prepare the prompt for the language model
        prompt = (
            "You are a helpful assistant answering questions based on the provided context and conversation history. "
            "Use the context and history to provide accurate and relevant answers under 150 words. If there is insufficient information, "
            "politely decline to answer.\n\n"
            f"Context:\n{context}\n\n"
            f"Conversation History:\n{combined_history}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        # Step 4: Generate the answer using a language model
        answer = call_openai_api(prompt)

        return answer

    except Exception as e:
        traceback.print_exc()
        return f"An error occurred during processing: {e}"


