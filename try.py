import streamlit as st
import json
import random
import nltk

from nltk.sentiment import SentimentIntensityAnalyzer

# Load the data from JSON file
with open('intents.json', 'r') as f:
    data = json.load(f)

# Initialize the sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Initialize the chat history and sentiment scores lists
chat_history = []
sentiment_scores = []

# Define a function to identify the tag for the input message
def get_tag(message):
    for pattern in data['intents'][0]['patterns']:
        if pattern in message:
            return data['patterns'][pattern]
    return None

# Define a function to generate a response for the input message
def get_response(message, tag):
    if tag in data['intents'][0]['responses']:
        response = random.choice(data['responses'][tag])
    else:
        response = "Sorry, I don't understand. Can you please try again?"
    return response

# Define the Streamlit app layout
st.title('NLP-based Chatbot')
input_message = st.text_input('You: ', key='input')
chat_log = st.empty()

# Check if the user has entered any input
if input_message:
    # Add the user's message to the chat history
    chat_history.append(('User', input_message))

    # Get the tag for the user's message
    tag = get_tag(input_message)

    # Generate a response for the user's message
    response = get_response(input_message, tag)

    # Add the chatbot's response to the chat history
    chat_history.append(('Chatbot', response))

    # Calculate the sentiment score for the chatbot's response
    sentiment_score = sia.polarity_scores(response)['compound']
    sentiment_scores.append(sentiment_score)

    # Display the chat history and sentiment analysis results in the Streamlit app
    chat_log.write('Chatbot: ' + response)
    chat_log.write('')

    for (sender, message) in chat_history:
        chat_log.write(sender + ': ' + message)
    chat_log.write('')

    sentiment = sum(sentiment_scores) / len(sentiment_scores)
    if sentiment > 0.5:
        sentiment_text = 'Positive'
    elif sentiment < -0.5:
        sentiment_text = 'Negative'
    else:
        sentiment_text = 'Neutral'

    st.write('Sentiment analysis: ' + sentiment_text + ' (score: ' + str(sentiment) + ')')
