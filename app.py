import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
nltk.download('punkt') 
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
from PIL import Image
import numpy as np
from scipy.sparse import csr_matrix




st.set_page_config(page_title="Spam Detection app")


# Displaying images using streamlit
#image = Image.open('spam.jpeg')

#st.image(image, width=500)

# function defining
def transform_text(text):
    #converting to lower case
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    # removing the special character
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # removing stop words
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    #applying stemming
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

tfidf = pickle.load(open('bot_vectorizer.pkl','rb'))
model = pickle.load(open('bot_model.pkl','rb'))

# st.title("Email/SMS Spam Classifier")

st.header("Topic Classifier")

input_sms = st.text_area("Topic Classifier", max_chars=256)


if st.button('Predict'):
        # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms]).toarray()
    # 3. predict
    result = model.predict(vector_input)[0]
        # 4. Display
    results = model.predict(vector_input)
    headers = ["Delivery", "Funny", "Good Bye", "Greeting", "Greetings", "Items", "Payments"]
    for result in results:
        st.header(headers[result])





# if st.button('Predict'):


#     if result == 0:
#         st.header("Delivery")        
#     elif result == 1:
#         st.header("Funny")
#     elif result == 2:
#         st.header("Good Bye")
#     elif result == 3:
#         st.header("Greeting")
#     elif result == 4:
#         st.header("Greetings")
#     elif result == 5:
#         st.header("Items")
#     else:
#         st.header("Payments")
