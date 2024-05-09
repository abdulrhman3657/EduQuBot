import streamlit as st
import openai
from langchain_core.messages import AIMessage,HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain 
from langchain.chains.combine_documents import create_stuff_documents_chain


# streamlit run chatbot.py

load_dotenv() # load openAI API


edubot_icon = "src\edubot_icon.png"
user_icon = "src\\user.png"

# "context" will be empty if the vector database is not loaded properly

def get_vectorstore(): 

    persist_directory = "./data/" # the path for the vector database

    # embedding_function: use the same embedding when creating the database
    vector_store = Chroma(persist_directory = persist_directory,embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002"))


    return vector_store # get the vector data from the database


# A vector store retriever is a retriever that uses a vector store to retrieve 
# documents. It is a lightweight wrapper around the vector store class to make
# it conform to the retriever interface. It uses the search methods implemented
# by a vector store, like similarity search 
# and MMR, to query the texts in the vector store.
# By default, the vector store retriever uses similarity search.
# You can also set a retrieval method that sets a similarity score threshold 
# and only returns documents with a score above that threshold.


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.5,  max_tokens=1200)  # ft:gpt-3.5-turbo-1106:personal::9G7Xz5en

    retriever = vector_store.as_retriever(search_kwargs={"k": 5}) # use the vector store to retrieve the context
    # search_kwargs={"k": 5}: the number of returned relavent answers
    # By default, the vector store retriever uses similarity search


    prompt = ChatPromptTemplate.from_messages([ # chat history memory
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user","{input}"),
        ("user","Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])


    # If there is no chat_history, then the input is just passed directly to the retriever. 
    # If there is chat_history, then the prompt and LLM will be used to generate a search query.
    # That search query is then passed to the retriever.

    # take the retriever with the conversation history then return the context (same as retriever but with memory)
    retriever_chain = create_history_aware_retriever(llm,retriever,prompt) # takes conversation history and returns the context.

    return retriever_chain # the actual retriever


def get_conversational_rag_chain(retriever_chain):

    llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.5, max_tokens=1200)

    prompt = ChatPromptTemplate.from_messages([ # system prompt
        ("system","Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user","{input}"),
    ])


    # pass a list of context examples to the model
    # EX: pass the most relevant 3 examples to the model
    stuff_documents = create_stuff_documents_chain(llm,prompt)


    # This chain takes in a user question, 
    # which is then passed to the retriever to fetch relevant context. 
    # the context (and original inputs) are then passed to an LLM to generate a response
    # original inputs: is the user new question and the chat history
    return create_retrieval_chain(retriever_chain,stuff_documents) 
    # takes the user question along with the chat history and the retrieved context and passes it to the model ?


def get_response(user_input): 

    # Session State is a way to share variables between reruns, for each user session.
    # Append user's question to the chat history
    # a session exist as long as the web tab is open
    # sessions are independent
    # session_state stores the values of the variables
    st.session_state.chat_history.append(HumanMessage(content=user_input)) # appened the model response every time get_response() is called
    
    # Get the context retrieval chain
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    # Get the conversational chain
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain) # input retriever_chain to the model with the chat history
    
    # Invoke the conversational chain to get a response
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    }) # input conversation_rag_chain with the user question to get the response
    
    # Append AI's response to the chat history
    st.session_state.chat_history.append(AIMessage(content=response['answer']))
    
    # Return the response for display
    return response # response contain "chat_history", "input", "context", "answer"



#app config
st.set_page_config(page_title ="EduBot",page_icon = edubot_icon)
st.title("EduBot")
if "chat_history" not in st.session_state: # if chat_history is not already created in the session
    st.session_state.chat_history = [ # create the first message in the chat history list
        AIMessage(content="كيف استطيع مساعدتك EduBot مرحبا أنا") # Hello, I am ,How can i help you
    ]

if "vector_store" not in st.session_state: # if vector_store is not already add to the session
    st.session_state.vector_store = get_vectorstore() # get the vector data once in a session

# user input
user_query = st.chat_input("اكتب هنا")
if user_query is not None and user_query !="":
    response = get_response(user_query) 
    # response contain "chat_history", "input", "context", "answer"

    # for debugging(remove)
    # the RAG output. the relevant results

    # st.write(response)
    st.write(response["context"]) # only RAG results

    # st.write(response["context"][0]) # the first RAG result
    


# conversation. using response["input"] and using response["answer"]
for message in st.session_state.chat_history:

    if isinstance(message,AIMessage): # AIMessage: model response
        with st.chat_message("AI",avatar = edubot_icon):
            st.write(message.content)

    elif isinstance(message,HumanMessage): # HumanMessage: user question
        with st.chat_message("user",avatar = user_icon):
            st.write(message.content)
