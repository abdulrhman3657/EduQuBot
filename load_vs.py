import json
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Path to your JSON file (text)
filename = 'data\articles.json' # the text dataset in json form

# Load the JSON file into a list
with open(filename, 'r', encoding='utf-8') as file:
    data_list = json.load(file) # load the dataset in a list

persist_directory = "./data/" # the path for the vector database


# data_list: the text data in a list
# OpenAIEmbeddings(): the type of embedding
# persist_directory: the path for the vector database
vector_store = Chroma.from_texts(data_list,OpenAIEmbeddings(model="text-embedding-ada-002"),persist_directory=persist_directory)

vector_store.persist()
