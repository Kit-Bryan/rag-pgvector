#!/usr/bin/env python
# coding: utf-8

# In[10]:


get_ipython().run_line_magic('pip', 'install pdfplumber langchain openai python-dotenv tiktoken')
get_ipython().run_line_magic('pip', 'install -U langchain-community psycopg2-binary pgvector')
get_ipython().run_line_magic('pip', 'install -U langchain-openai')
get_ipython().run_line_magic('pip', 'install gradio')


# In[31]:


from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import SemanticChunker
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import pdfplumber
from langchain.schema import Document
import os

load_dotenv()


# In[61]:


def load_pdf_with_pdfplumber(file_path, overlap_lines=8):
    documents = []
    prev_tail = ""

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if not text:
                continue

            # Add overlap from previous page
            combined_text = prev_tail + "\n" + text

            # Save last few lines for next iteration
            lines = text.splitlines()
            prev_tail = "\n".join(lines[-overlap_lines:]) if len(lines) >= overlap_lines else text

            documents.append(Document(page_content=combined_text, metadata={"page": page_num + 1}))

    return documents

file_path = "pdf/Sample_Partnership_Agreement.pdf"
documents = load_pdf_with_pdfplumber(file_path)
COLLECTION_NAME = os.path.splitext(os.path.basename(file_path))[0].lower().replace(" ", "_")

# loader = TextLoader('state_of_the_union.txt', encoding='utf-8')
# documents = loader.load()

print(documents)  # prints the document objects
print(len(documents))  # 1 - we've only read one file/document into the loader


# In[116]:


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
texts = text_splitter.split_documents(documents)

# semantic_splitter = SemanticChunker( OpenAIEmbeddings(), breakpoint_threshold_type="standard_deviation")
# texts = semantic_splitter.split_documents(documents)

print(texts)
print(len(texts))


# In[122]:


print(texts[4])


# In[40]:


embeddings = OpenAIEmbeddings()
vector = embeddings.embed_query('Testing the embedding model')
print(len(vector))  # 1536 dimensions
print(vector)


# In[41]:


doc_vectors = embeddings.embed_documents([chunk.page_content for chunk in texts[:5]])
print(len(doc_vectors))  # 5 vectors in the output
print(doc_vectors)
print(len(doc_vectors[0]))    # this will output the first chunk's 1536-dimensional vector


# In[ ]:


from langchain.vectorstores.pgvector import PGVector
# from langchain.vectorstores.pgvector import DistanceStrategy
CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5434/vector_db"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    # distance_strategy=DistanceStrategy.COSINE
)


# In[42]:


query = "Tell me about Section 24.4"

similar = db.similarity_search_with_score(query, k=2) # Return k number of documents

for doc in similar:
    print(doc)


# In[ ]:


retriever = db.as_retriever(search_kwargs={
    'k' : 3
})

query = "Tell me about Section 4. Term"
retriever.invoke(query) #Gets 3 most similar documents in the vector database to the given query.


# In[49]:


from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    "Use the following documents to answer the question. If you don't know the answer, just say that you don't know, dont try to make up an answer\n\n{context}\n\nQuestion: {input}."
)

# Create the document chain
document_chain = create_stuff_documents_chain(ChatOpenAI(model="gpt-4.1-mini"), prompt=prompt) # handles the "stuffing" strategy of combining documents and connect with LLM with system prompt.

# Create the retrieval chain
qa = create_retrieval_chain(retriever, document_chain) # Connects retriever and document chain


# In[51]:


# Test
query = "what happens if the partnership or the other Partners fail to accept the Offer"
qa.invoke({"input" : query})


# In[50]:


import gradio as gr

def answer_question(message, chat_history):
    result = qa.invoke({"input": message})
    return result["answer"] if "answer" in result else str(result)


gr.ChatInterface(fn=answer_question, title="Q&A Assistant").launch(share=True)


# In[ ]:




