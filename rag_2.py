#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('pip', 'install pdfplumber langchain openai langchain-experimental python-dotenv tiktoken')
get_ipython().run_line_magic('pip', 'install -U langchain-community psycopg2-binary pgvector')
get_ipython().run_line_magic('pip', 'install -U langchain-openai')
get_ipython().run_line_magic('pip', 'install gradio')


# In[77]:


from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
import pdfplumber
from langchain.schema import Document
import os

load_dotenv()


# In[92]:


import os
import pdfplumber

import re
import pdfplumber
from langchain.schema import Document

def load_pdf_with_pdfplumber(file_path):

    documents = []

    # Read full PDF text
    with pdfplumber.open(file_path) as pdf:
        all_text = "\n".join([page.extract_text() or "" for page in pdf.pages])



    # Pattern matches lines like "6.2. Product Development at Dyson"
    section_pattern = re.compile(r'\n(\d+\.\d+)\.\s+([^\n]+)')
    matches = list(section_pattern.finditer(all_text))



    for i in range(len(matches)):
        start_idx = matches[i].start()
        end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(all_text)

        section_number = matches[i].group(1).strip()
        section_title = matches[i].group(2).strip()
        section_content = all_text[start_idx:end_idx].strip()

        documents.append(Document(
            page_content=section_content,
            metadata={"section": section_number, "title": section_title}
        ))


    if matches:
        first_section_start = matches[0].start()
        intro_text = all_text[:first_section_start].strip()

        if intro_text:
            # Try to extract the section label and title from the intro text
            intro_lines = intro_text.splitlines()
            chapter_line = next((line for line in intro_lines if "Chapter" in line), None)
            title_line = next((line for line in intro_lines if line.strip() and "Chapter" not in line), "")

            documents.append(Document(
                page_content=intro_text,
                metadata={
                    "section": chapter_line.strip() if chapter_line else "Unknown",
                    "title": title_line.strip() if title_line else "Untitled"
                }
            ))

    return documents



# ===== Run and print =====
file_path = "pdf/Unique_New_Product_Innovation_Based.pdf"
documents = load_pdf_with_pdfplumber(file_path)
COLLECTION_NAME = os.path.splitext(os.path.basename(file_path))[0].lower().replace(" ", "_")

for doc in documents:
    print({doc.metadata['section']} , {doc.metadata['title']})
    print(doc.page_content[:600], "...")

print(f"\nTotal sections parsed: {len(documents)}")


# In[93]:


from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

def split_documents_with_chunk_index(documents, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    result = []

    for doc in documents:
        chunks = splitter.split_documents([doc])

        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_index": i + 1,
                "chunk_total": len(chunks)
            })
            result.append(chunk)

    return result

# Split and preserve chunk index/total
texts = split_documents_with_chunk_index(documents, chunk_size=1000, chunk_overlap=200)

# Output check
for t in texts[:3]:  # preview first 3 chunks
    print(f"Section {t.metadata.get('section')} - Chunk {t.metadata.get('chunk_index')}/{t.metadata.get('chunk_total')}")
    print(t.page_content[:300], "...\n")

print(f"\nTotal chunks: {len(texts)}")


# In[94]:


# print(texts[0])
print(texts[1])
# print(texts[2])
# print(texts[3])


# In[99]:


from langchain.vectorstores.pgvector import PGVector

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# from langchain.vectorstores.pgvector import DistanceStrategy
COLLECTION_NAME = os.path.splitext(os.path.basename(file_path))[0].lower().replace(" ", "_")
CONNECTION_STRING = "postgresql+psycopg2://postgres:postgres@localhost:5434/vector_db"

db = PGVector.from_documents(
    embedding=embeddings,
    documents=texts,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
    # distance_strategy=DistanceStrategy.COSINE
)
retriever = db.as_retriever(search_kwargs={
    'k' : 5
})


# In[100]:


from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI 
from langchain.prompts import PromptTemplate

prompt = PromptTemplate.from_template(
    """You are given chunks of a document, each with associated metadata.

Each chunk includes:
- `section`: the section number from the original document (e.g., "6.2")
- `title`: the title of that section
- `chunk_index`: the position of the chunk within the section (e.g., 1 of 3)
- `chunk_total`: the total number of chunks in that section

Use this information to answer the question as accurately as possible.

If the answer is not present in the context, reply with: "I don't know."

Context:
{context}

Question: {input}

"""
)

# Create the document chain
document_chain = create_stuff_documents_chain(ChatOpenAI(model="gpt-4.1-mini"), prompt=prompt) # handles the "stuffing" strategy of combining documents and connect with LLM with system prompt.

# Create the retrieval chain
qa = create_retrieval_chain(retriever, document_chain) # Connects retriever and document chain


# In[101]:


# Test
query = "What belief did James Dyson maintained?"
qa.invoke({"input" : query})


# In[102]:


import gradio as gr

def answer_question(message, chat_history):
    result = qa.invoke({"input": message})
    print(result)
    return result["answer"] if "answer" in result else str(result)


gr.ChatInterface(fn=answer_question, title="Q&A Assistant").launch(share=True)


# In[ ]:





# In[ ]:




