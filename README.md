# RAG-Enhanced-Text-Classification-for-Support-Ticket-Categorization

## Technologies
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=Python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/HuggingFace-29B5E8?style=for-the-badge&logo=HuggingFace&logoColor=white)
![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-4285F4?style=for-the-badge&logo=SentenceTransformers&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-FF9900?style=for-the-badge&logo=numpy&logoColor=white)
![FAISS](https://img.shields.io/badge/FAISS-<COLOR_CODE>?style=for-the-badge&logoColor=white)

## Overview
This task involves developing a text classification system using a Retrieval-Augmented Generation (RAG) approach with a pre-trained large language model (LLM) to categorize customer support tickets. The system retrieves relevant information from a knowledge base and generates appropriate classification labels for each support ticket. The goal is to accurately classify tickets into predefined categories based on their content.

## Problem Statement
The problem involves the need to efficiently categorize customer support tickets into specific categories based on their content. With a large volume of support requests, manually sorting these tickets is time-consuming and prone to errors. The challenge is to develop an automated text classification system using a Retrieval-Augmented Generation (RAG) approach, leveraging the capabilities of a pre-trained large language model (LLM). The system must accurately retrieve relevant information from a knowledge base and classify each support ticket into the correct category, such as "Login Issues" or "Billing," to streamline the support process and improve response times.

## Technology Stack
1. **Programming Language**: Python
2. **Machine Learning Framework**: PyTorch
3. **Transformers Library**: Hugging Face (AutoTokenizer, AutoModelForCausalLM)
4. **Sentence Embeddings**: SentenceTransformers
5. **Vector Search**: FAISS (Facebook AI Similarity Search)
6. **Data Handling**: NumPy

## Project Description
The project aims to develop an automated text classification system to categorize customer support tickets using a Retrieval-Augmented Generation (RAG) approach. Leveraging a pre-trained large language model (LLM), the system retrieves relevant information from a knowledge base and generates accurate classification labels for support tickets. The system is designed to handle various categories, such as "Login Issues" or "Billing," based on the content of the tickets. The implementation utilizes PyTorch for model operations, Hugging Face's Transformers library for tokenization and model management, SentenceTransformers for embedding generation, and FAISS for efficient vector search. The final deliverables include a Python Jupyter notebook demonstrating the solution, a README file detailing the approach, and a requirements.txt file listing dependencies. The goal is to create an efficient and scalable solution that streamlines the support process by reducing manual ticket classification.

## Approach and Rationale

1. **Large Language Model**: We used the facebook/opt-1.3b model, an open-source LLM, for its balance of performance and efficiency.

2. **Retrieval-Augmented Generation (RAG)**:
   - We created a knowledge base of category descriptions.
   - For each ticket, we retrieve the most relevant category information.
   - This relevant information is then used to augment the prompt for the LLM.

3. **Efficient Retrieval**: We used FAISS, a library for efficient similarity search, to quickly find the most relevant information from our knowledge base.

4. **Sentence Embeddings**: The all-MiniLM-L6-v2 model is used to create embeddings for both the knowledge base and the input tickets, allowing for semantic similarity comparison.

5. **Error Handling**: Comprehensive error handling and logging are implemented to ensure robustness and ease of debugging.

The rationale behind this approach is to combine the strengths of retrieval-based systems (quick access to relevant information) with the generative capabilities of large language models. This allows for more accurate and contextually relevant classifications.
