"""Traditional RAG Pipeline using LangChain, OpenAI, and FAISS."""

import time
from typing import List, Dict, Any

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


class TraditionalRAG:
    """Traditional RAG system using vector similarity search."""

    def __init__(
        self,
        openai_api_key: str,
        model_name: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.openai_api_key = openai_api_key
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            api_key=openai_api_key
        )

        self.llm = ChatOpenAI(
            model=model_name,
            temperature=0,
            api_key=openai_api_key
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

        self.vectorstore = None
        self.qa_chain = None

    def load_documents(self, file_path: str) -> List[Document]:
        """Load documents from file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        chunks = self.text_splitter.split_text(content)

        documents = [
            Document(page_content=chunk, metadata={"source": file_path, "chunk_id": i})
            for i, chunk in enumerate(chunks)
        ]

        print(f"Loaded {len(documents)} chunks from {file_path}")
        return documents

    def build_index(self, documents: List[Document]) -> None:
        """Build FAISS vector index from documents."""
        print("Building FAISS index...")
        start_time = time.time()

        self.vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

        build_time = time.time() - start_time
        print(f"FAISS index built in {build_time:.2f} seconds")

        self._create_qa_chain()

    def _create_qa_chain(self) -> None:
        """Create the QA chain using modern LCEL (no RetrievalQA)."""
        prompt_template = """You are a helpful AI assistant answering questions about the CloudStore API documentation.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, say so - don't make up information.

Context:
{context}

Question: {question}

Answer: """

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 4})

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.qa_chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG system."""
        if not self.qa_chain:
            raise ValueError("Index not built. Call build_index() first.")

        print(f"\nQuerying Traditional RAG: {question}")
        start_time = time.time()

        answer = self.qa_chain.invoke(question)
        source_docs = self.vectorstore.similarity_search(question, k=4)

        query_time = time.time() - start_time

        return {
            "answer": answer,
            "source_documents": source_docs,
            "metrics": {
                "query_time": query_time,
                "num_source_chunks": len(source_docs),
                "answer_tokens": len(answer.split()),
                "retrieval_method": "vector_similarity"
            }
        }

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search without generation."""
        if not self.vectorstore:
            raise ValueError("Index not built. Call build_index() first.")

        return self.vectorstore.similarity_search(query, k=k)

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk."""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"Index saved to {path}")

    def load_index(self, path: str) -> None:
        """Load FAISS index from disk."""
        self.vectorstore = FAISS.load_local(
            path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        self._create_qa_chain()
        print(f"Index loaded from {path}")
