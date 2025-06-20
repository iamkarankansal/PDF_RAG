from ..db.collections.files import files_collection
from bson import ObjectId
from openai import OpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore

load_dotenv()
client = OpenAI()


async def process_file(id: str, file_path: str, question: str):
    # Step 0: Mark as processing
    await files_collection.update_one(
        {"_id": ObjectId(id)},
        {"$set": {"status": "processing"}}
    )

    # Step 1: Load and split the PDF
    loader = PyPDFLoader(file_path=file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
    )
    split_docs = text_splitter.split_documents(docs)

    # Step 2: Create embeddings and store in Qdrant
    embedder = OpenAIEmbeddings(model="text-embedding-3-large")

    vector_store = QdrantVectorStore.from_documents(
        documents=split_docs,
        url="http://qdrant:6333",
        collection_name=f"""pdf_rag_{id}""",
        embedding=embedder
    )

    # Step 3: Retrieve relevant chunks
    relevant_chunks = vector_store.similarity_search(query=question)

    context = "\n\n".join([doc.page_content for doc in relevant_chunks])

    SYSTEM_PROMPT = f"""
    You are a helpful AI assistant who responds based on the following context.

    Context:
    {context}
    """

    # Step 4: Generate answer using OpenAI
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
    )
    output_text = response.choices[0].message.content

    # Step 5: Update DB with result
    await files_collection.update_one(
        {"_id": ObjectId(id)},
        {
            "$set": {
                "status": "processed",
                "result": output_text
            }
        }
    )
