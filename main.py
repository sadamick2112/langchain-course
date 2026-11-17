import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

print("Initializing components...")

embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model="gpt-5")

vectorstore = PineconeVectorStore(index_name=os.environ["INDEX_NAME"], embedding=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# question is the original prompt
# context is what was retrieved -- what we're augmenting the prompt with
prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)


def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain wihtout LCEL.
    Manually retrieves documents, formats them, and generates a response.
    
    Limitations:
    - manual step-by-step execution
    - no built-in streaming support
    - no async support without additional code
    - harder to compose with other chains
    - more verbose and error-prone
    """

    # Step 1: retrieve relevants documents from Pinecone
    docs = retriever.invoke(query)

    # Step 2: Format documents into context string
    context = format_docs(docs)

    # Step 3: Format th eprompt with context and question
    messages = prompt_template.format_messages(context=context, question=query)

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(messages)

    # Step 5: Return the content
    return response.content


def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "What is Pinecone in machine learning?"

    # print("\n" + "=" * 70)
    # print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    # print("=" * 70)
    # result_raw = llm.invoke([HumanMessage(content=query)])
    # print("\nAnswer:")
    # print(result_raw.content)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Raw LLM Invocation (RAG - Naive impl with no LCEL)")
    print("=" * 70)
    print("\nAnswer:")
    print(retrieval_chain_without_lcel(query))