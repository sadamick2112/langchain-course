import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.output_parsers import StrOutputParser

# This adds the ability to create a Callable object that gets one or more values from a dictionary
from operator import itemgetter 



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

def create_retrieval_chain_with_lcel():
    """
    Create retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipoe operator (|)

    Since LangChain will wrap any custom Python function we add to the chain with a RunnableLambda 
    then we get these .invoke(), .stream(), .ainvoke(), .astream(), and .batch() for free:
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs

    - Type safety: Better integratino with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain providees better observability tools
    """
    retrieval_chain = (
        # RunnablePassthrough allows original value ("question") to be passed through but more values to be added 
        # to input dictionary (in this case the "context")
        RunnablePassthrough.assign(
            # 1. the itemgetter is assigned to "context" Callable so that when context is invoked, 
            #    it returns the question value from the dictionary that is passed as input to the chain 
            #    when invoking the chain.
            # 2. the retriever then performs the document retrieval from Pinecone
            # 3. the format_docs just strings together the documents returned as one long string
            # Output: the output of this RunnablePassthrough will be a dictionary containing 
            #     the original question and the context
            context=itemgetter("question") | retriever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return retrieval_chain

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)


if __name__ == "__main__":
    print("Retrieving...")

    # Query
    query = "What is Pinecone in machine learning?"

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 0: Raw LLM Invocation (No RAG)")
    print("=" * 70)
    result_raw = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result_raw.content)

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 1: Raw LLM Invocation (RAG - Naive impl with no LCEL)")
    print("=" * 70)
    print("\nAnswer:")
    print(retrieval_chain_without_lcel(query))

    print("\n" + "=" * 70)
    print("IMPLEMENTATION 2: Use implementation with LCEL (Better Approach)")
    print("=" * 70)
    print("\nAnswer:")
    chain_with_lcel = create_retrieval_chain_with_lcel()
    result_with_lcel = chain_with_lcel.invoke({"question": query})
    print(result_with_lcel)
