import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup Mock Vector Database
def setup_vector_store():
    print("📦 Building local FAISS vector store...")
    documents = [
        "Linux is a family of open-source Unix-like operating systems based on the Linux kernel.",
        "Kubuntu is an official flavor of the Ubuntu operating system that uses the KDE Plasma Desktop instead of GNOME.",
        "The KDE Plasma Desktop was created by Matthias Ettrich in 1996.",
        "RAG stands for Retrieval-Augmented Generation, combining search with LLMs."
    ]
    # In a real app, these would be chunks loaded from your PDFs
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(documents, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 2})

# 2. Build the Memory-Aware Chains
def build_conversational_rag_chain(retriever):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    # --- STEP A: History-Aware Retriever ---
    # This prompt tells the LLM to rewrite the user's question using the chat history
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- STEP B: Answer Generation ---
    # This prompt takes the retrieved documents and answers the question
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Keep the answer concise.\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- STEP C: Combine them ---
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain

# 3. CLI Application Loop with Memory
def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Please set your OPENAI_API_KEY environment variable.")
        return

    retriever = setup_vector_store()
    rag_chain = build_conversational_rag_chain(retriever)
    
    # This list acts as our Conversation Memory
    chat_history = []
    
    print("\n" + "-"*50)
    print("Welcome to the Conversational RAG CLI!")
    print("Try asking: 'What is Kubuntu?', then follow up with 'Who created its desktop?'")
    print("Type 'exit' to quit.")
    print("-" * 50)

    while True:
        query = input("\n📝 You: ")
        if query.lower() in ['exit', 'quit']:
            break
        if not query.strip():
            continue

        print("🤖 Thinking...")
        
        # Invoke the chain, passing in the current chat history
        response = rag_chain.invoke({
            "input": query,
            "chat_history": chat_history
        })
        
        answer = response["answer"]
        print(f"💬 AI: {answer}")

        # Update the memory for the next turn
        chat_history.extend([
            HumanMessage(content=query),
            AIMessage(content=answer)
        ])

if __name__ == "__main__":
    main()