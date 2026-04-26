import os
import sys
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# ==========================================
# 0. SETUP & PREREQUISITES
# ==========================================
# Install required packages: 
# pip install langchain langchain-openai faiss-cpu tiktoken


if not os.environ.get("OPENAI_API_KEY"):
    print("⚠️  WARNING: OPENAI_API_KEY environment variable not set.")
    print("Please set it in the script or via your terminal to run this bot.")
    sys.exit(1)

# ==========================================
# 1. CREATE MOCK KNOWLEDGE BASE
# ==========================================
# Here, we generate a mock FAQ file for "TechNova Electronics"
def create_mock_data():
    faq_content = """
    TechNova Electronics - Customer Support FAQ
    
    1. Shipping & Delivery:
    Standard shipping takes 3-5 business days. Expedited shipping takes 1-2 business days. 
    Orders placed after 2 PM EST will be processed the next business day. 
    We currently only ship within the United States and Canada.
    
    2. Return Policy:
    Customers can return products within 30 days of receipt for a full refund. 
    Items must be in their original packaging. To initiate a return, email returns@technova.com with your order number.
    Return shipping is free if the item was defective; otherwise, a $10 restocking fee applies.
    
    3. Warranty Information:
    All TechNova laptops come with a standard 1-year limited warranty covering hardware defects. 
    Accidental damage (like spills or drops) is not covered unless you purchased the "NovaCare+" extended warranty.
    
    4. Technical Support:
    For technical issues, please restart your device before contacting us. 
    Our support team is available Monday through Friday, 9 AM to 6 PM EST.
    """
    with open("technova_faq.txt", "w") as f:
        f.write(faq_content)
    return "technova_faq.txt"

# ==========================================
# 2. BUILD THE RAG PIPELINE WITH MEMORY
# ==========================================
def initialize_support_bot():
    print("🔄 Initializing TechNova Support Bot...")
    
    # A. Load and Split Data
    file_path = create_mock_data()
    loader = TextLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    splits = text_splitter.split_documents(docs)
    
    # B. Embed and Store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # C. Initialize LLM
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    # D. Create History-Aware Retriever
    # This prompt tells the LLM to rephrase the user's question based on chat history
    # so it can accurately search the vector database.
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

    # E. Create the Final Answering Chain
    # This prompt guides how the bot should behave and use the retrieved context.
    qa_system_prompt = (
        "You are a helpful and polite customer support agent for TechNova Electronics. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "If you don't know the answer or it's not in the context, politely state that you "
        "don't have that information and offer to connect them to a human agent. "
        "Keep your answers concise and professional. Do not make up information.\n\n"
        "Context: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # F. Combine everything into the final RAG chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# ==========================================
# 3. RUN THE CHAT INTERFACE
# ==========================================
if __name__ == "__main__":
    rag_chain = initialize_support_bot()
    chat_history = [] # This list will store our conversation memory
    
    print("\n✅ Setup Complete! You are now chatting with the TechNova Support Bot.")
    print("Type 'quit' or 'exit' to end the conversation.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Thank you for contacting TechNova. Have a great day!")
            break
            
        # Invoke the chain with the user's input AND the current chat history
        response = rag_chain.invoke({
            "input": user_input,
            "chat_history": chat_history
        })
        
        bot_answer = response["answer"]
        print(f"Bot: {bot_answer}\n")
        
        # Update the chat history so the bot remembers this turn for the next question
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=bot_answer)
        ])