from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.schema import AIMessage, HumanMessage

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    task="text-generation",
    model_kwargs={
        "max_new_tokens": 512,
        "top_k": 30,
        "temperature": 0.5,
        "repetition_penalty": 1.03,
    },
)

def process_response(full_response):
    response_start = full_response.find("Response:") + len("Response:")
    simplified_response = full_response[response_start:].strip()
    return simplified_response
def process_response2(full_response):
    response_start = full_response.find("Reformatted Response:") + len("Reformatted Response:")
    simplified_response = full_response[response_start:].strip()
    return simplified_response

trading_guide_prompt = PromptTemplate.from_template(
    """
    If the user great you just great him back and tell him if he needs any help without answering anythin else.
    You are a knowledgeable trading assistant specializing in cryptocurrency, financial markets, and investment strategies.
    Please carefully analyze the user's question and provide a clear, concise, and actionable response with specific suggestions if applicable:
    - If the user asks about trading strategies, explain the strategy step-by-step, highlighting key points such as risk management, timing, and tools.
    - If the user asks how much money to invest, offer a well-thought-out suggestion based on general financial principles, and provide specific asset recommendations (e.g., cryptocurrencies, stocks, or ETFs) along with suggested amounts.
    - If the user requests the latest news, Just write " here are the links : "
    
    Do NOT respond to any questions about unrelated fields (e.g., medicine, gaming, or personal life). Simply state: "I'm a trading guide, and I can only provide responses in related fields."
    Ensure your response is easy to understand, actionable, and beneficial for the user.
    Please make sure the response is readable, add spaces, punctuation, and make it easy to understand.
    after providing the answer ask the user if he wants to join our trading group for extra guidance.
    
For example:
example_input = I have $100 to invest. How should I do it?
example_response =
With $100 to invest, here’s a balanced strategy that minimizes risk while offering growth potential:
- **$40 in Bitcoin (BTC)**: As the leading cryptocurrency, Bitcoin is often considered a safe bet within the crypto space.
- **$30 in Ethereum (ETH)**: Ethereum is the second-largest cryptocurrency and has substantial growth potential, especially with ongoing developments in its ecosystem.
- **$20 in Solana (SOL)**: Solana is a high-performance blockchain with increasing adoption, though it carries more risk than BTC or ETH.
- **$10 in a stablecoin like USDT**: Keeping a portion in a stablecoin provides liquidity and helps you protect your capital from market volatility.
This strategy provides a solid mix of stability (BTC, ETH) and growth potential (SOL), with a safety buffer (USDT).
Remember to do your own research, and only invest what you're comfortable risking, especially in volatile markets.


    Question: {input}
    Response:
    
    """
)


prompt_clean = PromptTemplate.from_template("""Please reformat the following response to improve readability, adding spaces, punctuation, and making it easy to understand:
    
    Original Response:
    {input}
    
    Reformatted Response:""")

cleaning_chain = LLMChain(llm=llm, prompt=prompt_clean)
    
    
news_links = [
    "https://www.coindesk.com",
    "https://www.cointelegraph.com",
    "https://www.investing.com/crypto"
]


group_invitation_link = "https://waqarzaka.net/"

trading_chain = LLMChain(llm=llm, prompt=trading_guide_prompt)

st.set_page_config(page_title="Advanced Trading Assistant Chatbot")
st.header("Trading guide Waqar Zaka")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I'm Waqar Zaka your trading assistant. How can I help you?")
    ]


for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        st.chat_message("assistant").markdown(message.content)
    elif isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.content)

user_query = st.chat_input("Type a trading question...")

if user_query and user_query.strip():
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.chat_message("user").markdown(user_query)
    
    response = trading_chain.run({"input": user_query})
    
    if "latest news" in user_query.lower():
        response = "         Here are some reliable news sources you can check:\n"
        for link in news_links:
            response += f"- [News Source]({link})\n"
    
    if "join" in user_query.lower() or "group" in user_query.lower():
        response += f"\n If you'd like more guidance and want to join our trading group, here is the link: [Join the Group]({group_invitation_link})"
    
    if "medicine" in user_query.lower() or "game" in user_query.lower() or "personal" in user_query.lower():
        response = "I'm a trading guide, and I can only provide responses in related fields."
    simplified_response = process_response(response)
    #reformatted_response = cleaning_chain.run({"input":simplified_response}).strip()
    #reformatted_response = process_response(reformatted_response)
    st.session_state.chat_history.append(AIMessage(content=simplified_response))
    st.chat_message("assistant").markdown(simplified_response)
    
    
    
    
    