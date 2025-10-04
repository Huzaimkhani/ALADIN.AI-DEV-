import os
import json
import requests
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import time
import re
import hashlib
import urllib.parse
from solana.rpc.api import Client
import finnhub

# --- API Keys ---
FINNHUB_KEY = "d34oho9r01qhorbf1uqgd34oho9r01qhorbf1ur0"
COINMARKETCAP_KEY = "425ba6fa-9653-4518-80e2-670c889d38a2"
CURRENCYLAYER_KEY = "22bd072872251a32787e8ab6ec5f2de1"
AIMLAPI_KEY = "66d901f08be441119a4cfda40d4089e1"
SERPER_API = "1c06f769b862ff008429dc55a4873d1b91e76c0a"

# --- Your Solana Wallet Address ---
RECEIVER_WALLET = "EqdQrA4HVc9Y8Lg4pADSaghrxFA4GZPUV3phTaUeQKni"  

# --- Initialize Clients ---
finnhub_client = finnhub.Client(api_key=FINNHUB_KEY)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="Aladin.AI - Financial Research Assistant",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Modern Design ---
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e0e0e0;
        margin-bottom: 1rem;
    }
    .nav-button {
        width: 100%;
        padding: 0.75rem;
        margin: 0.25rem 0;
        border: none;
        border-radius: 10px;
        background: #f8f9fa;
        color: #333;
        font-weight: 500;
        text-align: left;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    .nav-button:hover {
        background: #667eea;
        color: white;
        transform: translateX(5px);
    }
    .nav-button.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    .query-counter {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    .search-box {
        background: white;
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        margin: 2rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease;
        border: 1px solid #e0e0e0;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    .feature-card:hover {
        transform: translateY(-5px);
    }
    .feature-icon {
        font-size: 2rem;
        margin-bottom: 0.5rem;
    }
    .data-table {
        background: white;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    .stButton button {
        width: 100%;
        border-radius: 10px;
        height: 3rem;
        font-weight: 600;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State Initialization ---
if 'selected_view' not in st.session_state:
    st.session_state.selected_view = "home"
if 'search_history' not in st.session_state:
    st.session_state.search_history = []
if 'crypto_data' not in st.session_state:
    st.session_state.crypto_data = None
if 'forex_data' not in st.session_state:
    st.session_state.forex_data = None
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = None
if 'payment_status' not in st.session_state:
    st.session_state.payment_status = {}
if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

# --- Freemium State Initialization ---
FREE_QUERIES_PER_DAY = 1
if 'queries_today' not in st.session_state:
    st.session_state.queries_today = 0
if 'last_query_date' not in st.session_state:
    st.session_state.last_query_date = datetime.now().date().isoformat()

# --- Finance Keywords ---
FINANCE_KEYWORDS = [
    'stock', 'stocks', 'investment', 'investing', 'market', 'financial', 'finance',
    'crypto', 'cryptocurrency', 'bitcoin', 'ethereum', 'forex', 'currency',
    'trading', 'trade', 'portfolio', 'asset', 'assets', 'wealth', 'money',
    'economy', 'economic', 'recession', 'inflation', 'deflation', 'interest rate',
    'fed', 'federal reserve', 's&p', 'nasdaq', 'dow jones', 'sec', 'ipo',
    'earnings', 'dividend', 'yield', 'bull', 'bear', 'volatility', 'liquidity',
    'capital', 'financial planning', 'retirement', '401k', 'ira', 'roth',
    'tax', 'accounting', 'audit', 'balance sheet', 'income statement',
    'cash flow', 'valuation', 'pe ratio', 'eps', 'revenue', 'profit', 'loss',
    'bank', 'loan', 'mortgage', 'credit', 'debt', 'default', 'bankruptcy',
    'merger', 'acquisition', 'takeover', 'private equity', 'venture capital',
    'hedge fund', 'mutual fund', 'etf', 'index fund', 'blue chip', 'penny stock',
    'option', 'futures', 'derivative', 'commodity', 'gold', 'silver', 'oil', 'gas',
    'energy', 'sector', 'industry', 'consumer discretionary', 'consumer staples',
    'healthcare', 'technology', 'financial services', 'utilities', 'real estate',
    'industrial', 'telecom', 'materials'
]

# --- Helper Functions (Keep your existing functions) ---

def is_finance_related(query):
    query_lower = query.lower()
    for keyword in FINANCE_KEYWORDS:
        if re.search(r'\b' + re.escape(keyword) + r'\b', query_lower):
            return True
    if re.search(r'\b[A-Z]{1,5}\b', query):
        return True
    if re.search(r'[A-Z]{3}/[A-Z]{3}', query):
        return True
    if re.search(r'\d+\.?\d*\%', query) or re.search(r'\$\d+', query):
        return True
    return False

def get_live_crypto_data(limit=100):
    crypto_data = []
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {
        'Accepts': 'application/json',
        'X-CMC_PRO_API_KEY': COINMARKETCAP_KEY
    }
    params = {
        'start': '1',
        'limit': str(limit),
        'convert': 'USD'
    }
    try:
        response = requests.get(url, headers=headers, params=params)
        data = response.json()
        if 'data' in data:
            for item in data['data']:
                quote = item['quote']['USD']
                crypto_data.append({
                    'symbol': item['symbol'],
                    'price': quote['price'],
                    'price_chg': quote['percent_change_24h'],
                    'volume_24h': quote['volume_24h'],
                    'vol_chg_24h': quote.get('volume_change_24h', 0),
                    'funding_rate': 0
                })
        time.sleep(0.2)
    except Exception as e:
        st.error(f"Error fetching crypto data: {str(e)}")
    return crypto_data

def get_live_forex_data():
    forex_data = []
    url = f"http://api.currencylayer.com/live?access_key={CURRENCYLAYER_KEY}"
    major_currencies = ['EUR', 'GBP', 'JPY', 'AUD', 'CAD', 'CHF', 'CNY', 'INR', 'MXN', 'BRL']
    try:
        response = requests.get(url)
        data = response.json()
        if 'quotes' in data and data['success']:
            for cur in major_currencies:
                rate = data['quotes'].get(f"USD{cur}", 1)
                forex_data.append({
                    'pair': f"USD/{cur}",
                    'price': rate,
                    'change': 0
                })
                forex_data.append({
                    'pair': f"{cur}/USD",
                    'price': 1 / rate if rate != 0 else 0,
                    'change': 0
                })
        time.sleep(0.2)
    except Exception as e:
        st.error(f"Error fetching forex data: {str(e)}")
    return forex_data

def get_live_stock_data():
    stock_data = []
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK.B', 'JPM', 'V', 
               'UNH', 'MA', 'HD', 'PG', 'DIS', 'PYPL', 'ADBE', 'NFLX', 'INTC', 'CSCO',
               'PEP', 'KO', 'MRK', 'PFE', 'ABT', 'CRM', 'ACN', 'ORCL', 'TMO', 'NKE',
               'MDT', 'DHR', 'UPS', 'LIN', 'TXN', 'QCOM', 'RTX', 'HON', 'LOW', 'IBM',
               'SBUX', 'MMM', 'GE', 'CAT', 'UNP', 'BLK', 'MS', 'GS', 'AXP', 'DE']
    for symbol in symbols:
        try:
            quote_data = finnhub_client.quote(symbol)
            if quote_data['c'] > 0:
                price_chg = ((quote_data['c'] - quote_data['pc']) / quote_data['pc']) * 100 if quote_data['pc'] > 0 else 0
                stock_data.append({
                    'symbol': symbol,
                    'price': quote_data['c'],
                    'change': price_chg,
                    'volume': quote_data['v'] if 'v' in quote_data else 0
                })
            time.sleep(0.2)
        except Exception as e:
            st.error(f"Error fetching data for {symbol}: {str(e)}")
    return stock_data

def search_financial_info(query):
    finance_query = f"finance {query}" if not any(keyword in query.lower() for keyword in FINANCE_KEYWORDS) else query
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": finance_query,
        "num": 5
    })
    headers = {
        'X-API-KEY': SERPER_API,
        'Content-Type': 'application/json'
    }
    try:
        response = requests.post(url, headers=headers, data=payload)
        return response.json()
    except Exception as e:
        st.error(f"Search error: {str(e)}")
        return {"organic": []}

def get_ai_response(query, context):
    url = "https://api.aimlapi.com/v1/chat/completions"
    headers = {
        'Authorization': f'Bearer {AIMLAPI_KEY}',
        'Content-Type': 'application/json'
    }
    system_prompt = """You are a financial research assistant. You ONLY answer questions related to finance,
    investing, stocks, cryptocurrencies, forex, and economic news.
    If a question is not related to these topics, politely decline to answer and explain that you are a finance specialist.
    Provide accurate, concise information based on the provided context. Always cite your sources clearly.
    If you're not sure about something, say so. Format your response with clear sections and use bullet points when appropriate."""
    user_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    payload = json.dumps({
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 1000,
        "temperature": 0.3
    })
    try:
        response = requests.post(url, headers=headers, data=payload)
        response_data = response.json()
        if 'choices' in response_data and len(response_data['choices']) > 0:
            return response_data['choices'][0]['message']['content']
        else:
            return "I apologize, but I'm having trouble processing your request at the moment. Please try again later."
    except Exception as e:
        return f"An error occurred: {str(e)}"

def process_search_results(results, query):
    if 'organic' not in results:
        return "No relevant information found.", []
    organic_results = results['organic']
    context = ""
    sources = []
    for i, result in enumerate(organic_results[:3]):
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        link = result.get('link', '')
        context += f"Source {i+1}: {title}. {snippet}\n\n"
        sources.append({"title": title, "link": link})
    return context, sources

# --- Solana Pay Payment System ---

def generate_payment_request(user_query: str, amount_sol: float = 0.01) -> dict:
    """Generate a unique Solana Pay payment request."""
    reference = hashlib.sha256((st.session_state.session_id + user_query).encode()).hexdigest()[:32]
    label = "Financial Research Assistant"
    message = f"Payment for query: {user_query[:20]}..."
    params = {
        "amount": str(amount_sol),
        "reference": reference,
        "label": label,
        "message": message
    }
    query_string = urllib.parse.urlencode(params)
    solana_pay_url = f"solana:{RECEIVER_WALLET}?{query_string}"
    qr_url = f"https://api.qrserver.com/v1/create-qr-code/?size=300x300&data={urllib.parse.quote(solana_pay_url)}"
    return {
        "reference": reference,
        "solana_pay_url": solana_pay_url,
        "amount_sol": amount_sol,
        "qr_url": qr_url
    }

def verify_solana_payment(tx_signature: str, expected_amount: float, receiver_wallet: str) -> bool:
    """Verify payment on Solana mainnet."""
    try:
        client = Client("https://api.mainnet-beta.solana.com")
        tx = client.get_transaction(tx_signature, encoding="jsonParsed")
        if not tx or not tx.value:
            return False
        for instruction in tx.value.transaction.message.instructions:
            if hasattr(instruction, 'parsed') and 'info' in instruction.parsed:
                info = instruction.parsed['info']
                if info.get('destination') == receiver_wallet:
                    lamports = int(info.get('lamports', 0))
                    sol_received = lamports / 1_000_000_000
                    if abs(sol_received - expected_amount) < 0.0001:
                        return True
        return False
    except Exception as e:
        st.error(f"Payment verification error: {str(e)}")
        return False

def mark_paid(reference: str):
    """Mark a payment reference as paid in session state."""
    st.session_state.payment_status[reference] = True

def is_paid(reference: str) -> bool:
    """Check if a payment reference has been paid."""
    return st.session_state.payment_status.get(reference, False)

def display_answer(response, sources):
    """Helper function to display the final answer and sources."""
    st.subheader("Answer:")
    st.write(response)
    if sources:
        st.subheader("Sources:")
        for i, source in enumerate(sources):
            st.write(f"{i+1}. [{source['title']}]({source['link']})")

# --- Freemium Logic ---
def check_and_update_query_count():
    """Checks if the user has queries left today and updates the count."""
    today_str = datetime.now().date().isoformat()
    
    # Reset counter if it's a new day
    if st.session_state.last_query_date != today_str:
        st.session_state.queries_today = 0
        st.session_state.last_query_date = today_str

    if st.session_state.queries_today < FREE_QUERIES_PER_DAY:
        st.session_state.queries_today += 1
        return True # Free query
    else:
        return False # Paid query required

def get_remaining_free_queries():
    """Returns the number of free queries left for the day."""
    today_str = datetime.now().date().isoformat()
    if st.session_state.last_query_date != today_str:
         # Counter would be reset on next query, but for display, show full quota
        return FREE_QUERIES_PER_DAY
    return max(0, FREE_QUERIES_PER_DAY - st.session_state.queries_today)

# --- Sidebar Navigation ---
with st.sidebar:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 🔮 Aladin.AI")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Navigation
    nav_options = {
        "🏠 Home": "home",
        "💬 Ask Finance Question": "ask",
        "₿ Live Crypto Table": "crypto_table", 
        "💱 Live Forex Table": "forex_table",
        "📈 Live Stock Table": "stock_table"
    }
    
    for label, view in nav_options.items():
        is_active = st.session_state.selected_view == view
        button_class = "nav-button active" if is_active else "nav-button"
        if st.button(label, key=f"nav_{view}"):
            st.session_state.selected_view = view
            st.rerun()
    
    # Freemium Info
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 💎 Freemium Info")
    remaining = get_remaining_free_queries()
    st.markdown(f'<div class="query-counter">Free queries left today: {remaining}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Recent Searches
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📚 Recent Searches")
    if st.session_state.search_history:
        for i, search_query in enumerate(st.session_state.search_history[-5:]):
            if st.button(f"• {search_query}", key=f"history_{i}"):
                st.session_state.selected_view = "ask"
                st.session_state.query = search_query
                st.rerun()
    else:
        st.write("No recent searches")
    
    if st.button("🔄 Refresh Data"):
        st.session_state.crypto_data = None
        st.session_state.forex_data = None
        st.session_state.stock_data = None
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# --- Main Content ---
view = st.session_state.selected_view

if view == "home":
    # Home Page with Aladin.AI Design
    st.markdown('<h1 class="main-header">Discover Magic Search at Aladin.AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Explore Personalized Insights in Crypto, Finance, and Stocks.</p>', unsafe_allow_html=True)
    
    # Main Search Box
    st.markdown('<div class="search-box">', unsafe_allow_html=True)
    st.markdown("### 🧞‍♂️ Talk to Aladin AI")
    query = st.text_input("Enter your financial question:", placeholder="e.g., What's the current trend for Bitcoin?")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("✨ Get Magic Insights", use_container_width=True):
            if query:
                st.session_state.selected_view = "ask"
                st.session_state.query = query
                st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Feature Cards
    st.markdown("### 📊 Explore Markets")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">₿</div>', unsafe_allow_html=True)
        st.markdown("**CRYPTO**")
        st.markdown("Live cryptocurrency data & insights")
        if st.button("Explore Crypto", key="home_crypto"):
            st.session_state.selected_view = "crypto_table"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">💼</div>', unsafe_allow_html=True)
        st.markdown("**FINANCE**")
        st.markdown("Comprehensive financial research")
        if st.button("Ask Questions", key="home_finance"):
            st.session_state.selected_view = "ask"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="feature-card">', unsafe_allow_html=True)
        st.markdown('<div class="feature-icon">📈</div>', unsafe_allow_html=True)
        st.markdown("**STOCK**")
        st.markdown("Real-time stock market data")
        if st.button("View Stocks", key="home_stock"):
            st.session_state.selected_view = "stock_table"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

elif view == "ask":
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("## 💬 Ask a Financial Question")
    
    query = st.text_input("Enter your query:", value=st.session_state.get('query', ''), 
                         placeholder="e.g., What are the best tech stocks to invest in right now?")
    
    if st.button("🔍 Research", use_container_width=True) and query:
        if not is_finance_related(query):
            st.warning("⚠️ This assistant only answers finance-related questions. Please ask about stocks, cryptocurrencies, forex, or other financial topics.")
        else:
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)
            
            # Freemium Check
            is_free_query = check_and_update_query_count()
            
            with st.spinner("🔮 Aladin is searching for magical insights..."):
                search_results = search_financial_info(query)
                context, sources = process_search_results(search_results, query)
                response = get_ai_response(query, context)

                if is_free_query:
                    display_answer(response, sources)
                else:
                    # Payment Integration for Paid Queries
                    payment_info = generate_payment_request(query, amount_sol=0.01)
                    reference = payment_info["reference"]

                    if is_paid(reference):
                        display_answer(response, sources)
                    else:
                        st.subheader("🔒 Pay to Unlock Answer (0.01 SOL)")
                        st.info(f"You have used your {FREE_QUERIES_PER_DAY} free queries for today. Please pay to continue.")
                        col1, col2 = st.columns([1, 1])
                        with col1:
                            st.image(payment_info["qr_url"])
                        with col2:
                            st.markdown("### Payment Options")
                            st.markdown(f"[📱 Pay with Solana Wallet]({payment_info['solana_pay_url']})")
                            st.markdown("---")
                            tx_sig = st.text_input("Paste Transaction Signature:")
                            if st.button("Verify Payment"):
                                with st.spinner("Verifying payment..."):
                                    if verify_solana_payment(tx_sig, 0.01, RECEIVER_WALLET):
                                        mark_paid(reference)
                                        st.success("Payment verified! 🎉")
                                        display_answer(response, sources)
                                    else:
                                        st.error("Payment not found or incorrect. Please try again.")
    st.markdown('</div>', unsafe_allow_html=True)

elif view == "crypto_table":
    st.markdown("## ₿ Live Cryptocurrency Table")
    if st.session_state.crypto_data is None:
        with st.spinner("Loading live crypto data..."):
            st.session_state.crypto_data = get_live_crypto_data(limit=100)
    
    if st.session_state.crypto_data:
        crypto_df = pd.DataFrame(st.session_state.crypto_data)
        crypto_search = st.text_input("🔍 Search Crypto by Symbol:", key="crypto_search")
        if crypto_search:
            crypto_df = crypto_df[crypto_df['symbol'].str.contains(crypto_search.upper(), case=False)]
        
        display_df = crypto_df.copy()
        display_df['Price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
        display_df['24h Change'] = display_df['price_chg'].apply(lambda x: f"{x:+.2f}%")
        display_df['24h Volume'] = display_df['volume_24h'].apply(lambda x: f"${x:,.0f}")
        display_df['Vol Change'] = display_df['vol_chg_24h'].apply(lambda x: f"{x:+.2f}%")
        display_df['Funding Rate'] = display_df['funding_rate'].apply(lambda x: f"{x:.2f}%")
        
        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.dataframe(
            display_df[['symbol', '24h Volume', 'Vol Change', 'Price', '24h Change', 'Funding Rate']],
            column_config={
                "symbol": "Symbol",
                "24h Volume": "Volume (24H)",
                "Vol Change": "Vol Chg 24H",
                "Price": "Price",
                "24h Change": "Price Chg 24H",
                "Funding Rate": "Funding Rate"
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif view == "forex_table":
    st.markdown("## 💱 Live Forex Table")
    if st.session_state.forex_data is None:
        with st.spinner("Loading live forex data..."):
            st.session_state.forex_data = get_live_forex_data()
    
    if st.session_state.forex_data:
        forex_df = pd.DataFrame(st.session_state.forex_data)
        forex_search = st.text_input("🔍 Search Forex by Pair:", key="forex_search")
        if forex_search:
            forex_df = forex_df[forex_df['pair'].str.contains(forex_search.upper(), case=False)]
        
        display_df = forex_df.copy()
        display_df['Price'] = display_df['price'].apply(lambda x: f"{x:.4f}")
        display_df['Change'] = display_df['change'].apply(lambda x: f"{x:+.2f}%")
        
        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.dataframe(
            display_df[['pair', 'Price', 'Change']],
            column_config={
                "pair": "Pair",
                "Price": "Price",
                "Change": "Change"
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

elif view == "stock_table":
    st.markdown("## 📈 Live Stock Table")
    if st.session_state.stock_data is None:
        with st.spinner("Loading live stock data..."):
            st.session_state.stock_data = get_live_stock_data()
    
    if st.session_state.stock_data:
        stock_df = pd.DataFrame(st.session_state.stock_data)
        stock_search = st.text_input("🔍 Search Stock by Symbol:", key="stock_search")
        if stock_search:
            stock_df = stock_df[stock_df['symbol'].str.contains(stock_search.upper(), case=False)]
        
        display_df = stock_df.copy()
        display_df['Price'] = display_df['price'].apply(lambda x: f"${x:,.2f}")
        display_df['Change'] = display_df['change'].apply(lambda x: f"{x:+.2f}%")
        display_df['Volume'] = display_df['volume'].apply(lambda x: f"{x:,.0f}")
        
        st.markdown('<div class="data-table">', unsafe_allow_html=True)
        st.dataframe(
            display_df[['symbol', 'Volume', 'Price', 'Change']],
            column_config={
                "symbol": "Symbol",
                "Volume": "Volume",
                "Price": "Price",
                "Change": "Change"
            },
            hide_index=True,
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)