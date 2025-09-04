import os
import json
import requests
import google.generativeai as genai

import streamlit as st
from linkpreview import link_preview
#from dotenv import load_dotenv
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_community.document_loaders import UnstructuredURLLoader, SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

#load_dotenv()
SERPER_API = "f2a8d5ba6723fd164caf5d2a0d48e29327a554f5"
OPENAI_API_KEY = os.getenv('API_KEY')
GOOGLE_API_KEY="1270a6bcd5c24c05bd93fca036082408"  # Updated to AI/ML API key

def search(query):
    """
    search the information through google
    """
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })

    headers = {
        'X-API-KEY': SERPER_API,
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload)
    response_data = response.json()
    return response_data

def get_gemini_response(query, prompt):
    """
    Call AI/ML API instead of Google Gemini
    """
    base_url = "https://api.aimlapi.com/v1"
    url = f"{base_url}/chat/completions"
    headers = {
        'Authorization': f'Bearer {GOOGLE_API_KEY}',
        'Content-Type': 'application/json'
    }
    payload = json.dumps({
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": query}
        ],
        "max_tokens": 500
    })

    response = requests.post(url, headers=headers, data=payload)
    response_data = response.json()
    
    if 'choices' in response_data and len(response_data['choices']) > 0:
        return response_data['choices'][0]['message']['content']
    else:
        raise Exception(f"Error in AI/ML API response: {response_data}")

def find_relevant_articles(response, query):
    """
    llm is used to find the relevant articles to the user query from response
    """
    response_data = json.dumps(response)
    
    template = """
    You are the best researcher of all time. you are extremely good at finding the relevant articles to the query;
    {response_data}
    Above is the list of search results of articles for the query: {query}.
    Please rank the best 3 articles from the list, return ONLY a valid JSON array of the urls (e.g., ["url1", "url2", "url3"]), do not include any other information or text.
    """.format(response_data=response_data, query=query)
    
    articles_urls = get_gemini_response(query, template)
    url_list = json.loads(articles_urls)
    return url_list
    
def scrape(articles_urls):
    """
    scrapes the information from the article 
    """
    loader = UnstructuredURLLoader(urls=articles_urls)
    data = loader.load()
    return data

def get_information_from_urls(article_urls):
    """
    Scrapes the text content from the article URLs using Selenium.
    """
    options = Options()
    options.headless = True
    
    # Initialize the WebDriver
    service = Service(ChromeDriverManager(driver_version='122.0.6261.112').install())
    driver = webdriver.Chrome(service=service, options=options)
    
    information = []
   
    try:
        for url in article_urls:
            try:
                driver.get(url)
                driver.implicitly_wait(10)
                text_content = driver.find_element(By.TAG_NAME, 'body').text
                information.append(text_content)
            except Exception as e:
                print(f'Error getting information {url}: {e}')
                continue
    finally:
        driver.quit()

    return information

def summarize(information):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n",],
        chunk_size=1000,
        chunk_overlap=20,
        length_function=len,
    )
    texts = information if len(information) < 2000 else text_splitter.create_documents(information)
    
    template = """
    write a concise summary of the articles
    "{texts}"
    CONCISE SUMMARY:
    """.format(texts=texts)
    response = get_gemini_response("", template)
    resp = response.strip()
    return resp

def main():
    st.title('Research Agent')
    query = st.text_input('Enter your query:')
    if st.button('Search'):
        response = search(query)
        url_list = find_relevant_articles(response, query)
        with st.expander("Articles"):
            for url in url_list:
                if url:
                    try:
                        # Generate a preview for the URL
                        preview = link_preview(url)
                        title = preview.title
                        description = preview.description
                        image = preview.image
                        
                        # Display the preview information
                        if title:
                            st.markdown(f'**Title:** {title}', unsafe_allow_html=True)
                        if description:
                            st.markdown(f'**Description:** {description}', unsafe_allow_html=True)
                        if image:
                            st.image(image, caption='Preview Image', width=100)
                        
                        st.markdown(f'**URL:** [{url}]({url})', unsafe_allow_html=True)
                    
                    except Exception as e:
                        print(str(e))
                
        data = scrape(url_list)
        print(data)
        # data = get_information_from_urls(url_list)
        summary = summarize(data)
        st.markdown("**Answer**")
        st.write(summary)

if __name__ == '__main__':
    main()