from flask import Flask, request, jsonify
from ai_agent.AGENT import (
    search,
    get_gemini_response,
    find_relevant_articles,
    scrape,
    get_information_from_urls,
    summarize )
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return "Flask server is running!"

@app.route('/search', methods=['POST'])
def search_summary():
    data = request.get_json()
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Query not provided"}), 400

    try:
        # Process the query
        response = search(query)
        url_list = find_relevant_articles(response, query)
        scraped_data = scrape(url_list)
        summary = summarize(scraped_data)
        result_with_links = [
            {
                "summary": summary,
                "source_links": url_list  # Add the URLs here
            }
        ]
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
