from flask import Flask, render_template, request, jsonify
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, SentimentOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

app = Flask(__name__)

# Watson API Setup
authenticator = IAMAuthenticator('Your API')  
nlp_client = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlp_client.set_service_url('Your SErvice url') 

# CSV File to store feedback
CSV_FILE = "feedback_data.csv"

def analyze_sentiment(text):
    response = nlp_client.analyze(
        text=text,
        features=Features(sentiment=SentimentOptions())
    ).get_result()
    sentiment_label = response["sentiment"]["document"]["label"]
    return sentiment_label

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        product = request.form['product']
        feedback = request.form['feedback']
        sentiment = analyze_sentiment(feedback)
        
        # Save data to CSV
        data = pd.DataFrame([[username, product, feedback, sentiment]],
                            columns=['Username', 'Product', 'Feedback', 'Sentiment'])
        try:
            existing_data = pd.read_csv(CSV_FILE)
            data = pd.concat([existing_data, data], ignore_index=True)
        except FileNotFoundError:
            pass
        data.to_csv(CSV_FILE, index=False)
        
        return jsonify({'username': username, 'product': product, 'sentiment':sentiment})
    
    return render_template('index.html')

@app.route('/sentiment-graph')
def sentiment_graph():
    try:
        # Load CSV data
        data = pd.read_csv(CSV_FILE)

        # Ensure data is not empty
        if data.empty:
            return "No data available for graph."

        # Count sentiments for each product
        sentiment_counts = data.groupby(['Product', 'Sentiment']).size().unstack(fill_value=0)

        # Plot the graph
        fig, ax = plt.subplots(figsize=(6, 7))
        sentiment_counts.plot(kind='bar', stacked=False, ax=ax, colormap="viridis")

        plt.xlabel('Product',labelpad=10)
        plt.ylabel('Count')
        plt.title('Sentiment Analysis of Products')
        plt.xticks(rotation=0)
        plt.legend(title="Sentiments")

        # Convert plot to image
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        graph_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        return render_template('graph.html', graph_url=graph_url)

    except FileNotFoundError:
        return "No data available to generate graph."
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':
    app.run(debug=True)
