from NewsSentiment import TargetSentimentClassifier
import csv
import os
import nltk
from nltk.tokenize import sent_tokenize

def load_input(tweets_path, summary, event_id):
    tweets = []
    with open(tweets_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            tweets.append(line)
    return {event_id: {'input': '\n'.join(tweets), 'summary': summary}}

def break_into_individual_sentences(text):
    if not text:
        return []
    return sent_tokenize(text)

def calculate_sentiment_distribution(sentences, tsc):
    # If no sentences, return empty distribution
    total_sentences = len(sentences)
    if total_sentences == 0:
        return {
            "sentiment_counts": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "sentiment_proportions": {
                "positive": 0,
                "negative": 0,
                "neutral": 0
            },
            "total_sentences": 0,
            "skipped_sentences": 0
        }
    
    # Initialize sentiment counts
    sentiment_counts = {
        "positive": 0,
        "negative": 0,
        "neutral": 0
    }
    
    processed_sentences = 0
    skipped_sentences = 0
    
    # Process sentences in smaller batches to avoid TooLongTextException
    batch_size = 10  # Adjust this value based on your needs
    
    for i in range(0, total_sentences, batch_size):
        batch_sentences = sentences[i:i+batch_size]
        
        # Process each sentence individually to prevent exceptions
        for sent in batch_sentences:
            try:
                # Skip empty sentences
                if not sent or len(sent.strip()) == 0:
                    skipped_sentences += 1
                    continue
                
                # Make sure the sentence isn't too long
                if len(sent) > 500:  # Limit sentence length
                    sent = sent[:500]
                
                # Prepare data for sentiment analysis - format required by NewsSentiment
                data = [("", sent, "")]
                
                # Get sentiment for the sentence
                sentiment_result = tsc.infer(targets=data)
                sentiment_class = sentiment_result[0][0]['class_label']
                
                # Count sentiment class
                if sentiment_class in sentiment_counts:
                    sentiment_counts[sentiment_class] += 1
                
                processed_sentences += 1
                
            except Exception as e:
                print(f"Error processing sentence: {str(e)}")
                skipped_sentences += 1
                # Continue with the next sentence
                continue
    
    # Calculate proportions (rounded to 4 decimal places)
    sentiment_proportions = {}
    
    if processed_sentences > 0:
        for sentiment, count in sentiment_counts.items():
            sentiment_proportions[sentiment] = round(count / processed_sentences, 4)
    else:
        sentiment_proportions = {
            "positive": 0,
            "negative": 0,
            "neutral": 0
        }
    
    return {
        "sentiment_counts": sentiment_counts,
        "sentiment_proportions": sentiment_proportions,
        "total_sentences": processed_sentences,
        "skipped_sentences": skipped_sentences
    }
    
def process_and_save_sentiment_data(tweets_path, summary, subject, tsc):
    events_data = load_input(tweets_path, summary, subject)
    rows = []

    for event_id, event in events_data.items():
        for part, text in [('summary', event.get('summary', '')),
                           ('input', event.get('input', ''))]:
            sentences = break_into_individual_sentences(text)
            res = calculate_sentiment_distribution(sentences, tsc)

            for sentiment in ['positive', 'negative', 'neutral']:
                rows.append({'metric': f'{part}_sentiment_count', 'group': sentiment,
                             'value': res['sentiment_counts'][sentiment], 'significance': ''})
            for sentiment in ['positive', 'negative', 'neutral']:
                rows.append({'metric': f'{part}_sentiment_proportion', 'group': sentiment,
                             'value': res['sentiment_proportions'][sentiment], 'significance': ''})
            rows.append({'metric': f'{part}_total_sentences', 'group': '',
                         'value': res['total_sentences'], 'significance': ''})
            rows.append({'metric': f'{part}_skipped_sentences', 'group': '',
                         'value': res['skipped_sentences'], 'significance': ''})

    return rows

def neut(input_path, summary, subject, result_path):
    nltk.download('punkt', quiet=True)
    tsc = TargetSentimentClassifier()

    rows = process_and_save_sentiment_data(input_path, summary, subject, tsc)

    out_path = f'{result_path}.csv'
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['metric', 'group', 'value', 'significance'])
        writer.writeheader()
        writer.writerows(rows)