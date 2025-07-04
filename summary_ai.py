from transformers import pipeline

def summarize_text(text):
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    result = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return result[0]['summary_text']

