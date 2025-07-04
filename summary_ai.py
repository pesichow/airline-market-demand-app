from transformers import pipeline

# Load a smaller summarization model (lightweight & Streamlit-compatible)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    if len(text.strip()) == 0:
        return "No data to summarize."

    try:
        # Limit input to 1024 characters (to prevent token overflow)
        text = text[:1024]

        summary = summarizer(
            text,
            max_length=120,
            min_length=30,
            do_sample=False
        )
        return summary[0]['summary_text']

    except Exception as e:
        return f"❌ Could not generate summary. Error: {str(e)}"
