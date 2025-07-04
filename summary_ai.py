from transformers import pipeline

# Use a small, pre-trained summarization model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    if len(text.strip()) == 0:
        return "No data to summarize."

    try:
        # Trim to avoid long input (which can break on free hosting)
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
