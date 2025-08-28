# summarizer.py
from transformers import pipeline

# Load the summarization pipeline from Hugging Face
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_text(text, max_length=130, min_length=30):
    """
    Summarizes the input text using a pretrained model.
    
    Args:
        text (str): Input article/text.
        max_length (int): Max length of the summary.
        min_length (int): Minimum length of the summary.
    
    Returns:
        str: Summarized text.
    """
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

if __name__ == "__main__":
    # Example input
    text = """Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    Leading AI textbooks define the field as the study of "intelligent agents":
    any device that perceives its environment and takes actions that maximize
    its chance of successfully achieving its goals."""
    
    print("Original Text:\n", text)
    print("\nSummary:\n", summarize_text(text))
