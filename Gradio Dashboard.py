import os
import pandas as pd
import numpy as np
import warnings
import torch
import gradio as gr
from dotenv import load_dotenv

# 1. SILENCE WARNINGS & LOAD ENV
# This hides the Pydantic/Python 3.14 compatibility warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
load_dotenv()

from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma

# 2. DATA PREPARATION
books = pd.read_csv("books_with_emotions.csv")

# Create high-res thumbnails and handle missing covers
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# 3. VECTOR DATABASE SETUP
# Note: Using Hugging Face because OpenAI requires a paid API key
raw_documents = TextLoader("tagged_description.txt", encoding="utf-8").load()

# Split by newline (one book per chunk)
text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

# Initialize local embeddings (Uses your RTX card if available)
model_kwargs = {'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs=model_kwargs
)

# Create the vector store
db_books = Chroma.from_documents(documents, embeddings)


# 4. RECOMMENDATION LOGIC
def retrieve_semantic_recommendations(
        query: str,
        category: str = "All",
        tone: str = "All",
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:
    # Semantic search
    recs = db_books.similarity_search(query, k=initial_top_k)

    # Clean ISBNs and extract from documents
    books_list = [int(rec.page_content.split()[0].strip('" ')) for rec in recs]

    # Filter the main dataframe
    book_recs = books[books["isbn13"].isin(books_list)].copy()

    # Filter by Category
    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    # Sort by Emotional Tone
    tone_map = {
        "Happy": "joy",
        "Surprising": "surprise",
        "Angry": "anger",
        "Suspenseful": "fear",
        "Sad": "sadness"
    }

    if tone in tone_map:
        book_recs.sort_values(by=tone_map[tone], ascending=False, inplace=True)

    return book_recs.head(final_top_k)


def recommend_books(query: str, category: str, tone: str):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        # Clean description
        desc = str(row["description"])
        truncated_description = " ".join(desc.split()[:30]) + "..."

        # Format authors nicely
        authors_raw = str(row["authors"])
        authors_split = authors_raw.split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = authors_raw

        caption = f"{row['title']} by {authors_str}\n\n{truncated_description}"
        results.append((row["large_thumbnail"], caption))

    return results


# 5. GRADIO DASHBOARD
categories = ["All"] + sorted(books["simple_categories"].unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks(theme=gr.themes.Glass()) as dashboard:
    gr.Markdown("# 📚 Semantic Book Recommender")

    with gr.Row():
        with gr.Column(scale=4):
            user_query = gr.Textbox(
                label="What kind of story are you looking for?",
                placeholder="e.g., A gritty detective novel set in futuristic Tokyo"
            )
        with gr.Column(scale=1):
            category_dropdown = gr.Dropdown(choices=categories, label="Category:", value="All")
            tone_dropdown = gr.Dropdown(choices=tones, label="Vibe/Tone:", value="All")
            submit_button = gr.Button("Find My Next Book", variant="primary")

    gr.Markdown("## Top Recommendations for You")
    output = gr.Gallery(label="Recommendations", columns=4, rows=2, object_fit="contain")

    submit_button.click(
        fn=recommend_books,
        inputs=[user_query, category_dropdown, tone_dropdown],
        outputs=output
    )

if __name__ == "__main__":
    # Launching locally
    dashboard.launch(inbrowser=True)