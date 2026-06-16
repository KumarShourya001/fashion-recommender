"""
Multimodal Fashion Recommender
Image / text product search and outfit recommendation over the H&M catalog,
built on FashionCLIP embeddings and FAISS vector search.

Author: Kumar Shourya
GitHub: https://github.com/KumarShourya001
"""

import os
import json
import zipfile
import numpy as np
import pandas as pd
import torch
import faiss
import gradio as gr
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from huggingface_hub import hf_hub_download

REPO_ID = "KrShourya/hm-fashion-recommender"
MODEL_ID = "patrickjohncyh/fashion-clip"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOP_K = 8
PER_ITEM = 4
NEIGHBOR_POOL = 200
PICK_POOL = 60
PICK_PAGE = 12
IMG_DIR = "/tmp/images"

# guidance shown on the image-upload tabs so users get good results
IMAGE_TIPS = (
    "**For best results, upload a clean product-style photo:**\n"
    "- A single clothing item, ideally on a plain / light background\n"
    "- The garment centered and filling most of the frame\n"
    "- Well-lit, front-facing, minimal clutter or people in the shot\n\n"
    "_The catalog is made of studio product shots, so flat-lay or catalog-style "
    "images match far better than mirror selfies or busy real-world photos._"
)

print("Loading FashionCLIP...")
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()

print("Downloading catalog data...")
emb_path = hf_hub_download(REPO_ID, "fclip_image_embeddings.npy", repo_type="dataset")
meta_path = hf_hub_download(REPO_ID, "articles_with_images.parquet", repo_type="dataset")
cooc_path = hf_hub_download(REPO_ID, "top_cooc.json", repo_type="dataset")

embeddings = np.load(emb_path).astype("float32")
catalog = pd.read_parquet(meta_path)
catalog["article_id"] = catalog["article_id"].astype(str).str.zfill(10)
with open(cooc_path) as f:
    cooccurrence = json.load(f)

article_to_row = {aid: i for i, aid in enumerate(catalog["article_id"])}
row_to_article = {i: aid for aid, i in article_to_row.items()}
categories = catalog["garment_group_name"].astype(str).tolist()
category_choices = ["All"] + sorted(set(categories))

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"FAISS index ready with {index.ntotal} products")

print("Preparing product images...")
zip_path = hf_hub_download(REPO_ID, "hm_images.zip", repo_type="dataset")
if not os.path.exists(IMG_DIR):
    os.makedirs(IMG_DIR, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(IMG_DIR)
print("Startup complete")

# ------------------------------- helpers ---------------------------------

def image_path(article_id):
    article_id = str(article_id).zfill(10)
    return os.path.join(IMG_DIR, article_id[:3], f"{article_id}.jpg")

def encode_image(pil_image):
    inputs = processor(images=[pil_image.convert("RGB")], return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        vision_out = model.vision_model(pixel_values=inputs["pixel_values"])
        features = model.visual_projection(vision_out.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")

def encode_text(text):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_out = model.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        features = model.text_projection(text_out.pooler_output)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")

def to_gallery(rows):
    items = []
    for r in rows:
        record = catalog.iloc[r]
        path = image_path(record["article_id"])
        if os.path.exists(path):
            items.append((path, str(record["article_metadata"])[:50]))
    return items

# ------------------------------- search ----------------------------------

def search_by_image(pil_image):
    if pil_image is None:
        return []
    query = encode_image(pil_image)
    _, idx = index.search(query, TOP_K)
    return to_gallery(idx[0])

def search_by_text(query_text):
    if not query_text or not query_text.strip():
        return []
    query = encode_text(query_text)
    _, idx = index.search(query, TOP_K)
    return to_gallery(idx[0])

# --------------------------- smart multimodal search ---------------------

def mmr_rerank(query_vec, candidate_rows, k, diversity=0.5):
    # rerank to balance relevance to the query against variety, so results
    # are not a row of near-identical items
    if not candidate_rows:
        return []
    cand = np.array(candidate_rows)
    cand_embs = embeddings[cand]
    query_sim = cand_embs @ query_vec.reshape(-1)
    chosen = []
    chosen_mask = np.zeros(len(cand), dtype=bool)
    k = min(k, len(cand))
    while len(chosen) < k:
        if not chosen:
            pick = int(np.argmax(query_sim))
        else:
            redundancy = (cand_embs @ cand_embs[chosen_mask].T).max(axis=1)
            score = (1 - diversity) * query_sim - diversity * redundancy
            score[chosen_mask] = -np.inf
            pick = int(np.argmax(score))
        chosen.append(int(cand[pick]))
        chosen_mask[pick] = True
    return chosen

def smart_search(pil_image, query_text, category, result_count, diversify):
    result_count = int(result_count)
    image_vec = encode_image(pil_image)[0] if pil_image is not None else None
    text_vec = encode_text(query_text)[0] if (query_text and query_text.strip()) else None

    if image_vec is None and text_vec is None:
        return []

    # average the two normalized vectors so an image can be steered by a text tweak
    if image_vec is not None and text_vec is not None:
        query_vec = (image_vec + text_vec) / 2.0
        query_vec = query_vec / np.linalg.norm(query_vec)
    elif image_vec is not None:
        query_vec = image_vec
    else:
        query_vec = text_vec

    query = query_vec.reshape(1, -1).astype("float32")
    needs_pool = category != "All" or diversify
    pool = max(result_count * 8, NEIGHBOR_POOL) if needs_pool else result_count
    pool = min(pool, index.ntotal)
    _, idx = index.search(query, pool)
    candidates = [int(i) for i in idx[0]]

    if category != "All":
        candidates = [i for i in candidates if categories[i] == category]

    if diversify:
        rows = mmr_rerank(query_vec, candidates, result_count)
    else:
        rows = candidates[:result_count]
    return to_gallery(rows)

# --------------------------- pick & build outfit -------------------------

def search_for_picking(query_text, page=0):
    if not query_text or not query_text.strip():
        return [], [], 0
    query = encode_text(query_text)
    _, idx = index.search(query, PICK_POOL)
    candidates = [int(i) for i in idx[0]]
    rows, gallery = [], []
    start = (page * PICK_PAGE) % len(candidates)
    for offset in range(PICK_PAGE):
        i = candidates[(start + offset) % len(candidates)]
        record = catalog.iloc[i]
        path = image_path(record["article_id"])
        if os.path.exists(path):
            rows.append(i)
            gallery.append((path, str(record["article_metadata"])[:40]))
    return gallery, rows, page

def shuffle_pick_results(query_text, page):
    next_page = page + 1
    return search_for_picking(query_text, next_page)

def liked_summary(liked_rows):
    if not liked_rows:
        return "**Liked items:** none yet"
    names = [str(catalog.iloc[r]["article_metadata"])[:25] for r in liked_rows]
    return f"**Liked items ({len(liked_rows)}):** " + ", ".join(names)

def toggle_like(liked_rows, visible_rows, evt: gr.SelectData):
    clicked = visible_rows[evt.index]
    if clicked in liked_rows:
        liked_rows.remove(clicked)
    else:
        liked_rows.append(clicked)
    return liked_rows, liked_summary(liked_rows)

def clear_selections():
    return [], liked_summary([]), [], 0

def same_category_neighbors(row):
    """Visually similar items within the same garment category."""
    category = categories[row]
    query = embeddings[row:row + 1]
    _, idx = index.search(query, NEIGHBOR_POOL)
    return [int(i) for i in idx[0] if int(i) != row and categories[int(i)] == category]

def cooccurrence_items(row):
    """Items frequently bought alongside this one (often complementary categories)."""
    aid = row_to_article[row]
    co_ids = cooccurrence.get(aid, [])
    return [article_to_row[c] for c in co_ids if c in article_to_row]

def build_outfit(liked_rows, page):
    if not liked_rows:
        return []
    suggestions = []
    seen = {row_to_article[r] for r in liked_rows}

    for r in liked_rows:
        visual = same_category_neighbors(r)
        complementary = cooccurrence_items(r)
        per_item_added = 0

        # try 2 co-occurring (complementary) items first
        if complementary:
            start_c = (page * 2) % len(complementary)
            off = 0
            while per_item_added < 2 and off < len(complementary):
                cand = complementary[(start_c + off) % len(complementary)]
                aid = row_to_article[cand]
                if aid not in seen:
                    suggestions.append(cand); seen.add(aid); per_item_added += 1
                off += 1

        # fill the rest (up to PER_ITEM) with same-category visual matches
        if visual:
            start_v = (page * PER_ITEM) % len(visual)
            off = 0
            while per_item_added < PER_ITEM and off < len(visual):
                cand = visual[(start_v + off) % len(visual)]
                aid = row_to_article[cand]
                if aid not in seen:
                    suggestions.append(cand); seen.add(aid); per_item_added += 1
                off += 1

    suggestions.sort(key=lambda r: categories[r])
    return to_gallery(suggestions)

def build_outfit_fresh(liked_rows, page):
    return build_outfit(liked_rows, 0), 0

def shuffle_suggestions(liked_rows, page):
    next_page = page + 1
    return build_outfit(liked_rows, next_page), next_page

# ------------------------------ wardrobe ---------------------------------

def recommend_from_wardrobe(files, style_description):
    vectors = []
    if files:
        for f in files:
            try:
                path = f.name if hasattr(f, "name") else f
                vectors.append(encode_image(Image.open(path).convert("RGB"))[0])
            except Exception:
                continue

    style_vector = None
    if style_description and style_description.strip():
        style_vector = encode_text(style_description)[0]

    if not vectors and style_vector is None:
        return []

    queries = vectors if vectors else [style_vector]
    suggestions, seen = [], set()
    for vector in queries:
        if style_vector is not None and vectors:
            vector = (vector + style_vector) / 2.0
            vector = vector / np.linalg.norm(vector)
        query = vector.reshape(1, -1).astype("float32")
        _, idx = index.search(query, PER_ITEM + 5)
        added = 0
        for i in idx[0]:
            aid = row_to_article[i]
            if aid not in seen:
                suggestions.append(i)
                seen.add(aid)
                added += 1
            if added >= PER_ITEM:
                break
    suggestions.sort(key=lambda r: categories[r])
    return to_gallery(suggestions)

# --------------------------------- ui ------------------------------------

theme = gr.themes.Soft(primary_hue="slate", secondary_hue="gray", neutral_hue="slate")

with gr.Blocks(title="Multimodal Fashion Recommender", theme=theme) as demo:
    gr.Markdown(
        """
        # Multimodal Fashion Recommender
        Search and personalize across **71,000+ H&M products** using image queries,
        natural-language search, and taste-based outfit recommendations.
        <br><sub>FashionCLIP embeddings · FAISS vector search · built by
        <a href="https://github.com/KumarShourya001">Kumar Shourya</a></sub>
        """
    )

    with gr.Tab("Image Search"):
        gr.Markdown("Upload a photo of a clothing item to retrieve visually similar products.")
        gr.Markdown(IMAGE_TIPS)
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="pil", label="Your image")
                image_button = gr.Button("Find Similar Products", variant="primary")
            with gr.Column(scale=2):
                image_results = gr.Gallery(label="Similar Products", columns=4, height="auto")
        image_button.click(search_by_image, inputs=image_input, outputs=image_results)

    with gr.Tab("Text Search"):
        gr.Markdown("Describe what you are looking for in plain language.")
        text_input = gr.Textbox(label="Describe what you want", placeholder="e.g. black oversized t-shirt")
        gr.Examples(
            examples=[
                "black oversized t-shirt", "floral summer dress", "blue denim jacket",
                "white sneakers", "beige knit sweater",
            ],
            inputs=text_input,
        )
        text_button = gr.Button("Search", variant="primary")
        text_results = gr.Gallery(label="Matching Products", columns=4, height="auto")
        text_button.click(search_by_text, inputs=text_input, outputs=text_results)

    with gr.Tab("Smart Search"):
        gr.Markdown(
            "Combine a reference image with a text tweak in one query - upload a product photo and add "
            "words like \"but in black\" or \"long sleeve version\" to steer the results. "
            "Either input works on its own too. Filter by category and control how many results you see."
        )
        gr.Markdown(IMAGE_TIPS)
        with gr.Row():
            with gr.Column(scale=1):
                smart_image = gr.Image(type="pil", label="Reference image (optional)")
                smart_text = gr.Textbox(label="Text refinement (optional)",
                                        placeholder="e.g. same style but in red")
                smart_category = gr.Dropdown(category_choices, value="All", label="Category")
                smart_count = gr.Slider(4, 24, value=TOP_K, step=4, label="Number of results")
                smart_diversify = gr.Checkbox(value=True, label="Diversify results")
                smart_button = gr.Button("Search", variant="primary")
            with gr.Column(scale=2):
                smart_results = gr.Gallery(label="Results", columns=4, height="auto")
        smart_button.click(
            smart_search,
            inputs=[smart_image, smart_text, smart_category, smart_count, smart_diversify],
            outputs=smart_results,
        )

    with gr.Tab("Pick & Build Outfit"):
        gr.Markdown(
            "Search for products and **click the ones you like** - a jacket, a t-shirt, anything. "
            "Picks persist across searches so you can assemble a set. "
            "Use **Shuffle Results** to browse more options for the same search. "
            "Each liked item then gets matching alternatives plus complementary pieces "
            "people bought with it, and **Shuffle Suggestions** swaps them for fresh options."
        )
        liked_state = gr.State([])
        visible_rows_state = gr.State([])
        page_state = gr.State(0)
        pick_page_state = gr.State(0)

        with gr.Row():
            pick_search_input = gr.Textbox(label="Search products", placeholder="e.g. denim jacket", scale=4)
            pick_search_button = gr.Button("Search", variant="primary", scale=1)

        pick_gallery = gr.Gallery(label="Click items you like", columns=6, height="auto", allow_preview=False)
        liked_display = gr.Markdown(liked_summary([]))

        with gr.Row():
            shuffle_results_button = gr.Button("Shuffle Results")
            outfit_button = gr.Button("Build My Outfit", variant="primary")
            shuffle_button = gr.Button("Shuffle Suggestions")
            clear_button = gr.Button("Clear Selections")

        outfit_results = gr.Gallery(label="Outfit Suggestions", columns=4, height="auto")

        pick_search_button.click(
            lambda q: search_for_picking(q, 0),
            inputs=pick_search_input,
            outputs=[pick_gallery, visible_rows_state, pick_page_state]
        )
        shuffle_results_button.click(
            shuffle_pick_results,
            inputs=[pick_search_input, pick_page_state],
            outputs=[pick_gallery, visible_rows_state, pick_page_state]
        )
        pick_gallery.select(toggle_like, inputs=[liked_state, visible_rows_state],
                            outputs=[liked_state, liked_display])
        outfit_button.click(build_outfit_fresh, inputs=[liked_state, page_state],
                            outputs=[outfit_results, page_state])
        shuffle_button.click(shuffle_suggestions, inputs=[liked_state, page_state],
                             outputs=[outfit_results, page_state])
        clear_button.click(clear_selections,
                           outputs=[liked_state, liked_display, outfit_results, page_state])

    with gr.Tab("Your Wardrobe"):
        gr.Markdown(
            "Upload **2-3 photos** of clothes you own or like, optionally describe your style, "
            "and get matching suggestions for each piece."
        )
        gr.Markdown(IMAGE_TIPS)
        with gr.Row():
            with gr.Column(scale=1):
                wardrobe_input = gr.File(label="Upload your clothing photos",
                                         file_count="multiple", file_types=["image"])
                style_input = gr.Textbox(label="Describe your style (optional)",
                                         placeholder="e.g. minimal, neutral colors, casual")
                wardrobe_button = gr.Button("Get Recommendations", variant="primary")
            with gr.Column(scale=2):
                wardrobe_results = gr.Gallery(label="Recommended For You", columns=4, height="auto")
        wardrobe_button.click(recommend_from_wardrobe,
                              inputs=[wardrobe_input, style_input], outputs=wardrobe_results)

    gr.Markdown(
        "<br><center><sub>Personal project by Kumar Shourya · "
        "<a href='https://github.com/KumarShourya001'>GitHub</a> · "
        "<a href='https://linkedin.com/in/kumar-shourya'>LinkedIn</a> · "
        "FashionCLIP + FAISS over 71K+ products</sub></center>"
    )

demo.launch()