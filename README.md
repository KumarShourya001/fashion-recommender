# Multimodal Fashion Recommender

A multimodal fashion search and recommendation system built on the
[H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)
dataset. It lets you search a catalog of **71,000+ products** by image or text, and build
outfits from items you like — powered by **FashionCLIP** embeddings and **FAISS** vector search.

**Live Demo:** https://huggingface.co/spaces/KrShourya/hm-fashion-recommender

---

## What it does

The app has five ways to explore the catalog:

- **Image Search** — upload a photo of a clothing item and retrieve visually similar products.
- **Text Search** — describe what you want in plain language (e.g. *"black oversized t-shirt"*) and get matching products.
- **Smart Search** — combine a reference image with a text tweak in one query (e.g. upload a jacket and add *"but in black"*), with category filtering and a diversity toggle. Either input also works on its own.
- **Pick & Build Outfit** — search, click the items you like (they persist across searches), and get matching alternatives plus complementary pieces other shoppers bought together, so you can assemble a full outfit.
- **Your Wardrobe** — upload a few photos of clothes you own and optionally describe your style, to get recommendations tuned to your taste.

## Why FashionCLIP

The first version used a generic CLIP model to embed product images. It worked end to end,
but retrieval quality was poor in a specific, measurable way: similarity scores clustered in a
narrow band (~0.93–0.95) across very different garments, so ranking was almost arbitrary, and
color and garment type were poorly separated (a purple hoodie returned pink hoodies and olive
cardigans; a plain black t-shirt returned black dresses).

The cause is that generic CLIP is trained on broad internet images, so "clothing on a white
background" collapses into one dense region of its embedding space.

Switching to **[FashionCLIP](https://huggingface.co/patrickjohncyh/fashion-clip)** — a CLIP
model fine-tuned on fashion product data — fixed this directly. Embeddings became far more
discriminative, retrieved items genuinely match by type and color, and the shared image/text
space enables reliable text-to-image search. The rest of the pipeline (FAISS index, search
logic) stayed identical, which isolated the embedding model as the real quality lever.

---

## How it works

**Search.** Every product image is encoded into a 512-dim FashionCLIP embedding. These are
indexed with FAISS (`IndexFlatIP` over L2-normalized vectors, so inner product equals cosine
similarity). An image query is encoded with FashionCLIP's vision encoder; a text query with its
text encoder — both are matched against the **same** image index, since CLIP places images and
text in a shared space. A combined query (Smart Search) averages the normalized image and text
vectors and re-normalizes, so a text phrase can steer an image result (e.g. the same garment in
a different color). Results can optionally be diversified with an MMR rerank that trades off
similarity to the query against redundancy between results, so one query doesn't return a row of
near-duplicates.

**Outfit building.** For each liked item, the app combines two signals: visually similar items
in the same garment category (FAISS), and items frequently **bought together** with it
(co-occurrence mined from transaction baskets). Co-occurrence is filtered to cut noise — weak
pairs are dropped with a minimum-count threshold, only the top co-occurring items per article
are kept, and matches are restricted to *different* garment categories so they act as genuine
complements while the visual signal covers same-category alternatives.

**Purchase prediction (Kaggle benchmark).** A separate pipeline predicts the next items a
customer will buy, blending recency-weighted repurchase, item co-occurrence, visual similarity,
and popularity. It scores **MAP@12 ≈ 0.021** on the Kaggle H&M benchmark (1.37M customers) —
roughly double a popularity-only baseline. The repurchase signal does most of the work; this is
documented as a clean, explainable baseline rather than a competition-tuned solution.

## The Dataset & Competition

The data comes from the **H&M Personalized Fashion Recommendations** Kaggle competition, where
the task was to predict which articles each customer would purchase in the seven days after the
training period, scored by **Mean Average Precision @ 12 (MAP@12)**. The full dataset (~25 GB)
includes product images, rich article metadata, customer profiles, and over **31 million
transactions**.

For this project, the data was filtered and synced across the three core tables, and only
articles with available images (**71,664**) were used for the embedding-based search. The
processed embeddings, metadata, and images are hosted on a
[Hugging Face dataset](https://huggingface.co/datasets/KrShourya/hm-fashion-recommender)
so the deployed app can load them at startup.

---

## Repository Structure

```
.
├── app.py                         # Gradio app (the deployed demo)
├── data_prep/
│   ├── 01_data_acquisition.ipynb  # Download dataset, convert CSV->Parquet, extract images
│   └── 02_data_filtering.ipynb    # Clean & sync tables, engineer article metadata text
├── notebooks/
│   ├── 03_embeddings.ipynb        # Encode product images with FashionCLIP
│   ├── 04_search.ipynb            # Build FAISS index; image + text search
│   └── 05_kaggle_submission.ipynb # Purchase-prediction pipeline (MAP@12)
└── src/
    ├── paths.py                   # Centralized project paths
    └── utils.py                   # Helper utilities
```

---

## Tech Stack

- **Models / ML:** FashionCLIP (Hugging Face Transformers, PyTorch), FAISS vector search
- **Data:** Pandas, NumPy — 31M+ transactions, 71K+ products
- **App / Deployment:** Gradio, Hugging Face Spaces, Hugging Face Datasets
- **Environment:** Google Colab (GPU) for embedding generation

---

## Running Locally (brief)

The app loads the model, embeddings, metadata, and images from the hosted Hugging Face dataset,
so it can be run directly:

```bash
pip install torch transformers faiss-cpu gradio pandas numpy pillow huggingface_hub
python app.py
```

The notebooks are intended to run on Google Colab with the dataset mounted from Drive; paths are
centralized in `src/paths.py`.

---

## Notes

This is a personal project. The multimodal search and outfit features are the main focus; the
Kaggle purchase-prediction pipeline is an explainable baseline built to engage with the original
competition task. The switch from generic CLIP to FashionCLIP, and the choice of
transaction-based signals over embedding similarity for purchase prediction, are documented in
the notebooks as deliberate, measured design decisions.

---

## Author

**Kumar Shourya** — B.Tech CSE, NIT Patna
[GitHub](https://github.com/KumarShourya001) · [LinkedIn](https://linkedin.com/in/kumar-shourya)
