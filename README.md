# Face Recognition — EfficientNet-B0 Embeddings + pgvector

A face registration, verification, and recognition pipeline built around a fine-tuned EfficientNet-B0 backbone used as a feature extractor, with similarity search backed by Postgres/pgvector on Supabase and served through FastAPI.

*[Baca dalam Bahasa Indonesia](README.id.md)*

## Overview

The system separates two concerns that are easy to conflate: **classification** (what the model was trained on) and **open-set recognition** (identifying anyone, including people never seen during training). It does this the same way most production face-recognition systems do:

1. Train a CNN classifier on a labeled face dataset.
2. Discard the classification head, keep only the convolutional backbone.
3. Use the backbone's penultimate feature vector as a fixed-length embedding.
4. Compare embeddings with cosine similarity instead of relying on the classifier's fixed label set.

That last step is what lets the API register a brand-new person at runtime, without retraining anything.

## How it works

| Step | Flow |
|---|---|
| **Register** | photo → face detection & alignment (MTCNN) → embedding (EfficientNet-B0) → stored in `face_embeddings` (Supabase/pgvector) |
| **Verify (1:1)** | photo + claimed name → embedding → cosine similarity against *that one* stored vector → accept/reject at threshold 0.85 |
| **Recognize (1:N)** | photo → embedding → cosine similarity against *every* stored vector → best match returned |

## Verified results

**Backbone training** — EfficientNet-B0 fine-tuned as a 16-class identity classifier, then the classifier head is dropped and the backbone reused as the embedder.

| Metric | Value |
|---|---|
| Classes | 16 identities |
| Test accuracy | 95% (131 held-out images, seeded split) |
| Embedding dimensionality | 1280 |

**End-to-end pipeline test** — three separate API calls, using photos that never touched the training set:

| Test | Similarity | Result |
|---|---|---|
| Same person, different photo (verification) | 0.8664 | ✓ Verified |
| Same person, same photo (recognition, 1:N) | 1.0000 | ✓ Recognized |
| Different, unregistered person (recognition, 1:N) | 0.0638 | ✗ Not recognized |

![Face recognition pipeline results](images/pipeline-results.png)

The gap between 0.0638 and 0.8664 is the part that matters: it shows the embedding space actually separates identities, rather than collapsing into near-identical vectors regardless of who's in the photo.

## Tech stack

- **PyTorch** + **torchvision** (EfficientNet-B0)
- **facenet-pytorch** (MTCNN face detection & alignment)
- **FastAPI** + **Uvicorn** + **pyngrok** (serving, tunneled out of Google Colab)
- **Supabase** (Postgres + **pgvector** for cosine similarity search)
- **Google Colab** (GPU training + runtime)

## Project structure

```
├── Face_Recognition_Training.ipynb   # trains the classifier, exports the backbone
├── Face_Recognition_Serving.ipynb    # FastAPI app: register / verify / recognize
├── images/
│   └── pipeline-results.png
└── README.md / README.id.md
```

## Running it yourself

**1. Training**
- Open `Face_Recognition_Training.ipynb` in Colab (GPU runtime).
- Point `data_dir` at a folder of face photos organized as `faces_data/<person_name>/*.jpg`.
- Run all cells. Checkpoints are saved to Google Drive automatically.

**2. Set up Supabase**
- Create a free project at [supabase.com](https://supabase.com).
- Grab the **connection string** (Session pooler or Transaction pooler — not "Direct connection", since Colab's network is IPv4-only) from *Project Settings → Database*.

**3. Serving**
- Open `Face_Recognition_Serving.ipynb` in Colab.
- Add two Colab Secrets: `SUPABASE_DB_URL` and `NGROK_AUTHTOKEN` (both also accept manual input as a fallback if left unset).
- Run all cells. The last cell prints a public ngrok URL.
- Open `<that-url>/docs` for an interactive Swagger UI to try all three endpoints without writing any client code.

## Endpoints

| Method | Path | Body | Purpose |
|---|---|---|---|
| `POST` | `/face_register/?name=...` | image file | Register a new face |
| `POST` | `/face_verification/?name=...` | image file | 1:1 check against one claimed identity |
| `POST` | `/face_recognition/` | image file | 1:N search across every registered face |

## Known limitations

- **Training/serving preprocessing differ.** Training resizes raw photos directly; serving runs them through MTCNN detection and alignment first. Both work, but they aren't processing images the exact same way — a dataset with pre-cropped, roughly-centered faces mostly hides this, but it's worth knowing about before pointing this at messier photos.
- **The reported 95% accuracy comes from one seeded train/test split** on a fairly small dataset (16 classes). It's a reasonable signal, not a rigorous benchmark — no cross-validation was run.
- **Serving is meant for local experimentation, not production hosting.** FastAPI + ngrok inside a Colab notebook gives you a real, callable API for testing, but the tunnel dies with the Colab runtime — it's not a deployment target.
- No rate limiting, auth, or duplicate-registration checks on the endpoints. Anyone with the URL can register, verify, or query while the tunnel is live.

## License

MIT — see [LICENSE](LICENSE).
