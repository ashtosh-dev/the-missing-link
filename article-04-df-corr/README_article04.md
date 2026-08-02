# Article 4A — The Math You Already Know Is Running Your RAG Pipeline

**Inner products, norms, and orthogonality — and the measuring instrument you have been using since your first physics class without knowing it had a name**

📝 [Read on Medium](https://medium.com/@ashtosh.shenoy/the-math-you-already-know-is-running-your-rag-pipeline-298eb4efa7a1)

---

## What This Article Covers

The dot product appears in three places across your coursework and AIML stack — each time with different notation, different context, and no pointer back to the others:

- **Linear algebra class** — the weights formula `cⱼ = (y·uⱼ) / (uⱼ·uⱼ)` for expanding vectors in an orthogonal basis
- **ML libraries** — `cosine_similarity(u, v)` in every vector database and NLP pipeline
- **Transformer attention** — `score(q, k) = qᵀk / √d`, the operation deciding which tokens attend to which

All three are the same operation. This article draws that line explicitly — through the geometry, the code, and the RAG pipeline payoff.

**Key concepts:** inner product, norm, unit vectors, orthogonality, orthogonal sets, orthonormal sets, scalar projection, weights formula, cosine similarity, scaled dot-product attention, pre-normalized embeddings, Inner Product (IP) distance.

---

## File Structure

```
article-04-inner-products/
│
├── inner_product_visualization.py   # All 6 figures — run this to generate everything
│
├── Visuals/
│   ├── fig0_three_disguises.png         # Same dot product: weights formula, cosine sim, attention
│   ├── fig1_weights_two_operations.png  # Geometric derivation of cⱼ in three panels
│   ├── fig2_orthogonality.png           # Three cases: together, perpendicular, apart
│   ├── fig3_cosine_bridge.png           # Physics formula → ML cosine_similarity
│   ├── fig4_attention_connection.png    # Weights formula vs attention score + bar chart
│   └── fig5_rag_pipeline.png            # Query vs document vectors, similarity ranking
│
└── README.md
```

---

## How to Run

```bash
cd article-04-inner-products
python inner_product_visualization.py
```

All 6 figures are saved to `Visuals/`. Expected terminal output:

```
Saved: Visuals/fig0_three_disguises.png
[fig0] weights formula cj = 0.7000
[fig0] cosine similarity  = 0.923077...
[fig0] attention score    = 0.762...

Saved: Visuals/fig1_weights_two_operations.png
[fig1] scalar projection = 1.600000
[fig1] c_j               = 0.800000
[fig1] c_j * u           = [1.6 0. ]
[fig1] y x-component     = 1.600000  (should match c_j*u[0] = 1.600000)

Saved: Visuals/fig2_orthogonality.png
[fig2] pair 1: u.v = 5.3600
[fig2] pair 2: u.v = 0.0000
[fig2] pair 3: u.v = -2.9600

Saved: Visuals/fig3_cosine_bridge.png
[fig3] dot(u,v)          = 6.320000
[fig3] ||u|| * ||v||     = 6.735357
[fig3] cosine_similarity = 0.938354
[fig3] theta             = 20.2799 degrees

Saved: Visuals/fig4_attention_connection.png
[fig4] Attention scores (softmax) for query 'bank':
  river     : 0.253417
  money     : 0.206164
  sat       : 0.186583
  the       : 0.178372
  on        : 0.175717

Saved: Visuals/fig5_rag_pipeline.png
[fig5] Cosine similarities:
  Attention Mechanism                : 0.998294
  Self-Attention                     : 0.991511
  K-Means Clustering                 : 0.972839
  Convolutional Nets                 : 0.903874
  Decision Trees                     : 0.754508

All 6 figures generated. Check Visuals/ directory.
```

---

## Requirements

```
numpy
matplotlib
```

Both are included in the repo-level `requirements.txt`.

---

## Key Concepts (Technical Notes)

- **Cosine similarity = cosθ**: `u·v / (||u||·||v||)` is the physics formula `u·v = ||u||||v||cosθ` rearranged. Same expression, different context.
- **Pre-normalized embeddings**: Production vector databases (Pinecone, Milvus, FAISS) normalize embeddings to unit length at index time. At query time, cosine similarity collapses to a plain dot product — which is why they default to Inner Product (IP) distance.
- **Weights formula derivation**: `cⱼ = (y·uⱼ)/(uⱼ·uⱼ)` is the scalar coefficient from the geometric decomposition of y along uⱼ: (shadow length) × (unit direction), where shadow = `y·uⱼ/||uⱼ||` and direction = `uⱼ/||uⱼ||`.
- **√d scaling in attention**: If q and k have components with mean 0 and variance 1, their dot product has variance d. Without dividing by √d, softmax inputs explode in high dimensions (d=768 for BERT), pushing gradients toward zero. Dividing by √d restores variance to 1.
- **Orthogonality vs variance in PCA**: Orthogonality prevents redundancy between principal components. Eigenvalues (variance captured) determine which components are important. Both are necessary.

---

## Part of The Missing Link Series

| Previous | This Article | Next |
|----------|-------------|------|
| [3B — Same Data. Different Coordinates.](https://medium.com/@ashtosh.shenoy/f04bede0f99c) | **4A — The Math You Already Know Is Running Your RAG Pipeline** | 4B — Orthogonal Projection & Gram-Schmidt *(coming soon)* |
