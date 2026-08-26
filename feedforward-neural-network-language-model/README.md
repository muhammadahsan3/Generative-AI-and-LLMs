# 🧠 Feedforward Neural Network for Language Modeling

A deep learning project that builds and trains an **N-gram Feedforward Neural Network (FNN)** in PyTorch for **language modeling and text generation**. The model learns word relationships from song lyrics and nursery rhymes, then generates new text sequences.

---

## 📌 Project Overview

This lab explores how Feedforward Neural Networks can be used for NLP tasks. Given a context of previous words, the model predicts the next word — essentially learning the structure and patterns of language from scratch using n-grams and word embeddings.

---

## 🗂️ Project Structure

```
├── FeedForwardNeuralNetworks.ipynb   # Main notebook
├── 2gram.pth                         # Saved 2-gram model weights (generated after training)
├── 4gram.pth                         # Saved 4-gram model weights (generated after training)
├── 8gram.pth                         # Saved 8-gram model weights (generated after training)
├── requirements.txt                  # Python dependencies
├── .gitignore                        # Files to ignore in Git
└── README.md                         # Project documentation
```

---

## 🧠 Model Architecture

```
Input       →  N context word indices
Embedding   →  EMBEDDING_DIM vectors per word (10-dimensional)
Flatten     →  Concatenated embeddings (EMBEDDING_DIM × CONTEXT_SIZE)
Hidden      →  128 neurons (ReLU activation)
Output      →  vocab_size neurons (next word probability)
```

---

## ⚙️ Project Workflow

1. **Text Input** — Uses "Never Gonna Give You Up" lyrics as training data
2. **Tokenization** — Basic English tokenizer from `torchtext`
3. **Preprocessing** — Removes punctuation, digits, and normalizes to lowercase
4. **Vocabulary Building** — Maps tokens to unique integer indices with `<unk>` token
5. **Embedding Layer** — Converts word indices into dense 10-dimensional vectors
6. **N-gram Generation** — Creates (context → target) pairs for context sizes 2, 4, and 8
7. **Batch Processing** — Uses `DataLoader` with padding to handle uneven batches
8. **Model Training** — SGD optimizer with StepLR learning rate scheduler, 100 epochs
9. **Text Generation** — `write_song()` generates 100 new words from a seed line
10. **Evaluation** — Plots training loss and perplexity curves for all 3 models
11. **Visualization** — t-SNE plots of word embeddings to see semantic clustering

---

## 🔍 N-gram Context Size Comparison

| Model | Context Size | Saved As |
|-------|-------------|----------|
| model_2 | 2 words | `2gram.pth` |
| model_4 | 4 words | `4gram.pth` |
| model_8 | 8 words | `8gram.pth` |

> 💡 Larger context sizes generally result in lower loss and perplexity.

---

## 📊 Key Concepts Covered

- **Tokenization & Indexing** — Converting raw text to numerical representations
- **Word Embeddings** — Dense vector representations that capture semantic meaning
- **N-gram Language Modeling** — Predicting next word from N previous words
- **Perplexity** — NLP evaluation metric derived from cross-entropy loss (`exp(loss)`)
- **t-SNE Visualization** — Reducing embedding dimensions to 2D for visual inspection
- **Batch Padding** — Handling uneven batch sizes in DataLoader

---

## 🛠️ Tech Stack

- Python 3
- PyTorch
- TorchText
- NLTK
- Scikit-learn (t-SNE)
- NumPy
- Pandas
- Matplotlib

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies
```bash
pip install numpy==1.26.4 pandas matplotlib seaborn scikit-learn nltk
pip install torch==2.2.2+cpu torchvision==0.17.2+cpu torchtext==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the notebook
```bash
jupyter notebook FeedForwardNeuralNetworks.ipynb
```

> ⚠️ After installing libraries, **restart the kernel** before running all cells.

---

## 🧪 Exercises Included

The notebook includes 6 hands-on exercises:

1. Source nursery rhymes into a text variable
2. Preprocess and create n-grams from the rhymes
3. Convert context words into embeddings and pass through a linear layer
4. Implement batch processing with padding
5. Train an N-gram language model on nursery rhymes
6. Generate a new nursery rhyme using the trained model

---

## 🚀 Future Improvements

- Use a larger and more diverse text corpus
- Replace FNN with an RNN or LSTM for better sequence modeling
- Add a beam search decoder for better text generation quality
- Build a simple Streamlit web app for interactive text generation

---

## 👨‍💻 Original Lab Authors

- [Joseph Santarcangelo](https://www.linkedin.com/in/joseph-s-50398b136/) — IBM
- [Roodra Kanwar](https://www.linkedin.com/in/roodrakanwar/) — Simon Fraser University

---

## 📄 License

© IBM Corporation. Dataset and lab structure originally from IBM Skills Network.  
Personal implementation and exercises completed independently.
