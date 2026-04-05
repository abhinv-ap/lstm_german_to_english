"""
German → English Neural Machine Translation
Using a Seq2Seq LSTM model (Encoder-Decoder Architecture)

WHAT THIS MODEL DOES:
  - Reads a German sentence word by word (Encoder)
  - Compresses it into a "thought vector" (context vector)
  - Generates English words one by one (Decoder)

CONCEPTS COVERED (great for interviews!):
  ✅ Tokenization & Vocabulary building
  ✅ Padding & batching sequences
  ✅ Embedding layers
  ✅ LSTM (Long Short-Term Memory)
  ✅ Encoder-Decoder architecture
  ✅ Teacher Forcing (training trick)
  ✅ Greedy decoding (inference)
  ✅ BLEU score evaluation
"""

import torch
import torch.nn as nn
import random
import math
from torch.utils.data import Dataset, DataLoader

# ─────────────────────────────────────────────
# 1. TOY DATASET
#    In a real project, you'd load a large dataset
#    like Multi30k or WMT. Here we use ~30 pairs
#    so you can train on a CPU in seconds.
# ─────────────────────────────────────────────
SENTENCE_PAIRS = [
    ("ich bin müde", "i am tired"),
    ("er spielt fußball", "he plays football"),
    ("sie liest ein buch", "she reads a book"),
    ("wir essen pizza", "we eat pizza"),
    ("das wetter ist schön", "the weather is nice"),
    ("ich liebe dich", "i love you"),
    ("guten morgen", "good morning"),
    ("gute nacht", "good night"),
    ("wie geht es dir", "how are you"),
    ("ich heiße hans", "my name is hans"),
    ("das ist ein hund", "that is a dog"),
    ("die katze schläft", "the cat sleeps"),
    ("er trinkt kaffee", "he drinks coffee"),
    ("ich spreche deutsch", "i speak german"),
    ("das buch ist rot", "the book is red"),
    ("sie singt ein lied", "she sings a song"),
    ("wir gehen nach hause", "we go home"),
    ("das kind lacht", "the child laughs"),
    ("er arbeitet hart", "he works hard"),
    ("ich mag äpfel", "i like apples"),
    ("die sonne scheint", "the sun shines"),
    ("er fährt auto", "he drives a car"),
    ("sie kocht suppe", "she cooks soup"),
    ("das haus ist groß", "the house is big"),
    ("ich bin glücklich", "i am happy"),
    ("er ist mein freund", "he is my friend"),
    ("die musik ist laut", "the music is loud"),
    ("ich bin müde und hungrig", "i am tired and hungry"),
    ("sie ist sehr klug", "she is very smart"),
    ("wir lernen zusammen", "we learn together"),
]

# ─────────────────────────────────────────────
# 2. VOCABULARY
#    Maps each unique word ↔ integer index
#    Special tokens:
#      <pad> = 0  → padding to equal lengths
#      <sos> = 1  → start of sentence
#      <eos> = 2  → end of sentence
#      <unk> = 3  → unknown word
# ─────────────────────────────────────────────
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3}
        self.idx2word = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.n_words = 4

    def add_sentence(self, sentence):
        for word in sentence.split():
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def encode(self, sentence):
        """Convert sentence string → list of integer indices"""
        return [self.word2idx.get(w, 3) for w in sentence.split()]

    def decode(self, indices):
        """Convert list of integer indices → sentence string"""
        words = []
        for idx in indices:
            word = self.idx2word.get(idx, "<unk>")
            if word in ("<sos>", "<pad>"):
                continue
            if word == "<eos>":
                break
            words.append(word)
        return " ".join(words)


def build_vocab(pairs):
    src_vocab = Vocabulary()  # German
    trg_vocab = Vocabulary()  # English
    for de, en in pairs:
        src_vocab.add_sentence(de)
        trg_vocab.add_sentence(en)
    return src_vocab, trg_vocab


# ─────────────────────────────────────────────
# 3. DATASET & DATALOADER
#    Converts raw pairs → padded tensors
# ─────────────────────────────────────────────
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.data = []
        for de, en in pairs:
            src = [1] + src_vocab.encode(de) + [2]   # <sos> ... <eos>
            trg = [1] + trg_vocab.encode(en) + [2]
            self.data.append((src, trg))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length"""
    srcs, trgs = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_trg = max(len(t) for t in trgs)
    padded_src = [s + [0] * (max_src - len(s)) for s in srcs]
    padded_trg = [t + [0] * (max_trg - len(t)) for t in trgs]
    return (
        torch.tensor(padded_src, dtype=torch.long),
        torch.tensor(padded_trg, dtype=torch.long),
    )


# ─────────────────────────────────────────────
# 4. ENCODER
#    Reads the German sentence and produces
#    a hidden state (the "thought vector")
#
#    Input:  [batch, src_len]
#    Output: hidden & cell states for Decoder
# ─────────────────────────────────────────────
class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        # Embedding: converts word indices → dense vectors
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # LSTM processes the sequence
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))   # [batch, src_len, embed_dim]
        _, (hidden, cell) = self.lstm(embedded)        # hidden/cell: [n_layers, batch, hidden_dim]
        return hidden, cell


# ─────────────────────────────────────────────
# 5. DECODER
#    Generates English words one at a time,
#    using the Encoder's hidden state as context
#
#    At each step:
#      input  = previous word (or <sos> at start)
#      output = probability distribution over English vocab
# ─────────────────────────────────────────────
class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, n_layers,
                            dropout=dropout, batch_first=True)
        # Linear layer maps hidden state → vocabulary scores
        self.fc_out = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg_token, hidden, cell):
        # trg_token: [batch] → unsqueeze to [batch, 1]
        trg_token = trg_token.unsqueeze(1)
        embedded = self.dropout(self.embedding(trg_token))  # [batch, 1, embed_dim]
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        prediction = self.fc_out(output.squeeze(1))         # [batch, vocab_size]
        return prediction, hidden, cell


# ─────────────────────────────────────────────
# 6. SEQ2SEQ (puts Encoder + Decoder together)
#
#    TEACHER FORCING:
#      During training, 50% of the time we feed
#      the CORRECT previous word to the decoder
#      instead of the predicted one.
#      → Speeds up learning & improves stability
# ─────────────────────────────────────────────
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, trg_vocab_size):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.trg_vocab_size = trg_vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.shape[0]
        trg_len = trg.shape[1]

        # Storage for decoder outputs at each time step
        outputs = torch.zeros(batch_size, trg_len, self.trg_vocab_size)

        # Encode the source sentence
        hidden, cell = self.encoder(src)

        # First decoder input = <sos> token
        dec_input = trg[:, 0]

        for t in range(1, trg_len):
            pred, hidden, cell = self.decoder(dec_input, hidden, cell)
            outputs[:, t, :] = pred

            # Teacher forcing decision
            use_teacher = random.random() < teacher_forcing_ratio
            top1 = pred.argmax(dim=1)
            dec_input = trg[:, t] if use_teacher else top1

        return outputs


# ─────────────────────────────────────────────
# 7. TRAINING LOOP
# ─────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for src, trg in loader:
        optimizer.zero_grad()
        output = model(src, trg)
        # output: [batch, trg_len, vocab_size]
        # Reshape for cross-entropy: skip <sos> at position 0
        output = output[:, 1:, :].reshape(-1, output.shape[-1])
        trg = trg[:, 1:].reshape(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# ─────────────────────────────────────────────
# 8. INFERENCE (translate a new sentence)
# ─────────────────────────────────────────────
def translate(model, sentence, src_vocab, trg_vocab, max_len=20):
    model.eval()
    with torch.no_grad():
        tokens = [1] + src_vocab.encode(sentence) + [2]  # <sos> + sentence + <eos>
        src = torch.tensor(tokens).unsqueeze(0)           # [1, src_len]
        hidden, cell = model.encoder(src)

        dec_input = torch.tensor([1])                     # start with <sos>
        result = []

        for _ in range(max_len):
            pred, hidden, cell = model.decoder(dec_input, hidden, cell)
            top1 = pred.argmax(dim=1)
            word = trg_vocab.idx2word.get(top1.item(), "<unk>")
            if word == "<eos>":
                break
            result.append(word)
            dec_input = top1

    return " ".join(result)


# ─────────────────────────────────────────────
# 9. SIMPLE BLEU SCORE (n-gram precision metric)
#    BLEU measures how many n-grams in the
#    predicted translation match the reference.
#    Score of 1.0 = perfect, 0.0 = no match.
# ─────────────────────────────────────────────
def compute_bleu(predicted, reference, n=2):
    pred_tokens = predicted.split()
    ref_tokens = reference.split()
    if len(pred_tokens) < n or len(ref_tokens) < n:
        return 1.0 if predicted == reference else 0.0
    pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens)-n+1)]
    ref_ngrams  = [tuple(ref_tokens[i:i+n])  for i in range(len(ref_tokens)-n+1)]
    matches = sum(1 for ng in pred_ngrams if ng in ref_ngrams)
    precision = matches / len(pred_ngrams) if pred_ngrams else 0
    brevity = math.exp(1 - len(ref_tokens)/len(pred_tokens)) if len(pred_tokens) < len(ref_tokens) else 1
    return brevity * precision


# ─────────────────────────────────────────────
# 10. MAIN — Build, Train, Evaluate
# ─────────────────────────────────────────────
def main():
    # Hyperparameters (easy to tune!)
    EMBED_DIM   = 64    # size of word embedding vectors
    HIDDEN_DIM  = 128   # LSTM hidden state size
    N_LAYERS    = 2     # number of stacked LSTM layers
    DROPOUT     = 0.3   # regularization
    BATCH_SIZE  = 8
    N_EPOCHS    = 100
    LR          = 0.001

    print("=" * 55)
    print("  German → English LSTM Translation Model")
    print("=" * 55)

    # Build vocabularies
    src_vocab, trg_vocab = build_vocab(SENTENCE_PAIRS)
    print(f"\n📚 German vocab size : {src_vocab.n_words}")
    print(f"📚 English vocab size: {trg_vocab.n_words}")

    # Dataset & DataLoader
    dataset = TranslationDataset(SENTENCE_PAIRS, src_vocab, trg_vocab)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # Build model
    encoder = Encoder(src_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    decoder = Decoder(trg_vocab.n_words, EMBED_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
    model   = Seq2Seq(encoder, decoder, trg_vocab.n_words)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🔢 Total trainable parameters: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad> tokens

    # Training
    print(f"\n🏋️  Training for {N_EPOCHS} epochs...\n")
    for epoch in range(1, N_EPOCHS + 1):
        loss = train_epoch(model, loader, optimizer, criterion)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch:3d}/{N_EPOCHS}  |  Loss: {loss:.4f}  |  Perplexity: {math.exp(loss):.2f}")

    # Evaluation
    print("\n" + "=" * 55)
    print("  TRANSLATION RESULTS")
    print("=" * 55)

    total_bleu = 0
    for de, en_ref in SENTENCE_PAIRS[:10]:
        en_pred = translate(model, de, src_vocab, trg_vocab)
        bleu    = compute_bleu(en_pred, en_ref)
        total_bleu += bleu
        print(f"\n  🇩🇪  {de}")
        print(f"  🎯  Expected : {en_ref}")
        print(f"  🤖  Predicted: {en_pred}")
        print(f"  📊  BLEU-2   : {bleu:.3f}")

    print(f"\n  Average BLEU-2: {total_bleu/10:.3f}")
    print("\n✅ Done! Model trained successfully.")

    # Save model
    torch.save({
        "model_state": model.state_dict(),
        "src_vocab":   src_vocab,
        "trg_vocab":   trg_vocab,
        "config": {
            "embed_dim":  EMBED_DIM,
            "hidden_dim": HIDDEN_DIM,
            "n_layers":   N_LAYERS,
            "dropout":    DROPOUT,
        }
    }, "translation_model.pt")
    print("💾 Model saved to translation_model.pt")

    return model, src_vocab, trg_vocab


if __name__ == "__main__":
    main()