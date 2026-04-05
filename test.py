import torch
import sys
from model import Encoder, Decoder, Seq2Seq, translate, Vocabulary

# This line tells pickle "when you see Vocabulary, look here"
sys.modules['__main__'].Vocabulary = Vocabulary

# Load the saved model
checkpoint = torch.load("translation_model.pt", weights_only=False)
cfg = checkpoint["config"]
src_vocab = checkpoint["src_vocab"]
trg_vocab = checkpoint["trg_vocab"]

encoder = Encoder(src_vocab.n_words, cfg["embed_dim"], cfg["hidden_dim"], cfg["n_layers"], cfg["dropout"])
decoder = Decoder(trg_vocab.n_words, cfg["embed_dim"], cfg["hidden_dim"], cfg["n_layers"], cfg["dropout"])
model   = Seq2Seq(encoder, decoder, trg_vocab.n_words)
model.load_state_dict(checkpoint["model_state"])

# Try your own sentences (must use words from training data)
test_sentences = [
    "ich bin müde",
    "guten morgen",
    "die katze schläft",
    "er trinkt kaffee",
]

print("\n--- Translation Test ---")
for sentence in test_sentences:
    result = translate(model, sentence, src_vocab, trg_vocab)
    print(f"🇩🇪 {sentence}")
    print(f"🇬🇧 {result}\n")