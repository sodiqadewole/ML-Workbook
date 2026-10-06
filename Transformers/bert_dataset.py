import torch
from torch.utils.data import Dataset, DataLoader
import random

class SimpleTokenizer:
    """A minimal vocabulary builder and tokenizer for demonstration."""
    def __init__(self, corpus):
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.cls_token = "[CLS]"
        self.sep_token = "[SEP]"
        self.mask_token = "[MASK]"
        
        # Unique tokens list
        unique_words = set(word for text in corpus for word in text.lower().split())
        self.vocab = [self.pad_token, self.unk_token, self.cls_token, self.sep_token, self.mask_token] + list(unique_words)
        self.w2i = {word: idx for idx, word in enumerate(self.vocab)}
        self.i2w = {idx: word for idx, word in enumerate(self.vocab)}
        
    @property
    def vocab_size(self):
        return len(self.vocab)
        
    def tokenize(self, text):
        return text.lower().split()
        
    def convert_tokens_to_ids(self, tokens):
        return [self.w2i.get(token, self.w2i[self.unk_token]) for token in tokens]

class BERTDataset(Dataset):
    def __init__(self, corpus, tokenizer, max_len=32):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lines = [tokenizer.tokenize(line) for line in corpus if line.strip()]
        self.corpus_len = len(self.lines)

    def __len__(self):
        # We will generate 1000 synthetic paired examples for training
        return 1000 

    def _get_nsp_pair(self):
        """Randomly selects if the next sentence is actual or random."""
        t1_idx = random.randint(0, self.corpus_len - 1)
        t1 = self.lines[t1_idx]
        
        # 50% chance of being the actual next sentence, 50% chance of random sentence
        if random.random() > 0.5:
            # Is Next Sentence
            t2 = self.lines[(t1_idx + 1) % self.corpus_len]
            is_next = 1
        else:
            # Is Not Next Sentence
            t2 = self.lines[random.randint(0, self.corpus_len - 1)]
            is_next = 0
            
        return t1, t2, is_next

    def _apply_mlm(self, tokens):
        """Applies the 15% masking logic used in BERT."""
        labels = []
        masked_tokens = []
        
        for token in tokens:
            prob = random.random()
            
            # Special tokens are never masked
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                masked_tokens.append(token)
                labels.append(-100) # -100 is the PyTorch standard to ignore in Loss calculation
                continue
                
            if prob < 0.15: # 15% of tokens are designated for MLM target
                prob /= 0.15
                
                # 80% change: replace with [MASK]
                if prob < 0.8:
                    masked_tokens.append(self.tokenizer.mask_token)
                # 10% chance: replace with a completely random token
                elif prob < 0.9:
                    masked_tokens.append(random.choice(self.tokenizer.vocab))
                # 10% chance: keep token unchanged
                else:
                    masked_tokens.append(token)
                    
                labels.append(self.tokenizer.w2i.get(token, self.tokenizer.w2i[self.tokenizer.unk_token]))
            else:
                masked_tokens.append(token)
                labels.append(-100) # Not a prediction target
                
        return masked_tokens, labels

    def __getitem__(self, idx):
        # 1. Get NSP pairs
        t1, t2, is_next = self._get_nsp_pair()
        
        # Truncate if total sequence exceeds max length (-3 accounts for [CLS], [SEP], [SEP])
        max_tokens_for_sentences = self.max_len - 3
        while len(t1) + len(t2) > max_tokens_for_sentences:
            if len(t1) > len(t2):
                t1.pop()
            else:
                t2.pop()
                
        # 2. Construct raw token lists and segment labels
        tokens = [self.tokenizer.cls_token] + t1 + [self.tokenizer.sep_token] + t2 + [self.tokenizer.sep_token]
        segment_ids = [0] * (len(t1) + 2) + [1] * (len(t2) + 1)
        
        # 3. Apply MLM masking rules
        masked_tokens, mlm_labels = self._apply_mlm(tokens)
        
        # 4. Map tokens to Vocabulary IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(masked_tokens)
        
        # 5. Handle Padding
        padding_len = self.max_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * padding_len
        
        input_ids += [self.tokenizer.w2i[self.tokenizer.pad_token]] * padding_len
        segment_ids += [0] * padding_len
        mlm_labels += [-100] * padding_len
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "mlm_labels": torch.tensor(mlm_labels, dtype=torch.long),
            "nsp_label": torch.tensor(is_next, dtype=torch.long)
        }