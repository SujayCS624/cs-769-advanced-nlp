import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """

    embedding_matrix = np.zeros((len(vocab), emb_size))

    # Reading the FastText word embeddings as per 
    # https://fasttext.cc/docs/en/english-vectors.html
    with open(emb_file, 'r', encoding='utf-8') as f:
        num_words, emb_dim = map(int, f.readline().split())
        for line in f:
            tokens = line.rstrip().split(' ')
            word = tokens[0]
            vector = np.asarray(tokens[1:], dtype='float32')

            if word in vocab.word2id:
                idx = vocab.word2id[word]
                embedding_matrix[idx] = vector

    return embedding_matrix


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        hid_layer_size = self.args.hid_size
        vocab_size = len(self.vocab)
        emb_size = self.args.emb_size
        pad_id = self.vocab.pad_id
        tag_size = self.tag_size
        emb_drop_prob = self.args.emb_drop

        # Embedding layer
        self.embeddings = nn.Embedding(vocab_size, emb_size, padding_idx = pad_id)

        # Define network architecture with 3 layers and ReLU activation function
        self.classifier = nn.Sequential(
            nn.Linear(emb_size, hid_layer_size),
            nn.ReLU(),
            nn.Linear(hid_layer_size, hid_layer_size),
            nn.ReLU(),
            nn.Linear(hid_layer_size, tag_size)
        )

        # Define the embedding dropout layer for regularization
        self.word_dropout = nn.Dropout(p=emb_drop_prob)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        nn.init.uniform_(self.embeddings.weight, -0.08, 0.08)
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                nn.init.uniform_(layer.weight, -0.08, 0.08)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """

        vocab = self.vocab
        emb_file = self.args.emb_file
        emb_size = self.args.emb_size

        pre_trained_embeddings = load_embedding(
            vocab, emb_file, emb_size
        )

        for i, embedding in enumerate(pre_trained_embeddings):
            if embedding is not None:
                with torch.no_grad():
                    self.embeddings.weight[i] = torch.Tensor(embedding)        


    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """

        # Find number of words in a sentence excluding the padding
        word_counts = torch.sum(x != self.vocab.pad_id, dim=1)
        # Convert each sentence in the mini-batch into word embeddings
        embedded_x = self.embeddings(x)
        # Randomly drop word embeddings
        embedded_x = self.word_dropout(embedded_x)
        
        # Calculate sum of word embeddings for the sentence
        embedded_x_sum = torch.sum(embedded_x, dim=1)
        # Calculate mean of word embeddings for the sentence
        embedded_x_avg = torch.div(embedded_x_sum, torch.unsqueeze(word_counts, dim=1))
        # Return unnormalized scores
        scores = self.classifier(embedded_x_avg)

        return scores
