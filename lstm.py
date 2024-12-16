import torch

torch.manual_seed(1)

training_data = [
    ("The dog ate the apple".lower().split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".lower().split(), ["NN", "V", "DET", "NN"])
]


word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:  # word has not been assigned an index yet
            word_to_ix[word] = len(word_to_ix)

word_to_ix

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 8

class LSTMTagger(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(9, EMBEDDING_DIM)
        self.lstm = torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM)

        self.pos_predictor = torch.nn.Linear(HIDDEN_DIM, 3)

    def forward(self, token_ids):
        embeds = self.embedding_layer(token_ids)
        lstm_out, _ = self.lstm(embeds.view(len(token_ids), 1, -1))
        logits = self.pos_predictor(lstm_out.view(len(token_ids), -1))
        probs = torch.nn.functional.softmax(logits, dim=1)

        return probs

model = LSTMTagger()
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
with torch.no_grad():
    inputs = prepare_sequence(training_data[0][0], word_to_ix)
    tag_scores = model(inputs)

    print(tag_scores)