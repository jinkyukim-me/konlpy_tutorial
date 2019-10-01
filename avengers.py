import torch
import torch.nn as nn
import torch.optim as optim
torch.manual_seed(70)

sentence = "We are going to watch Avengers End Game".split()
vocab = {tkn: i for i, tkn in enumerate(sentence, 1)}
vocab['<unk>'] = 0
rev_vocab = {v: k for k, v in vocab.items()}
decode = lambda y: [rev_vocab.get(x) for x in y]

def construct_data(sentence, vocab):
    numericalize = lambda x: vocab.get(x) if vocab.get(x) is not None else 0
    totensor = lambda x: torch.LongTensor(x)
    idxes = [numericalize(token) for token in sentence]
    x, t = idxes[:-1], idxes[1:]
    return totensor(x).unsqueeze(0), totensor(t).unsqueeze(0)

class Net(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
        self.rnn_layer = nn.RNN(input_size, hidden_size, batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        output = self.embedding_layer(x)
        output, hidden = self.rnn_layer(output)
        output = self.linear(output)
        return output.view(-1, output.size(2))

x, t = construct_data(sentence, vocab)

vocab_size = len(vocab)
input_size = 5
hidden_size = 20

model = Net(vocab_size, input_size, hidden_size, batch_first=True)
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(params=model.parameters())

for step in range(151):
    optimizer.zero_grad()
    output=model(x)
    loss = loss_function(output, t.view(-1))
    loss.backward()
    optimizer.step()
    if step % 30 ==0:
        print("[{:02d}/151] {:.4f}".format(step+1, loss))
        pred = output.softmax(-1).argmax(-1).tolist()
        print(" ".join(["We"] + decode(pred)))
        print()
