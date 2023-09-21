from torch import nn

class sentiment_model(nn.Module):
    def __init__(self, embed_out, hidden_size, numb_label, dropout_p):
        super().__init__()
        self.fc1 = nn.Linear(embed_out, hidden_size)
        self.activation_func = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, numb_label)
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, embeddings, drop_out):
        if drop_out:
            embeddings = self.dropout(embeddings)
        fc1_out = self.fc1(embeddings)
        active_out = self.activation_func(fc1_out)
        fc2_out = self.fc2(active_out)
        return fc2_out