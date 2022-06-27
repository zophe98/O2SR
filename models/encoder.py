import torch
from torch import nn
from .rnn_encoder import SentenceEncoderRNN

class QuestionEncoder(nn.Module):
    def __init__(self,
                 vocab_size,
                 word_dim=300,
                 rnn_dim=512,
                 bidirectional=True,
                 rnn_cell='lstm',
                 embedding=None,
                 update_embedding=True):
        super(QuestionEncoder, self).__init__()

        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        if embedding is not None:
            self.embedding = embedding
        else:
            self.embedding = nn.Embedding(vocab_size, word_dim, padding_idx=0)

        self.embedding.weight.requires_grad = update_embedding

        self.embeding_proj = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(word_dim, rnn_dim, bias=False)
        )

        self.sentenct_encoder = SentenceEncoderRNN(
            vocab_size,
            rnn_dim,
            input_dropout_p=0.3,
            n_layers=1,
            bidirectional=bidirectional,
            rnn_cell=rnn_cell
        )
        self.act = nn.ReLU()
    def forward(self, questions, question_len, caption=False):
        """
        Args:
            question: [Tensor] (batch_size, max_question_length)
            question_len: [Tensor] (batch_size)
        return:
            glob_embedding [Tensor] (batch_size, module_dim)
            local_embedding [Tensor] (batch_size, max_question_length, module_dim)
        """
        batchsize = questions.size(0)
        # (B,L, word_dim)
        question_embedding = self.embedding(questions)
        # (B,L,module_dim)
        question_embedding = self.act(self.embeding_proj(question_embedding))

        output, hidden, bi_hidden = self.sentenct_encoder(question_embedding, input_lengths=question_len)

        # output of shape (batch, seq_len, hidden_size)
        # hidden of shape (batch, hidden_size)

        if not caption:
            return output, hidden
        else:
            return output, hidden, bi_hidden



