import torch
import torch.utils.data
import torch.nn.functional as F

from ecom.attention import Attention

class BalancedPoolLSTM(torch.nn.Module):
    def __init__(self, n_inp, em_sz, nh, out_cat, nl=2, **kwargs):
        super().__init__()
        emb_do = kwargs.pop('emb_do', 0.15)
        rnn_do = kwargs.pop('rnn_do', 0.25)
        out_do = kwargs.pop('out_do', 0.35)
        do_scale = kwargs.pop('do_scale', 1)
        self.nl, self.nh, self.out_cat = nl, nh, out_cat
        self.emb = torch.nn.Embedding(n_inp, em_sz, padding_idx=0)
        self.emb_drop = torch.nn.Dropout(emb_do*do_scale)
        self.rnn = torch.nn.LSTM(em_sz, nh, num_layers=nl, dropout=rnn_do*do_scale)
        self.out_drop = torch.nn.Dropout(out_do*do_scale)
        self.out = torch.nn.Linear(4*nh, out_cat)
        #self.out = torch.nn.Linear(nh, out_cat)
        #self.attention=Attention(seq_len=276,hidden_emb=nh)

    def pool(self, x, bs, is_max, k=1):
        f = F.adaptive_max_pool1d if is_max else F.adaptive_avg_pool1d
        return f(x.permute(1, 2, 0), (k,)).view(bs, -1)

    def forward(self, inp):
        sl, bs = inp.size()
        emb = self.emb_drop(self.emb(inp))
        rnn_out, h = self.rnn(emb, None)
        avgpool = self.pool(rnn_out, bs, False)
        maxpool = self.pool(rnn_out, bs, True,k=1)
        minpool = -self.pool(-rnn_out, bs, True,k=1)
        #median_pool=rnn_out.permute(1,2,0).median(dim=2)[0].view(bs,-1) #standard_deviation_pool=rnn_out.permute(1,2,0).std(dim=2).view(bs,-1)
        x = torch.cat([rnn_out[-1], avgpool, maxpool, minpool], 1)
        #attention_x=self.attention(rnn_out)
        #attention_x=attention_x.view(bs, -1).contiguous()
        return self.out(self.out_drop(x))
        #return self.out(self.out_drop(attention_x))
