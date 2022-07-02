import torch

def get_atten_mask(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len, max_len])
    for i in range(batch_size):
        length = seq_lens[i]
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()

def get_atten_mask_frame(seq_lens, batch_size):
    max_len = seq_lens[0]
    atten_mask = torch.ones([batch_size, max_len*20, max_len*20])
    for i in range(batch_size):
        length = seq_lens[i]*20
        atten_mask[i, :length, :length] = 0
    return atten_mask.bool()


def std_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = torch.tensor(seq_lens) / (torch.tensor(seq_lens) - 1)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, length:, :] = 1e-9
    return atten_mask, weight_unbaised

def mean_mask(seq_lens, batchsize, dim):
    max_len = seq_lens[0]
    weight_unbaised = seq_lens[0] / torch.tensor(seq_lens)
    atten_mask = torch.ones([batchsize, max_len, dim])
    for i in range(batchsize):
        length = seq_lens[i]
        atten_mask[i, :length, :] = 0
    return atten_mask.bool(), weight_unbaised