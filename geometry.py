import sys
import torch


def conicity(weight):
    mean = torch.mean(weight, dim=0)
    cos = torch.nn.CosineSimilarity(dim=0)
    atm = 0
    for i in range(weight.shape[0]):
        # print(weight[i].shape, mean.shape)
        atm += cos(weight[i], mean)
    return atm / weight.shape[0]


checkpoint = sys.argv[1]
state_dict = torch.load(checkpoint, map_location='cpu')

embed = state_dict['emb_e_real.weight']

print(conicity(embed))
