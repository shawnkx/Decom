import torch
from torch.nn import functional as F, Parameter
from torch.autograd import Variable


from spodernet.utils.global_config import Config
from torch.nn.init import xavier_normal_, xavier_uniform_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence



class Complex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(Complex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.emb_e_real(e1).squeeze()
        rel_embedded_real = self.emb_rel_real(rel).squeeze()
        e1_embedded_img =  self.emb_e_img(e1).squeeze()
        rel_embedded_img = self.emb_rel_img(rel).squeeze()

        e1_embedded_real = self.inp_drop(e1_embedded_real)
        rel_embedded_real = self.inp_drop(rel_embedded_real)
        e1_embedded_img = self.inp_drop(e1_embedded_img)
        rel_embedded_img = self.inp_drop(rel_embedded_img)

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, self.emb_e_real.weight.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, self.emb_e_img.weight.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, self.emb_e_img.weight.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, self.emb_e_real.weight.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)
        #print(pred.size)

        return pred


class DistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMult, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = e1_embedded.squeeze()
        rel_embedded = rel_embedded.squeeze()

        e1_embedded = self.inp_drop(e1_embedded)
        rel_embedded = self.inp_drop(rel_embedded)

        pred = torch.mm(e1_embedded*rel_embedded, self.emb_e.weight.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred



class ConvE(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ConvE, self).__init__()
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.feature_map_drop = torch.nn.Dropout2d(Config.feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv2d(1, 32, (3, 3), 1, 0, bias=Config.use_bias)
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(32)
        self.bn2 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(10368,Config.embedding_dim)
        print(num_entities, num_relations)

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)

    def forward(self, e1, rel):
        e1_embedded= self.emb_e(e1).view(-1, 1, 10, 20)
        rel_embedded = self.emb_rel(rel).view(-1, 1, 10, 20)

        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 2)

        stacked_inputs = self.bn0(stacked_inputs)
        x= self.inp_drop(stacked_inputs)
        x= self.conv1(x)
        x= self.bn1(x)
        x= F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(Config.batch_size, -1)
        #print(x.size())
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = torch.mm(x, self.emb_e.weight.transpose(1,0))
        x += self.b.expand_as(x)
        pred = F.sigmoid(x)

        return pred


# Add your own model here
class LDecomDistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(LDecomDistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.lde_size = 400

        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        self.ent_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)
        self.rel_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.lde_size)
        self.bn2 = torch.nn.BatchNorm1d(self.lde_size)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.ent_decompress.weight.data)
        xavier_normal_(self.rel_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded= self.emb_e(e1).view(bsz, -1)
        #print(e1_embedded.shape)
        rel_embedded= self.emb_rel(rel).view(bsz, -1)
        e1_embedded = self.inp_drop(self.bn0(e1_embedded))
        rel_embedded = self.inp_drop(self.bn0(rel_embedded))

        e1_decompressed = self.ent_decompress(e1_embedded)
        e1_decompressed = self.bn1(e1_decompressed)
        e1_decompressed = self.hidden_drop((e1_decompressed))
        rel_decompressed = self.rel_decompress(rel_embedded)
        rel_decompressed = self.bn2(rel_decompressed).view(bsz, -1)
        rel_decompressed = self.hidden_drop((rel_decompressed))
        e2_embedded = self.emb_e.weight
        e2_embedded = self.inp_drop(self.bn0(e2_embedded))
        e2_decompressed = self.ent_decompress(e2_embedded)
        e2_decompressed = self.bn1(e2_decompressed).view(self.num_entities, -1)
        pred = torch.mm(e1_decompressed*rel_decompressed, e2_decompressed.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred

class LEnDecomDistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(LEnDecomDistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.lde_size = 400

        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, self.lde_size, padding_idx=0)

        self.ent_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)
        #self.rel_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.lde_size)
        self.bn2 = torch.nn.BatchNorm1d(self.lde_size)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.ent_decompress.weight.data)
        #xavier_normal_(self.rel_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded= self.emb_e(e1).view(bsz, -1)
        #print(e1_embedded.shape)
        rel_embedded= self.emb_rel(rel).view(bsz, -1)
        e1_embedded = self.inp_drop(self.bn0(e1_embedded))
        rel_embedded = self.inp_drop(self.bn2(rel_embedded))

        e1_decompressed = self.ent_decompress(e1_embedded)
        e1_decompressed = self.bn1(e1_decompressed)
        e1_decompressed = self.hidden_drop((e1_decompressed))
        rel_decompressed = rel_embedded
        #rel_decompressed = self.rel_decompress(rel_embedded)
        #rel_decompressed = self.bn2(rel_decompressed).view(bsz, -1)
        #rel_decompressed = self.hidden_drop((rel_decompressed))
        e2_embedded = self.emb_e.weight
        e2_embedded = self.inp_drop(self.bn0(e2_embedded))
        e2_decompressed = self.ent_decompress(e2_embedded)
        e2_decompressed = self.bn1(e2_decompressed).view(self.num_entities, -1)
        pred = torch.mm(e1_decompressed*rel_decompressed, e2_decompressed.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred


class LRelDecomDistMult(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(LRelDecomDistMult, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.lde_size = 200

        self.emb_e = torch.nn.Embedding(num_entities, self.lde_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        #self.ent_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)
        self.rel_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.lde_size)
        self.bn2 = torch.nn.BatchNorm1d(self.lde_size)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.rel_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded= self.emb_e(e1).view(bsz, -1)
        #print(e1_embedded.shape)
        rel_embedded= self.emb_rel(rel).view(bsz, -1)
        e1_embedded = self.inp_drop(self.bn2(e1_embedded))
        rel_embedded = self.inp_drop(self.bn0(rel_embedded))
        e1_decompressed = e1_embedded
        rel_decompressed = rel_embedded
        rel_decompressed = self.rel_decompress(rel_embedded)
        rel_decompressed = self.bn1(rel_decompressed).view(bsz, -1)
        rel_decompressed = self.hidden_drop((rel_decompressed))
        e2_embedded = self.emb_e.weight
        e2_embedded = self.inp_drop(self.bn2(e2_embedded))
        #e2_decompressed = self.ent_decompress(e2_embedded)
        #e2_decompressed = self.bn1(e2_decompressed).view(self.num_entities, -1)
        e2_decompressed = e2_embedded.view(self.num_entities, -1)
        pred = torch.mm(e1_decompressed*rel_decompressed, e2_decompressed.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred




class DistMultDecompress(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(DistMultDecompress, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_filters = 4
        self.kernel_size = 3

        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        self.ent_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        self.rel_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.num_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.num_filters)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.ent_decompress.weight.data)
        xavier_normal_(self.rel_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = self.inp_drop(self.bn0(e1_embedded))
        rel_embedded = self.inp_drop(self.bn0(rel_embedded))

        e1_decompressed = self.ent_decompress(e1_embedded)
        e1_decompressed = self.bn1(e1_decompressed).view(bsz, -1)
        e1_decompressed = self.hidden_drop((e1_decompressed))
        rel_decompressed = self.rel_decompress(rel_embedded)
        rel_decompressed = self.bn2(rel_decompressed).view(bsz, -1)
        rel_decompressed = self.hidden_drop((rel_decompressed))
        e2_embedded = self.emb_e.weight.unsqueeze(1)
        e2_embedded = self.inp_drop(self.bn0(e2_embedded))
        e2_decompressed = self.ent_decompress(e2_embedded)
        e2_decompressed = self.bn1(e2_decompressed).view(self.num_entities, -1)
        pred = torch.mm(e1_decompressed*rel_decompressed, e2_decompressed.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred




class EnDistMultDecompress(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(EnDistMultDecompress, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_filters = 4
        self.kernel_size = 3
        rel_emb_size = (Config.embedding_dim - self.kernel_size + 1) * self.num_filters
        self.emb_e = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, rel_emb_size, padding_idx=0)

        self.ent_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        #self.rel_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.num_filters)
        #self.bn2 = torch.nn.BatchNorm1d(self.num_filters)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        xavier_normal_(self.ent_decompress.weight.data)
        #xavier_normal_(self.rel_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = self.inp_drop(self.bn0(e1_embedded))
        rel_embedded = self.inp_drop(self.bn0(rel_embedded))

        e1_decompressed = self.ent_decompress(e1_embedded)
        e1_decompressed = self.bn1(e1_decompressed).view(bsz, -1)
        e1_decompressed = self.hidden_drop((e1_decompressed))
        rel_decompressed = rel_embedded.view(bsz, -1)
        e2_embedded = self.emb_e.weight.unsqueeze(1)
        e2_embedded = self.inp_drop(self.bn0(e2_embedded))
        e2_decompressed = self.ent_decompress(e2_embedded)
        e2_decompressed = self.bn1(e2_decompressed).view(self.num_entities, -1)

        pred = torch.mm(e1_decompressed*rel_decompressed, e2_decompressed.transpose(1,0))
        pred = F.sigmoid(pred)

        return pred


class RelDistMultDecompress(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(RelDistMultDecompress, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_filters = 4
        self.kernel_size = 3
        en_emb_size = (Config.embedding_dim - self.kernel_size + 1) * self.num_filters
        self.emb_e = torch.nn.Embedding(num_entities, en_emb_size, padding_idx=0)
        self.emb_rel = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        #self.ent_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        self.rel_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.num_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.num_filters)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e.weight.data)
        xavier_normal_(self.emb_rel.weight.data)
        #xavier_normal_(self.ent_decompress.weight.data)
        xavier_normal_(self.rel_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded= self.emb_e(e1)
        rel_embedded= self.emb_rel(rel)
        e1_embedded = self.inp_drop(self.bn0(e1_embedded))
        rel_embedded = self.inp_drop(self.bn0(rel_embedded))
        e1_decompressed = e1_embedded.view(bsz, -1)
        rel_decompressed = self.rel_decompress(rel_embedded)
        rel_decompressed = self.bn2(rel_decompressed).view(bsz, -1)
        rel_decompressed = self.hidden_drop((rel_decompressed))

        # e1_decompressed = e1_embedded
        # rel_decompressed = rel_embedded
        # e1_decompressed, rel_decompressed = e1_embedded.squeeze(), rel_embedded.squeeze()
        pred = torch.mm(e1_decompressed*rel_decompressed, self.emb_e.weight.transpose(1,0))
        
        prediction = F.sigmoid(pred)

        return prediction



class ComplexDecompress(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(ComplexDecompress, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_filters = 4
        self.kernel_size = 3

        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)

        self.ent_real_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        self.ent_img_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        self.rel_real_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        self.rel_img_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        #print(Config.dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.num_filters)
        self.bn2 = torch.nn.BatchNorm1d(self.num_filters)
        self.bn3 = torch.nn.BatchNorm1d(self.num_filters)
        self.bn4 = torch.nn.BatchNorm1d(self.num_filters)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)
        xavier_normal_(self.ent_real_decompress.weight.data)
        xavier_normal_(self.ent_img_decompress.weight.data)
        xavier_normal_(self.rel_real_decompress.weight.data)
        xavier_normal_(self.rel_img_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded_real = self.emb_e_real(e1)
        rel_embedded_real = self.emb_rel_real(rel)
        e1_embedded_img =  self.emb_e_img(e1)
        rel_embedded_img = self.emb_rel_img(rel)

        e1_embedded_real = self.inp_drop(self.bn0(e1_embedded_real))
        rel_embedded_real = self.inp_drop(self.bn0(rel_embedded_real))
        e1_embedded_img = self.inp_drop(self.bn0(e1_embedded_img))
        rel_embedded_img = self.inp_drop(self.bn0(rel_embedded_img))

        e1_embedded_real = self.hidden_drop(self.bn1(self.ent_real_decompress(e1_embedded_real)).view(bsz, -1)) 
        #print(e1_embedded_real.size())
        rel_embedded_real = self.hidden_drop(self.bn2(self.rel_real_decompress(rel_embedded_real)).view(bsz, -1))
        e1_embedded_img = self.hidden_drop(self.bn3(self.ent_img_decompress(e1_embedded_img)).view(bsz, -1))
        rel_embedded_img = self.hidden_drop(self.bn4(self.rel_img_decompress(rel_embedded_img)).view(bsz, -1))
        e2_embedded_real = self.hidden_drop(self.bn1(self.ent_real_decompress(self.emb_e_real.weight.unsqueeze(1))).view(self.num_entities, -1))
        e2_embedded_img = self.hidden_drop(self.bn3(self.ent_img_decompress(self.emb_e_img.weight.unsqueeze(1))).view(self.num_entities, -1))
        #2_embedded_img = self.

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, e2_embedded_real.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, e2_embedded_img.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, e2_embedded_img.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, e2_embedded_real.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred

class EnComplexDecompress(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(EnComplexDecompress, self).__init__()
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.num_filters = 4
        self.kernel_size = 3
        rel_emb_size = (Config.embedding_dim - self.kernel_size + 1) * self.num_filters
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, rel_emb_size, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, rel_emb_size, padding_idx=0)

        self.ent_real_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        self.ent_img_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        #self.rel_real_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)
        #self.rel_img_decompress = torch.nn.Conv1d(1, self.num_filters, self.kernel_size, 1, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        #print(Config.dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(1)
        self.bn1 = torch.nn.BatchNorm1d(self.num_filters)
        #self.bn2 = torch.nn.BatchNorm1d(self.num_filters)
        self.bn3 = torch.nn.BatchNorm1d(self.num_filters)
        #self.bn4 = torch.nn.BatchNorm1d(self.num_filters)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)
        xavier_normal_(self.ent_real_decompress.weight.data)
        xavier_normal_(self.ent_img_decompress.weight.data)
        #xavier_normal_(self.rel_real_decompress.weight.data)
        #xavier_normal_(self.rel_img_decompress.weight.data)

    def forward(self, e1, rel):
        bsz = e1.shape[0]

        e1_embedded_real = self.emb_e_real(e1)
        rel_embedded_real = self.emb_rel_real(rel)
        e1_embedded_img =  self.emb_e_img(e1)
        rel_embedded_img = self.emb_rel_img(rel)

        e1_embedded_real = self.inp_drop(self.bn0(e1_embedded_real))
        rel_embedded_real = self.inp_drop(self.bn0(rel_embedded_real))
        e1_embedded_img = self.inp_drop(self.bn0(e1_embedded_img))
        rel_embedded_img = self.inp_drop(self.bn0(rel_embedded_img))

        e1_embedded_real = self.hidden_drop(self.bn1(self.ent_real_decompress(e1_embedded_real)).view(bsz, -1)) 
        #print(e1_embedded_real.size())
        rel_embedded_real = rel_embedded_real.view(bsz, -1)
        e1_embedded_img = self.hidden_drop(self.bn3(self.ent_img_decompress(e1_embedded_img)).view(bsz, -1))
        rel_embedded_img = rel_embedded_img.view(bsz, -1)
        e2_embedded_real = self.hidden_drop(self.bn1(self.ent_real_decompress(self.emb_e_real.weight.unsqueeze(1))).view(self.num_entities, -1))
        e2_embedded_img = self.hidden_drop(self.bn3(self.ent_img_decompress(self.emb_e_img.weight.unsqueeze(1))).view(self.num_entities, -1))
        #2_embedded_img = self.

        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, e2_embedded_real.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, e2_embedded_img.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, e2_embedded_img.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, e2_embedded_real.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)

        return pred



class LComplexDeComplex(torch.nn.Module):
    def __init__(self, num_entities, num_relations):
        super(LDeComplex, self).__init__()
        self.num_entities = num_entities
        self.emb_e_real = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_e_img = torch.nn.Embedding(num_entities, Config.embedding_dim, padding_idx=0)
        self.emb_rel_real = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.emb_rel_img = torch.nn.Embedding(num_relations, Config.embedding_dim, padding_idx=0)
        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.lde_size = 200
        self.ent_real_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)
        self.rel_real_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)
        self.ent_img_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)
        self.rel_img_decompress = torch.nn.Linear(Config.embedding_dim, self.lde_size, bias=False)

        self.inp_drop = torch.nn.Dropout(Config.input_dropout)
        self.hidden_drop = torch.nn.Dropout(Config.dropout)
        self.bn0 = torch.nn.BatchNorm1d(Config.embedding_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.lde_size)
        self.bn2 = torch.nn.BatchNorm1d(self.lde_size)
        self.bn3 = torch.nn.BatchNorm1d(self.lde_size)
        self.bn4 = torch.nn.BatchNorm1d(self.lde_size)
        self.loss = torch.nn.BCELoss()

    def init(self):
        xavier_normal_(self.emb_e_real.weight.data)
        xavier_normal_(self.emb_e_img.weight.data)
        xavier_normal_(self.emb_rel_real.weight.data)
        xavier_normal_(self.emb_rel_img.weight.data)
        xavier_normal_(self.ent_real_decompress.weight.data)
        xavier_normal_(self.rel_real_decompress.weight.data)
        xavier_normal_(self.ent_img_decompress.weight.data)
        xavier_normal_(self.rel_img_decompress.weight.data)

    def forward(self, e1, rel):

        e1_embedded_real = self.inp_drop(self.bn0(self.emb_e_real(e1).squeeze()))
        rel_embedded_real = self.inp_drop(self.bn0(self.emb_rel_real(rel).squeeze()))
        e1_embedded_img =  self.inp_drop(self.bn0(self.emb_e_img(e1).squeeze()))
        rel_embedded_img = self.inp_drop(self.bn0(self.emb_rel_img(rel).squeeze()))

        e1_embedded_real = self.hidden_drop(self.bn1(self.ent_real_decompress(e1_embedded_real)))
        e1_embedded_img = self.hidden_drop(self.bn2(self.ent_real_decompress(e1_embedded_img)))
        rel_embedded_real = self.hidden_drop(self.bn3(self.ent_real_decompress(rel_embedded_real)))
        rel_embedded_img = self.hidden_drop(self.bn4(self.ent_real_decompress(rel_embedded_img)))
        e2_embedded_real = self.hidden_drop(self.bn1(self.ent_real_decompress(self.emb_e_real.weight)))
        e2_embedded_img = self.hidden_drop(self.bn2(self.ent_real_decompress(self.emb_e_img.weight)))
        # complex space bilinear product (equivalent to HolE)
        realrealreal = torch.mm(e1_embedded_real*rel_embedded_real, e2_embedded_real.transpose(1,0))
        realimgimg = torch.mm(e1_embedded_real*rel_embedded_img, e2_embedded_img.transpose(1,0))
        imgrealimg = torch.mm(e1_embedded_img*rel_embedded_real, e2_embedded_img.transpose(1,0))
        imgimgreal = torch.mm(e1_embedded_img*rel_embedded_img, e2_embedded_real.transpose(1,0))
        pred = realrealreal + realimgimg + imgrealimg - imgimgreal
        pred = F.sigmoid(pred)
        #print(pred.size)

        return pred


