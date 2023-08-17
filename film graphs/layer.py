from torch.nn.parameter import Parameter
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from scipy import sparse 
import numpy as np
from torch.nn.functional import normalize

class noflayer(nn.Module):
    def __init__(self, nnode, in_features, out_features, adj, max_degree, residual=False, variant=False):
        super(noflayer, self).__init__()
        self.max_degree = 170
        self.variant = variant
        self.nnode = nnode
        if self.variant:
            self.in_features = 2*in_features
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        #self.adj = adj.to_dense()
        self.adj = adj
        #self.adj = torch.sparse.mm(self.adj, self.adj)
        #self.rowsum_adj = torch.sum(self.adj,1)
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.a = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.b = nn.Parameter(torch.empty(size=(2*self.in_features, 1)))
        nn.init.xavier_uniform_(self.b.data, gain=1.414)
        self.alpha = 0.2
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.act_fn = nn.ReLU()
        self.f = Parameter(torch.ones(self.nnode))
        self.weight_matrix_att = Parameter(torch.FloatTensor(self.in_features, self.in_features))
        
        self.weight_matrix_att_prime_1 = torch.nn.Parameter(torch.Tensor(self.in_features, self.in_features))
        self.weight_matrix_att_prime_2 = torch.nn.Parameter(torch.Tensor(self.in_features, 1))
        self.weight_matrix_att_prime_3 = torch.nn.Parameter(torch.Tensor(self.in_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_matrix_att)
        torch.nn.init.xavier_uniform_(self.weight)
        #torch.nn.init.xavier_uniform_(self.weight_matrix2)

        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_1)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_2)
        torch.nn.init.xavier_uniform_(self.weight_matrix_att_prime_3)
        

    def attention(self, feature, beta):
        feature = torch.mm(feature, self.weight_matrix_att) 
        feat_1 = torch.matmul(feature, self.a[:self.in_features, :].clone())
        feat_2 = torch.matmul(feature, self.a[self.in_features:, :].clone())
        
        e = feat_1 + feat_2.T
        e = self.leakyrelu(e)
        
        #zero_vec = -9e15*torch.ones_like(e)
        #att = torch.where(self.adj > 0, e, zero_vec)
        #att = F.softmax(att, dim=1).clone()
        
        nonzero_indices = self.adj.coalesce().indices().t()
        values = e[nonzero_indices[:, 0], nonzero_indices[:, 1]]
        att = torch.sparse.FloatTensor(nonzero_indices.t(), values, torch.Size([self.adj.shape[0], self.adj.shape[1]]))
        att = torch.sparse.softmax(att,dim=1)
        
        U = att.clone()
        P = 0.5*U.clone()
        return U,P, att
    
    def forward_lifting_bases(self, feature, P, U, layer):
        #update = torch.einsum('ij,jk->ik', U.float(), feature.float())
        update = torch.sparse.mm(U,feature)
        #feat_even_bar = feature + update
        feat_even_bar = update
        Adj_hadamard_P = torch.mul(self.adj,P)
        rowsum = torch.sparse.sum(Adj_hadamard_P,1)

        nonzero_indices = rowsum.coalesce().indices().t()
        nonzero_indices = torch.cat((nonzero_indices,nonzero_indices),1) 
        value = rowsum.coalesce().values()
        diag = torch.sparse_coo_tensor(nonzero_indices.t(), value, size=(self.adj.shape[0],self.adj.shape[0]))

        feat_odd_bar = torch.sparse.mm(self.adj,feature)-torch.sparse.mm(diag,feat_even_bar)
        return feat_odd_bar, feat_even_bar
    
    
    def inverse_lifting_bases(self, feat_odd_bar, feat_even_bar, P, U,  h0):
        #feat_even_bar =  0.5*feat_even_bar + 0.5*h0
        feat_prime = 0.7*feat_even_bar + 0.3*(feat_odd_bar)
        feat_prime = 0.5*feat_prime + 0.5*h0
        return feat_prime

    
    def forward(self, input, h0, lamda, alpha, l):
        beta = math.log(lamda/l+1)
        hi = input
        #hi=torch.mm(self.support1, input)
        U,P,att = self.attention(hi, beta)
        feat_odd_bar, feat_even_bar = self.forward_lifting_bases(hi, P, U, l)  
        feat_prime = self.inverse_lifting_bases(feat_odd_bar, feat_even_bar, P, U, h0)
        #feat_prime = torch.mm(self.bala1.float(),feat_prime.float())
        output = feat_prime
        return output

class WaveletConvolution(nn.Module):

    def __init__(self, nnode, in_features, out_features, residual=False,
        variant=False):
        super(WaveletConvolution, self).__init__()
        self.variant = variant
        self.nnode = nnode
        if self.variant:
            self.in_features = 2 * in_features
        else:
            self.in_features = in_features
        self.out_features = out_features
        self.residual = residual
        self.myf = 1.2
        self.weight = Parameter(torch.FloatTensor(self.in_features, self.
            out_features))
        #self.f = Parameter(torch.ones(self.nnode))
        #self.f = Parameter(torch.ones(self.nnode,1))
        self.f = torch.diag(torch.ones(nnode)*self.myf).cuda()
        self.sign_mask_2 = torch.zeros((nnode, self.in_features)).cuda()
        self.reset_parameters()
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)
        #torch.nn.init.xavier_uniform_(self.f)

    def soft_thresholding(self, feat, threshold):
        sign_mask = self.sign_mask_2
        sign_mask[feat < 0] = -1
        sign_mask[feat > 0] = 1
        mod_feat = torch.abs(feat)
        mod_feat[mod_feat > threshold] -= threshold
        mod_feat[mod_feat <= threshold] = 0
        feat = torch.mul(mod_feat.clone(), sign_mask.clone())
        return feat
    
    def forward(self, input, support0, support1, h0, lamda, alpha, l):
        beta = math.log(lamda / l + 1)
        #bala1 = torch.spmm(support0, torch.diag(self.f))
        bala1 = torch.spmm(support0, self.f)
        bala2 = torch.mm(bala1, support1)
        hi = torch.mm(bala2, input)
        hi = self.soft_thresholding(hi, threshold=10**-6)
        if self.variant:
            support = torch.cat([hi, h0], 1)
            r = (1 - alpha) * hi + alpha * h0
        else:
            support = (1 - alpha) * hi + alpha * h0
            r = support
        #output = beta * torch.mm(support, self.weight) + (1 - beta) * r
        output = hi
        #output = torch.mm(hi, self.weight)
        if self.residual:
            output = output + input
        return output
