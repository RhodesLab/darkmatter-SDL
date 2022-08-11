import numpy as np
import torch
from torch import nn
from torch.functional import F
from torch.optim import Adam
from torch_geometric.nn import MetaLayer, MessagePassing
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, Softplus
from torch.autograd import Variable, grad

#n: number of particles (number of vertices), #n_f number of node features = in_channel #n_e number of edges = E

def make_packer(n, n_f): # create a list of node features instead of matrix with rows per vertice (particles)
    def pack(x):
        return x.reshape(-1, n_f*n)
    return pack

def make_unpacker(n, n_f): #unpack and create matrix from list of features
    def unpack(x):
        return x.reshape(-1, n, n_f)
    return unpack

def get_edge_index(n, sim): #building edge_index graph matrix, if not string, string_ball, all vertices are connected to each other
    if sim in ['string', 'string_ball']:
        #Should just be along it.
        top = torch.arange(0, n-1)
        bottom = torch.arange(1, n)
        edge_index = torch.cat(
            (torch.cat((top, bottom))[None],
             torch.cat((bottom, top))[None]), dim=0
        )
    else:
        adj = (np.ones((n, n)) - np.eye(n)).astype(int)
        edge_index = torch.from_numpy(np.array(np.where(adj)))

    return edge_index


class GN(MessagePassing):
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'): #hidden layer = 300, aggregate method is by adding
        super(GN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(  #NN model with multiple layers, learning a function from the 2*n_f node features from two vertices for each edge, which send to message function on the edge with msg_dim dimensions
            Lin(2*n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            ##(Can turn on or off this layer:)
#             Lin(hidden, hidden), 
#             ReLU(),
            Lin(hidden, msg_dim)
        )
        
        self.node_fnc = Seq(  #NN model with multiple hidden layers, turn message funciton + node features into a ndim output ( predicted value )
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
#             Lin(hidden, hidden),
#             ReLU(),
            Lin(hidden, ndim)
        )
    
    #[docs]
    def forward(self, x, edge_index):
        #x is [n, n_f], i.e. n vertices each with n_f many features
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x) #size is (n,n) which means all nodes are interacting and will thus be propogated, i.e. compute message function then aggregrate
      
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f], i.e. x_i records n_f features for the source of each edge and x_j records the n_f features for the target of each edge
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels], i.e. tmp gives for each edge, n_f features for the source and n_f targets for the edge
        return self.msg_fnc(tmp) # gives a matrix of [n_e, ndim] which gives for each edge its predicted message function
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim], n many nodes, each has msg_dim-dimensional vector as message, by aggregrating all msg_dim - dimensional messages pointing to it

        tmp = torch.cat([x, aggr_out], dim=1) #for each node, a new msg_dimensional vector is added to host new updated prediction
        return self.node_fnc(tmp) #[n, nupdate]


class OGN(GN):
    def __init__(
		self, n_f, msg_dim, ndim, dt,
		edge_index, aggr='add', hidden=300, nt=1):

        super(OGN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim
    
    def just_derivative(self, g, augment=False, augmentation=3): #What is g? g has attributes edge index, x, y. If augment = False, just_derivative gives the typical propogation
        #x is [n, n_f]f, number of nodes with n_f node features each. Additional f might be a typo?
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)  #adding augmentation into the features of x by a random vector
        
        edge_index = g.edge_index
        
        return self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)
    
    def loss(self, g, augment=True, square=False, augmentation=3, **kwargs): #loss is calculated as square distance between g.y and propogating g??
        if square:
            return torch.sum((g.y - self.just_derivative(g, augment=augment, augmentation=augmentation))**2)
        else:
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment)))




class varGN(MessagePassing): #variance, gaussian learning model, KL model
    def __init__(self, n_f, msg_dim, ndim, hidden=300, aggr='add'):
        super(varGN, self).__init__(aggr=aggr)  # "Add" aggregation.
        self.msg_fnc = Seq(
            Lin(2*n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
#             Lin(hidden, hidden),
#             ReLU(),
            Lin(hidden, msg_dim*2) #mu, logvar
        )
        
        self.node_fnc = Seq(
            Lin(msg_dim+n_f, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
            Lin(hidden, hidden),
            ReLU(),
#             Lin(hidden, hidden),
#             ReLU(),
            Lin(hidden, ndim)
        )
        self.sample = True
    
    #[docs]
    def forward(self, x, edge_index):
        #x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        raw_msg = self.msg_fnc(tmp)
        mu = raw_msg[:, 0::2]  #extracting mu out through even indices
        logvar = raw_msg[:, 1::2] #log variance through odd indices
        actual_msg = mu
        if self.sample:
            actual_msg += torch.randn(mu.shape).to(x_i.device)*torch.exp(logvar/2)

        return actual_msg
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        tmp = torch.cat([x, aggr_out], dim=1)
        return self.node_fnc(tmp) #[n, nupdate]


class varOGN(varGN):
    def __init__(
		self, n_f, msg_dim, ndim, dt,
		edge_index, aggr='add', hidden=300, nt=1):

        super(varOGN, self).__init__(n_f, msg_dim, ndim, hidden=hidden, aggr=aggr)
        self.dt = dt
        self.nt = nt
        self.edge_index = edge_index
        self.ndim = ndim
    
    def just_derivative(self, g, augment=False):
        #x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*3
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
        
        edge_index = g.edge_index
        
        return self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x)
    
    def loss(self, g, augment=True, square=False, **kwargs):
        if square:
            return torch.sum((g.y - self.just_derivative(g, augment=augment))**2)
        else:
            return torch.sum(torch.abs(g.y - self.just_derivative(g, augment=augment)))


class HGN(MessagePassing): #Hamiltonian Graph Neural Network
    def __init__(self, n_f, ndim, hidden=300):
        super(HGN, self).__init__(aggr='add')  # "Add" aggregation.
        self.pair_energy = Seq(
            Lin(2*n_f, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, 1)
        )
        
        self.self_energy = Seq(
            Lin(n_f, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, hidden),
            Softplus(),
            Lin(hidden, 1)
        )
        self.ndim = ndim
    
    def forward(self, x, edge_index):
        #x is [n, n_f]
        x = x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)
      
    def message(self, x_i, x_j):
        # x_i has shape [n_e, n_f]; x_j has shape [n_e, n_f]
        tmp = torch.cat([x_i, x_j], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.pair_energy(tmp)
    
    def update(self, aggr_out, x=None):
        # aggr_out has shape [n, msg_dim]

        sum_pair_energies = aggr_out
        self_energies = self.self_energy(x)
        return sum_pair_energies + self_energies
    
    def just_derivative(self, g, augment=False, augmentation=3):
        #x is [n, n_f]f
        x = g.x
        ndim = self.ndim
        if augment:
            augmentation = torch.randn(1, ndim)*augmentation
            augmentation = augmentation.repeat(len(x), 1).to(x.device)
            x = x.index_add(1, torch.arange(ndim).to(x.device), augmentation)
            
        #Make momenta:
        x = Variable(torch.cat((x[:, :ndim], x[:, ndim:2*ndim]*x[:, [-1]*ndim], x[:, 2*ndim:]), dim=1), requires_grad=True)
        
        edge_index = g.edge_index
        total_energy = self.propagate(
                edge_index, size=(x.size(0), x.size(0)),
                x=x).sum()
        
        dH = grad(total_energy, x, create_graph=True)[0]
        dH_dq = dH[:, :ndim]
        dH_dp = dH[:, ndim:2*ndim]
        
        dq_dt = dH_dp
        dp_dt = -dH_dq
        dv_dt = dp_dt/x[:, [-1]*ndim]
        return torch.cat((dq_dt, dv_dt), dim=1)
    
    def loss(self, g, augment=True, square=False, reg=True, augmentation=3, **kwargs):
        all_derivatives = self.just_derivative(g, augment=augment, augmentation=augmentation)
        ndim = self.ndim
        dv_dt = all_derivatives[:, self.ndim:]

        if reg:
            ## If predicting dq_dt too, the following regularization is important:
            edge_index = g.edge_index
            x = g.x
            #make momenta:
            x = Variable(torch.cat((x[:, :ndim], x[:, ndim:2*ndim]*x[:, [-1]*ndim], x[:, 2*ndim:]), dim=1), requires_grad=True)
            self_energies = self.self_energy(x)
            total_energy = self.propagate(
                    edge_index, size=(x.size(0), x.size(0)),
                    x=x)
            #pair_energies = total_energy - self_energies
            #regularization = 1e-3 * torch.sum((pair_energies)**2)
            dH = grad(total_energy.sum(), x, create_graph=True)[0]
            dH_dother = dH[2*ndim:]
            #Punish total energy and gradient with respect to other variables:
            regularization = 1e-6 * (torch.sum((total_energy)**2) + torch.sum((dH_dother)**2))
            return torch.sum(torch.abs(g.y - dv_dt)) + regularization
        else:
            return torch.sum(torch.abs(g.y - dv_dt))
        #return torch.sum(torch.abs(g.y - dv_dt))


