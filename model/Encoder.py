import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Graph():
    """
    The Graph to model the skeletons extracted by the openpose

    Args:
        strategy (string): must be one of the follow candidates
        - uniform: Uniform Labeling
        - distance: Distance Partitioning
        - spatial: Spatial Configuration
        For more information, please refer to the section 'Partition Strategies'
            in our paper (https://arxiv.org/abs/1801.07455).

        layout (string): must be one of the follow candidates
        - openpose: Is consists of 18 joints. For more information, please
            refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose#output
        - ntu-rgb+d: Is consists of 25 joints. For more information, please
            refer to https://github.com/shahroudy/NTURGB-D

        max_hop (int): the maximal distance between two connected nodes
        dilation (int): controls the spacing between the kernel points
    """
    
    def __init__(self,
                 layout='openpose',
                 strategy='uniform',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node,
                                        self.edge,
                                        max_hop=self.num_node)
        self.get_adjacency(strategy)

    def __str__(self):
        return repr(self.A)

    def get_edge(self, layout):
        # edge is a list of [child, parent] paris

        if layout == 'openpose':
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [(4, 3), (3, 2), (7, 6), (6, 5),
                             (13, 12), (12, 11), (10, 9), (9, 8), (11, 5),
                             (8, 2), (5, 1), (2, 1), (0, 1), (15, 0), (14, 0),
                             (17, 15), (16, 14)]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == 'ntu-rgb+d':
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (2, 21), (3, 21),
                              (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                              (10, 9), (11, 10), (12, 11), (13, 1), (14, 13),
                              (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
                              (20, 19), (22, 23), (23, 8), (24, 25), (25, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == 'ntu_edge':
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [(1, 2), (3, 2), (4, 3), (5, 2), (6, 5), (7, 6),
                              (8, 7), (9, 2), (10, 9), (11, 10), (12, 11),
                              (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                              (18, 17), (19, 18), (20, 19), (21, 22), (22, 8),
                              (23, 24), (24, 12)]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        elif layout == 'coco':
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
                              [6, 12], [7, 13], [6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        elif layout == 'coco_upper':
            self.num_node = 11
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [[6, 7], [8, 6], [9, 7],
                              [10, 8], [11, 9], [2, 3], [2, 1], [3, 1], [4, 2],
                              [5, 3], [4, 6], [5, 7]]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 0
        # elif layout=='customer settings'
        #     pass
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy):
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis ==
                                                                hop]
            self.A = A
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[
                                    i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[
                                    i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            A = np.stack(A)
            self.A = A
        else:
            raise ValueError("This strategy does not exist.")


def get_hop_distance(num_node, edge, max_hop=1):
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    # compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-1)
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i]**(-0.5)
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


def zero(x):
    return 0


def iden(x):
    return x


class ConvTemporalGraphical(nn.Module):
    r"""The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution. Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides of
            the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes. 
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        assert A.size(0) == kernel_size
        self.kernel_size = kernel_size
        self.A = A
        self.conv = nn.Conv2d(in_channels,
                              out_channels * kernel_size,
                              kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0),
                              stride=(t_stride, 1),
                              dilation=(t_dilation, 1),
                              bias=bias)

    def forward(self, x):
        x = self.conv(x)

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum('nkctv,kvw->nctw', (x, self.A))

        return x.contiguous()


class Body_Block(nn.Module):
    """
    Applies a spatial temporal graph convolution over an input graph sequence.
    
    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)` format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}, V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]`,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 A3, A5,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(Body_Block, self).__init__()

        self.gcn_3 = ConvTemporalGraphical(in_channels, out_channels, A3, 3, bias=False)

        self.tcn_3 = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (3, 1),
                (stride, 1),
                padding=(1, 0),
                bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001),
            nn.Dropout(dropout, inplace=True),
        )
        
        self.gcn_5 = ConvTemporalGraphical(in_channels, out_channels, A5, 5, bias=False)

        self.tcn_5 = nn.Sequential(
            nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (5, 1),
                (stride, 1),
                padding=(2, 0),
                bias=False
            ),
            nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = iden

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001),
            )
        
        self.last = nn.Sequential(
                nn.Conv2d(out_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1),
                          bias=False),
                nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001),
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        
        x_3 = self.gcn_3(x)
        x_3 = self.tcn_3(x_3)
        
        x_5 = self.gcn_5(x)
        x_5 = self.tcn_5(x_5)
        
        x = self.last(x_3 + x_5 + res)
        
        return self.relu(x)


class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()

        self.m_3 = nn.Conv2d(in_channels, out_channels, kernel_size = (3, 1), padding = (1, 0), bias = False)
        self.bn_m_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_3 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 3), padding = (0, 1), bias = False)
        self.bn_t_3 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.m_5 = nn.Conv2d(in_channels, out_channels, kernel_size = (5, 1), padding = (2, 0), bias = False)
        self.bn_m_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_5 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 5), padding = (0, 2), bias = False)
        self.bn_t_5 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.last = nn.Conv2d(out_channels, out_channels, kernel_size = (1, 1), padding = (0, 0), bias = False)
        self.bn_last = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):
        x_3 = self.relu(self.bn_m_3(self.m_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_m_5(self.m_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5
        x = self.relu(self.bn_last(self.last(x)))

        return x


class Face_Block(nn.Module):
    def __init__(self, in_channels, out_channels, is_down = False):
        super(Face_Block, self).__init__()

        self.relu = nn.ReLU()

        if is_down:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 1, 1), padding = (1, 0, 0), bias = False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), stride = (1, 2, 2), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size = (5, 1, 1), padding = (2, 0, 0), bias = False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
        else:
            self.s_3 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 3, 3), padding = (0, 1, 1), bias = False)
            self.bn_s_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_3 = nn.Conv3d(out_channels, out_channels, kernel_size = (3, 1, 1), padding = (1, 0, 0), bias = False)
            self.bn_t_3 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

            self.s_5 = nn.Conv3d(in_channels, out_channels, kernel_size = (1, 5, 5), padding = (0, 2, 2), bias = False)
            self.bn_s_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
            self.t_5 = nn.Conv3d(out_channels, out_channels, kernel_size = (5, 1, 1), padding = (2, 0, 0), bias = False)
            self.bn_t_5 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

        self.last = nn.Conv3d(out_channels, out_channels, kernel_size = (1, 1, 1), padding = (0, 0, 0), bias = False)
        self.bn_last = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):
        x_3 = self.relu(self.bn_s_3(self.s_3(x)))
        x_3 = self.relu(self.bn_t_3(self.t_3(x_3)))

        x_5 = self.relu(self.bn_s_5(self.s_5(x)))
        x_5 = self.relu(self.bn_t_5(self.t_5(x_5)))

        x = x_3 + x_5

        x = self.relu(self.bn_last(self.last(x)))

        return x


class body_encoder(nn.Module):
    """
    Spatial temporal graph convolutional networks.
    
    Args:
        in_channels (int): Number of channels in the input data
        num_class (int): Number of classes for the classification task
        graph_cfg (dict): The arguments for building the graph
        **kwargs (optional): Other parameters for graph convolution units

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    
    def __init__(self,
                 in_channels,
                 num_class,
                 graph_cfg,
                 data_bn=True,
                 **kwargs):
        super().__init__()
        
        # load graph
        self.graph_3 = Graph(max_hop = 1, **graph_cfg)
        A3 = torch.tensor(self.graph_3.A, dtype=torch.float32).cuda()
        self.register_buffer('A3', A3)
        
        self.graph_5 = Graph(max_hop = 2, **graph_cfg)
        A5 = torch.tensor(self.graph_5.A, dtype=torch.float32).cuda()
        self.register_buffer('A5', A5)
        
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      A3.size(1)) if data_bn else iden
        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        
        self.body_networks = nn.ModuleList((
            Body_Block(in_channels, 32, A3.clone(), A5.clone(), 1, residual=False, **kwargs0),
            Body_Block(32, 64, A3.clone(), A5.clone(), 1, residual=False, **kwargs),
            Body_Block(64, 128, A3.clone(), A5.clone(), 1, residual=False, **kwargs),
        ))
        
        self.__init_weight()

    def forward(self, x):
        # data normalization
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)
        
        # forward
        for gcn in self.body_networks:
            x = gcn(x)
        
        # global pooling
        x = torch.mean(x, -1).unsqueeze(3)
        x = x.view(N, M, -1, T, 1).mean(dim=1)
        
        x = x.view(x.size(0), x.size(2), -1)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    def __init__(self):
        super(audio_encoder, self).__init__()
        
        self.block1 = Audio_Block(1, 32)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        
        self.block2 = Audio_Block(32, 64)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        
        self.block3 = Audio_Block(64, 128)
        
        self.__init_weight()
            
    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        
        x = torch.mean(x, dim = 2, keepdim = True)
        x = x.squeeze(2).transpose(1, 2)
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class face_encoder(nn.Module):
    def __init__(self):
        super(face_encoder, self).__init__()
        
        self.block1 = Face_Block(1, 32, is_down = True)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        
        self.block2 = Face_Block(32, 64, is_down = False)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        
        self.block3 = Face_Block(64, 128, is_down = False)
        
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        self.__init_weight()     

    def forward(self, x):
        x = self.block1(x)
        x = self.pool1(x)
        
        x = self.block2(x)
        x = self.pool2(x)
        
        x = self.block3(x)
        
        x = x.transpose(1,2)
        B, T, C, W, H = x.shape  
        x = x.reshape(B*T, C, W, H)
        
        x = self.maxpool(x)
        
        x = x.view(B, T, C)  
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
