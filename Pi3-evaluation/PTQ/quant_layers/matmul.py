import numpy as np
import torch
from torch import nn
from torch import Tensor 
from torch.nn import functional as F
from itertools import product     

class MinMaxQuantMatMul(nn.Module):
    """Matrix Multiplication base class"""
    def __init__(self, A_bit=8, B_bit=8, mode="raw"):
        super().__init__()
        self.A_bit=A_bit
        self.B_bit=B_bit
        self.A_interval=None
        self.B_interval=None
        self.A_qmax=2**(self.A_bit-1)
        self.B_qmax=2**(self.B_bit-1)
        self.mode=mode
        self.raw_input = None
        self.raw_out = None
    
    def forward(self, A,B):
        if self.mode=='raw':
            out=A @ B
        elif self.mode=="quant_forward":
            out=self.quant_forward(A,B)
        elif self.mode=="calibration_step1":
            out=self.calibration_step1(A,B)
        elif self.mode=="calibration_step2":
            out=self.calibration_step2(A,B)
        else:
            raise NotImplementedError
        return out
    
    def quant_input(self,x,interval,qmax):
        x_sim=(x/interval).round_().clamp_(-qmax,qmax-1)
        x_sim.mul_(interval)
        return x_sim
    
    def quant_forward(self,A,B):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        A_sim=self.quant_input(A,self.A_interval,self.A_qmax)
        B_sim=self.quant_input(B,self.B_interval,self.B_qmax)
        out=A_sim@B_sim
        return out

    def calibration_step1(self,A,B):
        # step1: collection the FP32 values
        self.raw_input=A.cpu().detach(), B.cpu().detach()
        out=A@B
        self.raw_out=out.cpu().detach()
        return out
    
    def calibration_step2(self,A,B):
        # step2: search for the best S^w and S^o of each layer
        self.A_interval=(A.data.abs().max()/(self.A_qmax-0.5)).detach()
        self.B_interval=(B.data.abs().max()/(self.B_qmax-0.5)).detach()
        self.calibrated=True
        out=self.quant_forward(A,B)        
        return out

class PTQSLQuantMatMul(MinMaxQuantMatMul):
    """
    Chunk matrix into blockes and quantize.
    Chunking follows naive padding strategy.
    Alternately search for best intervals of each individual blocks for A and B.

    two different scenarios:
    - Q @ K:
        - A's shape: B,H,S,W
        - B's shape: B,H,W,S
    - scores @ V:
        - A's shape: B,H,S,S
        - B's shape: B,H,S,W
    - interval shape: 1,n_G,1,n_V,1,n_H,1
    """
    def __init__(self, A_bit=8, B_bit=8, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode)
        self.metric = metric
        self.search_round = search_round
        self.eq_alpha = eq_alpha
        self.eq_beta = eq_beta
        self.eq_n = eq_n
        self.parallel_eq_n = parallel_eq_n
        self.n_G_A = n_G_A
        self.n_V_A = n_V_A
        self.n_H_A = n_H_A
        self.n_G_B = n_G_B
        self.n_V_B = n_V_B
        self.n_H_B = n_H_B
        # init these parameters in self.calibration_step1
        self.crb_groups_A = None
        self.crb_groups_B = None
        self.crb_rows_A = None
        self.crb_cols_A = None
        self.crb_rows_B = None
        self.crb_cols_B = None
        self.pad_groups_A = None
        self.pad_groups_B = None
        self.pad_rows_A = None
        self.pad_rows_B = None
        self.pad_cols_A = None
        self.pad_cols_B = None
        self.raw_grad = None
        self.init_layerwise = init_layerwise

    def _get_padding_parameters(self, A, B):
        self.crb_groups_A = (A.shape[1]+self.n_G_A-1) // self.n_G_A
        self.crb_groups_B = (B.shape[1]+self.n_G_B-1) // self.n_G_B
        self.crb_rows_A = (A.shape[2]+self.n_V_A-1) // self.n_V_A
        self.crb_cols_A = (A.shape[3]+self.n_H_A-1) // self.n_H_A
        self.crb_rows_B = (B.shape[2]+self.n_V_B-1) // self.n_V_B
        self.crb_cols_B = (B.shape[3]+self.n_H_B-1) // self.n_H_B

        self.pad_groups_A = self.crb_groups_A*self.n_G_A - A.shape[1]
        self.pad_rows_A = self.crb_rows_A*self.n_V_A - A.shape[2]
        self.pad_cols_A = self.crb_cols_A*self.n_H_A - A.shape[3]
        self.pad_groups_B = self.crb_groups_B*self.n_G_B - B.shape[1]
        self.pad_rows_B = self.crb_rows_B*self.n_V_B - B.shape[2]
        self.pad_cols_B = self.crb_cols_B*self.n_H_B - B.shape[3]

    def quant_input_A(self, x):
        x = F.pad(x, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A])
        x = x.view(-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
        x = (x/self.A_interval).round_().clamp(-self.A_qmax,self.A_qmax-1).mul_(self.A_interval)
        x = x.view(-1,self.n_G_A*self.crb_groups_A,self.n_V_A*self.crb_rows_A,self.n_H_A*self.crb_cols_A)
        x = x[:,:x.shape[1]-self.pad_groups_A,:x.shape[2]-self.pad_rows_A,:x.shape[3]-self.pad_cols_A]
        return x
    
    def quant_input_B(self, x):
        x = F.pad(x, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B])
        x = x.view(-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
        x = (x/self.B_interval).round_().clamp(-self.B_qmax,self.B_qmax-1).mul_(self.B_interval)
        x = x.view(-1,self.n_G_B*self.crb_groups_B,self.n_V_B*self.crb_rows_B,self.n_H_B*self.crb_cols_B)
        x = x[:,:x.shape[1]-self.pad_groups_B,:x.shape[2]-self.pad_rows_B,:x.shape[3]-self.pad_cols_B]
        return x

    def quant_forward(self, A, B):
        assert self.calibrated is not None,f"You should run calibrate_forward before run quant_forward for {self}"
        A_sim=self.quant_input_A(A)
        B_sim=self.quant_input_B(B)
        out=A_sim@B_sim
        return out

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, dim=-1):
        """
        tensor_raw: *, features, *
        tensor_sim: *, features, *
        similarity: *
        It's your job to calculate mean on non-feature * dims!

        Similarity without inherent feature structure is more welcome to parallelism.
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=dim) # should only support dim=-1 and cannot be paralleled
        elif metric == "pearson":
            similarity = F.cosine_similarity(tensor_raw-torch.mean(tensor_raw), tensor_sim-torch.mean(tensor_sim), dim=dim)
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                raw_grad = self.raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=dim)
        return similarity

    def _search_best_A_interval(self, A, B, A_interval_candidates):
        """
        使用三分法在 [eq_alpha, eq_beta] 上搜索最优 A interval
        """
        A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(
            1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
        tmp_A_interval = self.A_interval.unsqueeze(0)
        B_sim = self.quant_input_B(B).unsqueeze(0)
        
        for v, h in product(range(self.n_V_A), range(self.n_H_A)):
            l, r = self.eq_alpha, self.eq_beta
            for _ in range(20):  # 迭代20次即可
                m1 = l + (r - l) / 3
                m2 = r - (r - l) / 3
                # 计算两点的相似度
                def get_sim(scale):
                    cur_interval = tmp_A_interval.clone()
                    cur_interval[:,:,:,:,v:v+1,:,h:h+1,:] = self.A_interval.unsqueeze(0) * scale
                    A_sim = (A_pad/cur_interval).round_().clamp_(-self.A_qmax,self.A_qmax-1).mul_(cur_interval)
                    A_sim = A_sim.view(1,-1,A.shape[1]+self.pad_groups_A,A.shape[2]+self.pad_rows_A,A.shape[3]+self.pad_cols_A)
                    A_sim = A_sim[:,:,:A.shape[1],:A.shape[2],:A.shape[3]]
                    out_sim = A_sim @ B_sim
                    sim = self._get_similarity(self.raw_out, out_sim, self.metric)
                    return sim.mean().item()
                sim1, sim2 = get_sim(m1), get_sim(m2)
                if sim1 < sim2:
                    l = m1
                else:
                    r = m2
            best_scale = (l + r) / 2
            tmp_A_interval[:,:,:,:,v:v+1,:,h:h+1,:] = self.A_interval.unsqueeze(0) * best_scale
        self.A_interval = tmp_A_interval.squeeze(0)


    def _search_best_B_interval(self, A, B, B_interval_candidates):
        """
        使用三分法在 [eq_alpha, eq_beta] 上搜索最优 B interval
        """
        B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(
            1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
        tmp_B_interval = self.B_interval.unsqueeze(0)
        A_sim = self.quant_input_A(A).unsqueeze(0)
        
        for v, h in product(range(self.n_V_B), range(self.n_H_B)):
            l, r = self.eq_alpha, self.eq_beta
            for _ in range(20):
                m1 = l + (r - l) / 3
                m2 = r - (r - l) / 3
                def get_sim(scale):
                    cur_interval = tmp_B_interval.clone()
                    cur_interval[:,:,:,:,v:v+1,:,h:h+1,:] = self.B_interval.unsqueeze(0) * scale
                    B_sim = (B_pad/cur_interval).round_().clamp_(-self.B_qmax,self.B_qmax-1).mul_(cur_interval)
                    B_sim = B_sim.view(1,-1,B.shape[1]+self.pad_groups_B,B.shape[2]+self.pad_rows_B,B.shape[3]+self.pad_cols_B)
                    B_sim = B_sim[:,:,:B.shape[1],:B.shape[2],:B.shape[3]]
                    out_sim = A_sim @ B_sim
                    sim = self._get_similarity(self.raw_out, out_sim, self.metric)
                    return sim.mean().item()
                sim1, sim2 = get_sim(m1), get_sim(m2)
                if sim1 < sim2:
                    l = m1
                else:
                    r = m2
            best_scale = (l + r) / 2
            tmp_B_interval[:,:,:,:,v:v+1,:,h:h+1,:] = self.B_interval.unsqueeze(0) * best_scale
        self.B_interval = tmp_B_interval.squeeze(0)


    def _initialize_intervals(self, A, B):
        # pad A and B for future quantization
        self._get_padding_parameters(A, B) # put it here because hessian does not use calibration step 1
        A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A) # shape: 1,B,n_G,crb_groups,n_V,crb_rows,n_H,crb_cols
        B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)

        # initialize intervals with minmax intervals
        if self.init_layerwise:
            self.A_interval = (A.abs().max()/(self.A_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_A,1,self.n_V_A,1,self.n_H_A,1)
            self.B_interval = (B.abs().max()/(self.B_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_B,1,self.n_V_B,1,self.n_H_B,1)
        else:
            self.A_interval=(A_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.A_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
            self.B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1

    def calibration_step2(self, A, B):
        # put raw outs/grads on GPU
        self.raw_out = self.raw_out.unsqueeze(0).to(A.device)
        self.raw_grad = self.raw_grad.to(A.device) if self.raw_grad != None else None

        self._initialize_intervals(A, B)

        # prepare weight intervals and similarities
        A_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.A_interval.unsqueeze(0)
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)

        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A, B, A_interval_candidates)
            # search for best B interval
            self._search_best_B_interval(A, B, B_interval_candidates)

        # put raw data back to cpu
        self.raw_out = self.raw_out.squeeze(0).to("cpu")
        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        # finish calibration and output the result
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        out=self.quant_forward(A,B)
        return out    

class SoSPTQSLQuantMatMul(PTQSLQuantMatMul):
    """
    Sublayerwise PTQ on matmul modules with Split-of-Softmax (SoS) on score matrix.
    
    Data after softmaxing has highly biased distribution, making it difficult to quantize with uniform quantization.
    An elegant tradeoff between great majority of unimportant values and few crucial values is impossible under low bit quantization.
    Therefore, we propose to split complete interval of (0, 1) into several smaller intervals and perform uniform quantization on each.
    We could manually assgin or search for the best split point.
    Currently, we only consider single split point scenarios, since this proves to be effective enough.

    The algorithm no longer requires PTQSL on score matrix, and will ignore relevant parameters.

    with proper hardware implementation, we don't need to use a sign bit anymore.
    """
    def __init__(self, A_bit=8, B_bit=8, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False,
                 split=None):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, 
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, 
                         n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        self.A_qmax = 2**(self.A_bit-1) # well, still need it 
        self.split = split
        if split != None:
            self.A_interval = self.split/(self.A_qmax-1)

    def quant_input_A(self, x):
        x_high = (x.clamp(self.split, 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
        x_low = (x.clamp(0, self.split)/self.A_interval).round_().clamp_(0,self.A_qmax-1)*self.A_interval
        return x_high + x_low

    def _search_best_A_interval(self, A, B, split_candidates=None):
        """
        使用三分法搜索最佳 split 点
        """
        A_ = A.unsqueeze(0)
        B_sim = B.unsqueeze(0)

        l, r = 2**(-20), 1.0  # split 搜索范围 (0, 1)
        for _ in range(25):
            m1 = l + (r - l) / 3
            m2 = r - (r - l) / 3
            def get_sim(split):
                cur_interval = split / (self.A_qmax - 1)
                A_high = (A_.clamp(split, 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
                A_low  = (A_.clamp(0, split)/cur_interval).round_().clamp_(0,self.A_qmax-1)*cur_interval
                A_sim = A_high + A_low
                out_sim = A_sim @ B_sim
                return self._get_similarity(self.raw_out, out_sim, self.metric).mean().item()
            sim1, sim2 = get_sim(m1), get_sim(m2)
            if sim1 < sim2:
                l = m1
            else:
                r = m2
        self.split = (l + r) / 2
        self.A_interval = self.split / (self.A_qmax - 1)


    def _initialize_intervals(self, A, B):
        # pad A and B for future quantization
        self._get_padding_parameters(A, B)
        B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)

        # initialize intervals with minmax intervals
        self.split = 0.01
        self.A_interval = self.split/(self.A_qmax-1)
        if self.init_layerwise:
            self.B_interval = (B.abs().max()/(self.B_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_B,1,self.n_V_B,1,self.n_H_B,1)
        else:
            self.B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
    
    def calibration_step2(self, A, B):
        # put raw outs/grads on GPU
        self.raw_out = self.raw_out.unsqueeze(0).to(A.device)
        self.raw_grad = self.raw_grad.to(A.device) if self.raw_grad != None else None

        self._initialize_intervals(A, B)

        # prepare weight intervals and similarities
        A_split_candidates = torch.tensor([2**(-i) for i in range(20)]).cuda()
        # split_eq_alpha, split_eq_beta, split_eq_n = 0.002, 0.03, 50
        # A_split_candidates = torch.tensor([split_eq_alpha + (split_eq_beta- split_eq_alpha)*i/split_eq_n for i in range(split_eq_n + 1)]).cuda()
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)

        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A, B, A_split_candidates)
            # search for best B interval
            self._search_best_B_interval(A, B, B_interval_candidates)

        # put raw data back to cpu
        self.raw_out = self.raw_out.squeeze(0).to("cpu")
        self.raw_grad = self.raw_grad.to("cpu") if self.raw_grad != None else None

        # finish calibration and output the result
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        out=self.quant_forward(A,B)
        return out    

class PTQSLBatchingQuantMatMul(PTQSLQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)

    def _initialize_calib_parameters(self):
        """ 
        set parameters for feeding calibration data
        """
        self.calib_size = int(self.raw_input[0].shape[0])
        self.calib_batch_size = int(self.raw_input[0].shape[0])
        while True:
            numel = ((self.raw_input[0].numel()+self.raw_input[1].numel()+2*self.raw_out.numel())/self.calib_size*self.calib_batch_size) # number of parameters on GPU
            self.parallel_eq_n = int((3*1024*1024*1024/4)//numel)
            if self.parallel_eq_n <= 1:
                self.calib_need_batching = True
                self.calib_batch_size //= 2
            else:
                break

    def _get_padding_parameters(self, A, B):
        """
        We adopt a head-wise quantization here
        """
        self.n_G_A = A.shape[1]
        self.n_G_B = B.shape[1]
        super()._get_padding_parameters(A,B)
    
    def _initialize_intervals(self):
        # pad A and B for future quantization
        self._get_padding_parameters(self.raw_input[0], self.raw_input[1]) # put it here because hessian does not use calibration step 1

        # initialize intervals with minmax intervals
        tmp_A_intervals = []
        tmp_B_intervals = []
        for b_st in range(0,self.calib_size,self.calib_batch_size):
            b_ed = min(self.calib_size, b_st+self.calib_batch_size)
            A, B = self.raw_input[0][b_st:b_ed].cuda(), self.raw_input[1][b_st:b_ed].cuda()
            if self.init_layerwise:
                A_interval = (A.abs().max()/(self.A_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_A,1,self.n_V_A,1,self.n_H_A,1)
                B_interval = (B.abs().max()/(self.B_qmax-0.5)).detach().view(1,1,1,1,1,1,1).repeat(1,self.n_G_B,1,self.n_V_B,1,self.n_H_B,1)
            else:
                A_pad = F.pad(A, [0,self.pad_cols_A,0,self.pad_rows_A,0,self.pad_groups_A]).unsqueeze(0).view(1,-1,self.n_G_A,self.crb_groups_A,self.n_V_A,self.crb_rows_A,self.n_H_A,self.crb_cols_A)
                B_pad = F.pad(B, [0,self.pad_cols_B,0,self.pad_rows_B,0,self.pad_groups_B]).unsqueeze(0).view(1,-1,self.n_G_B,self.crb_groups_B,self.n_V_B,self.crb_rows_B,self.n_H_B,self.crb_cols_B)
                A_interval=(A_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.A_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
                B_interval=(B_pad.abs().amax([0,1,3,5,7], keepdim=True)/(self.B_qmax-0.5)).detach().squeeze(0) # shape: 1,n_G,1,n_V,1,n_H,1
            tmp_A_intervals.append(A_interval)
            tmp_B_intervals.append(B_interval)
        self.A_interval = torch.cat(tmp_A_intervals, dim=0).amax(0, keepdim=True)
        self.B_interval = torch.cat(tmp_B_intervals, dim=0).amax(0, keepdim=True)

    def _get_similarity(self, tensor_raw, tensor_sim, metric=None, dim=-1, raw_grad=None):
        """
        tensor_raw: *, features, *
        tensor_sim: *, features, *
        similarity: *
        It's your job to calculate mean on non-feature * dims!

        Similarity without inherent feature structure is more welcome to parallelism.
        """
        if metric == "cosine":
            similarity = F.cosine_similarity(tensor_raw, tensor_sim, dim=dim) # should only support dim=-1 and cannot be paralleled
        elif metric == "pearson":
            similarity = F.cosine_similarity(tensor_raw-torch.mean(tensor_raw,dim=dim,keepdim=True), tensor_sim-torch.mean(tensor_sim,dim=dim,keepdim=True), dim=dim) # should only support dim=-1 and cannot be paralleled
            # a quick implementation of pearson similarity
            # tensor_raw: 1,B,H,dim1,dim3
            # tensor_sim: parallel_eq_n,B,H,dim1,dim3
            # parallel_eq_n,B,H,dim1,dim3 = tensor_sim.shape
            # tensor_sim = tensor_sim.view(parallel_eq_n,B,-1)
            # tensor_raw = tensor_raw.view(1,B,-1)
            # tensor_sim_mean = tensor_sim.mean(dim=[1,2],keepdim=True)
            # tensor_raw_mean = tensor_raw.mean(dim=[1,2],keepdim=True)
            # similarity = F.cosine_similarity(tensor_raw-tensor_raw_mean,tensor_sim-tensor_sim_mean,dim=-1) # shape: parallel_eq_n,B
            # similarity = similarity.reshape(parallel_eq_n,B,1,1) # restore two dims
        else:
            if metric == "L1_norm":
                similarity = -torch.abs(tensor_raw - tensor_sim)
            elif metric == "L2_norm":
                similarity = -(tensor_raw - tensor_sim) ** 2
            elif metric == "linear_weighted_L2_norm":
                similarity = -tensor_raw.abs() * (tensor_raw - tensor_sim) ** 2
            elif metric == "square_weighted_L2_norm":
                similarity = -(tensor_raw * (tensor_raw - tensor_sim)) ** 2
            elif metric == "hessian":
                assert raw_grad != None, f"No raw_grad in PTQSLBatchingQuantMatMul!"
                raw_grad = raw_grad.reshape_as(tensor_raw)
                similarity = -(raw_grad * (tensor_raw - tensor_sim)) ** 2
            else:
                raise NotImplementedError(f"metric {metric} not implemented!")
            similarity = torch.mean(similarity, dim=dim)
        return similarity

    def _search_best_A_interval(self, A_interval_candidates):

        # tmp_A_interval shape: 1,1,n_G,1,n_V,1,n_H,1
        tmp_A_interval = self.A_interval.unsqueeze(0)

        # 遍历 v,h
        for v, h in product(range(self.n_V_A), range(self.n_H_A)):

            # === 遍历 calibration batches ===
            # 注：三分搜索对该 (v,h) 的所有 batch 一起统计 score
            A_batches = []
            B_batches = []
            raw_out_batches = []
            raw_grad_batches = []

            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)

                A_batches.append(self.raw_input[0][b_st:b_ed].cuda())
                B_batches.append(self.raw_input[1][b_st:b_ed].cuda())
                raw_out_batches.append(self.raw_out[b_st:b_ed].unsqueeze(0).cuda())
                raw_grad_batches.append(self.raw_grad[b_st:b_ed].cuda())

            # 预量化 B
            B_sim_batches = [self.quant_input_B(B).unsqueeze(0) for B in B_batches]

            # === 定义 eval(p_st,p_ed) ===
            def eval_idx_range(idx_st, idx_ed):
                """
                评估某个 interval 索引范围，返回 score (float)
                """
                total_score = 0.

                for A, B_sim, raw_out, raw_grad in zip(
                    A_batches, B_sim_batches, raw_out_batches, raw_grad_batches
                ):
                    # ----- pad A -----
                    A_pad = F.pad(
                        A, [0, self.pad_cols_A, 0, self.pad_rows_A, 0, self.pad_groups_A]
                    ).unsqueeze(0).view(
                        1, -1, self.n_G_A, self.crb_groups_A,
                        self.n_V_A, self.crb_rows_A, self.n_H_A, self.crb_cols_A
                    )

                    similarities = []

                    # 遍历候选 index 范围
                    for idx in range(idx_st, idx_ed):

                        # --- 拷贝 interval 并替换当前 (v,h) ---
                        cur_A_interval = tmp_A_interval.repeat(1,1,1,1,1,1,1,1)
                        cur_A_interval[:, :, :, :, v:v+1, :, h:h+1, :] = \
                            A_interval_candidates[idx:idx+1, :, :, :, v:v+1, :, h:h+1, :]

                        # --- 量化 A ---
                        A_sim = (A_pad / cur_A_interval).round_().clamp_(
                            -self.A_qmax, self.A_qmax - 1
                        ) * cur_A_interval

                        # reshape back
                        A_sim = A_sim.view(
                            1, -1,
                            A.shape[1] + self.pad_groups_A,
                            A.shape[2] + self.pad_rows_A,
                            A.shape[3] + self.pad_cols_A
                        )[:, :, :A.shape[1], :A.shape[2], :A.shape[3]]

                        # --- 矩阵乘 ---
                        out_sim = A_sim @ B_sim

                        # --- similarity ---
                        sim = self._get_similarity(
                            raw_out, out_sim, self.metric, raw_grad=raw_grad
                        )
                        sim = sim.mean(dim=3).sum(dim=1)  # shape: 1
                        similarities.append(sim)

                    if similarities:
                        similarities = torch.cat(similarities, dim=0)
                        total_score += similarities.sum().item()

                return total_score

            # === 三分搜索 ===
            L, R = 0, self.eq_n - 1
            all_scores = []

            while R - L > 2:
                m1 = L + (R - L) // 3
                m2 = R - (R - L) // 3

                score1 = eval_idx_range(m1, m1 + 1)
                score2 = eval_idx_range(m2, m2 + 1)
                all_scores.append((m1, score1))
                all_scores.append((m2, score2))

                if score1 < score2:
                    L = m1
                else:
                    R = m2

            # === 最后穷举 L..R ===
            best_score = float("-inf")
            best_idx = L

            for idx in range(L, R + 1):
                score = eval_idx_range(idx, idx + 1)
                all_scores.append((idx, score))

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # === 更新 interval ===
            tmp_A_interval[:, :, :, :, v:v+1, :, h:h+1, :] = \
                A_interval_candidates[best_idx:best_idx+1, :, :, :, v:v+1, :, h:h+1, :]

        # 去掉 batch dim
        self.A_interval = tmp_A_interval.squeeze(0)



    def _search_best_B_interval(self, B_interval_candidates):

        # tmp_B_interval shape: 1,1,n_G,1,n_V,1,n_H,1
        tmp_B_interval = self.B_interval.unsqueeze(0)

        # === 遍历 v,h ===
        for v, h in product(range(self.n_V_B), range(self.n_H_B)):

            # === 预先把 batch 切好 ===
            A_batches = []
            B_batches = []
            raw_out_batches = []
            raw_grad_batches = []

            for b_st in range(0, self.calib_size, self.calib_batch_size):
                b_ed = min(self.calib_size, b_st + self.calib_batch_size)

                A = self.raw_input[0][b_st:b_ed].cuda()
                B = self.raw_input[1][b_st:b_ed].cuda()

                A_batches.append(A)
                B_batches.append(B)
                raw_out_batches.append(self.raw_out[b_st:b_ed].unsqueeze(0).cuda())
                raw_grad_batches.append(self.raw_grad[b_st:b_ed].cuda())

            # 预量化 A
            A_sim_batches = [self.quant_input_A(A).unsqueeze(0) for A in A_batches]

            # === 定义评估函数：评估 idx_st..idx_ed ===
            def eval_idx_range(idx_st, idx_ed):

                total_score = 0.

                # 遍历所有 calib batch 累积 score
                for A_sim, B, raw_out, raw_grad in zip(
                    A_sim_batches, B_batches, raw_out_batches, raw_grad_batches
                ):

                    # --- pad B ---
                    B_pad = F.pad(
                        B, [0, self.pad_cols_B, 0, self.pad_rows_B, 0, self.pad_groups_B]
                    ).unsqueeze(0).view(
                        1, -1,
                        self.n_G_B, self.crb_groups_B,
                        self.n_V_B, self.crb_rows_B,
                        self.n_H_B, self.crb_cols_B
                    )

                    similarities = []

                    # 遍历候选 idx
                    for idx in range(idx_st, idx_ed):

                        # --- 当前 interval ---
                        cur_B_interval = tmp_B_interval.repeat(1,1,1,1,1,1,1,1)
                        cur_B_interval[:, :, :, :, v:v+1, :, h:h+1, :] = \
                            B_interval_candidates[idx:idx+1, :, :, :, v:v+1, :, h:h+1, :]

                        # --- 量化 B ---
                        B_sim = (B_pad / cur_B_interval).round_().clamp_(
                            -self.B_qmax, self.B_qmax - 1
                        ) * cur_B_interval

                        # reshape 回原始大小
                        B_sim = B_sim.view(
                            1, -1,
                            B.shape[1] + self.pad_groups_B,
                            B.shape[2] + self.pad_rows_B,
                            B.shape[3] + self.pad_cols_B
                        )[:, :, :B.shape[1], :B.shape[2], :B.shape[3]]

                        # --- out_sim = A @ B ---
                        out_sim = A_sim @ B_sim

                        # --- similarity ---
                        sim = self._get_similarity(
                            raw_out, out_sim, self.metric, raw_grad=raw_grad
                        )
                        sim = sim.mean(dim=3).sum(dim=1)
                        similarities.append(sim)

                    if similarities:
                        similarities = torch.cat(similarities, dim=0)
                        total_score += similarities.sum().item()

                return total_score

            # === 三分搜索主循环 ===
            L, R = 0, self.eq_n - 1
            all_scores = []

            while R - L > 2:
                m1 = L + (R - L) // 3
                m2 = R - (R - L) // 3

                score1 = eval_idx_range(m1, m1 + 1)
                score2 = eval_idx_range(m2, m2 + 1)
                all_scores.append((m1, score1))
                all_scores.append((m2, score2))

                if score1 < score2:
                    L = m1
                else:
                    R = m2

            # === 最后区间穷举 ===
            best_score = float("-inf")
            best_idx = L

            for idx in range(L, R + 1):
                score = eval_idx_range(idx, idx + 1)
                all_scores.append((idx, score))

                if score > best_score:
                    best_score = score
                    best_idx = idx

            # === 更新 B interval ===
            tmp_B_interval[:, :, :, :, v:v+1, :, h:h+1, :] = \
                B_interval_candidates[best_idx:best_idx+1, :, :, :, v:v+1, :, h:h+1, :]

        # 去掉 batch 维
        self.B_interval = tmp_B_interval.squeeze(0)



    def calibration_step2(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.A_interval.unsqueeze(0)
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)
        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A_interval_candidates)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        # self.raw_input = self.raw_input.cpu().detach()
        # self.raw_out = self.raw_out.cpu().detach()
        # del self.raw_out, self.raw_grad

class SoSPTQSLBatchingQuantMatMul(PTQSLBatchingQuantMatMul):
    def __init__(self, A_bit=8, B_bit=8, mode="raw",
                 metric="L2_norm", search_round=1, eq_alpha=0.1, eq_beta=2, eq_n=100, parallel_eq_n=10,
                 n_G_A=1, n_V_A=1, n_H_A=1, n_G_B=1, n_V_B=1, n_H_B=1, init_layerwise=False,
                 split=None):
        super().__init__(A_bit=A_bit, B_bit=B_bit, mode=mode, 
                         metric=metric, search_round=search_round, eq_alpha=eq_alpha, eq_beta=eq_beta, eq_n=eq_n, parallel_eq_n=parallel_eq_n, 
                         n_G_A=n_G_A, n_V_A=n_V_A, n_H_A=n_H_A, n_G_B=n_G_B, n_V_B=n_V_B, n_H_B=n_H_B, init_layerwise=init_layerwise)
        self.n_G_A = 1
        self.n_V_A = 1
        self.n_H_A = 1
        # with proper hardware implementation, we don't need to use a sign bit anymore
        self.A_qmax = 2**(self.A_bit-1)
        self.split = split
        if split != None:
            self.A_interval = self.split/(self.A_qmax-1)

    def quant_input_A(self, x):
        x_high = (x.clamp(self.split, 1)*(self.A_qmax-1)).round_().clamp_(0,self.A_qmax-1)/(self.A_qmax-1)
        x_low = (x.clamp(0, self.split)/self.A_interval).round_().clamp_(0,self.A_qmax-1)*self.A_interval
        return x_high + x_low


    def _search_best_A_interval(self, split_candidates):

        # === 预先把所有 batch 切好，避免重复索引 ===
        A_batches = []
        B_batches = []
        raw_out_batches = []
        raw_grad_batches = []

        for b_st in range(0, self.calib_size, self.calib_batch_size):
            b_ed = min(self.calib_size, b_st + self.calib_batch_size)

            A_batches.append(self.raw_input[0][b_st:b_ed].unsqueeze(0).cuda())  # 1,b,H,S,S
            B_batches.append(self.raw_input[1][b_st:b_ed].unsqueeze(0).cuda())  # 1,b,S,S,C
            raw_out_batches.append(self.raw_out[b_st:b_ed].unsqueeze(0).cuda()) # 1,b,..
            raw_grad_batches.append(self.raw_grad[b_st:b_ed].cuda())

        # === 评估函数：计算 idx_st ~ idx_ed-1 的平均 similarity ===
        def eval_idx_range(idx_st, idx_ed):
            total_score = 0.0

            for A, B, raw_out, raw_grad in zip(
                A_batches, B_batches, raw_out_batches, raw_grad_batches
            ):
                B_sim = B  # B 不动

                similarities = []

                for idx in range(idx_st, idx_ed):

                    split_value = split_candidates[idx]
                    interval = split_value / (self.A_qmax - 1)

                    # === 正区间 ===
                    A_high = (
                        A.clamp(split_value, 1) * (self.A_qmax - 1)
                    ).round_().clamp_(0, self.A_qmax - 1) / (self.A_qmax - 1)

                    # === 负区间 ===
                    A_low = (
                        (A.clamp(0, split_value) / interval)
                        .round_().clamp_(0, self.A_qmax - 1) * interval
                    )

                    # 合并
                    A_sim = A_high + A_low  # (1,b,H,S,S)

                    # === 前向 ===
                    out_sim = A_sim @ B_sim

                    # === similarity ===
                    sim = self._get_similarity(raw_out, out_sim, self.metric, raw_grad=raw_grad)
                    sim = sim.mean([2, 3]).sum()  # (1,b) → scalar
                    similarities.append(sim)

                if similarities:
                    total_score += sum(similarities)

            return total_score

        # === 三分搜索（与 A/B 搜索统一） ===
        L, R = 0, len(split_candidates) - 1

        while R - L > 2:
            m1 = L + (R - L) // 3
            m2 = R - (R - L) // 3

            s1 = eval_idx_range(m1, m1 + 1)
            s2 = eval_idx_range(m2, m2 + 1)

            if s1 < s2:
                L = m1
            else:
                R = m2

        # === 最终穷举 ===
        best_score = float("-inf")
        best_idx = L

        for idx in range(L, R + 1):
            s = eval_idx_range(idx, idx + 1)
            if s > best_score:
                best_score = s
                best_idx = idx

        # === 应用最优 split ===
        self.split = split_candidates[best_idx]
        self.A_interval = self.split / (self.A_qmax - 1)


    def calibration_step2(self):
        self._initialize_calib_parameters()
        self._initialize_intervals()
        A_split_candidates = torch.tensor([2**(-i) for i in range(20)]).cuda()
        B_interval_candidates = torch.tensor([self.eq_alpha + i*(self.eq_beta - self.eq_alpha)/self.eq_n for i in range(self.eq_n + 1)]).cuda().view(-1,1,1,1,1,1,1,1) * self.B_interval.unsqueeze(0)
        for e in range(self.search_round):
            # search for best A interval
            self._search_best_A_interval(A_split_candidates)
            # search for best B interval
            self._search_best_B_interval(B_interval_candidates)
        self.calibrated = True
        del self.raw_input, self.raw_out, self.raw_grad
        # self.raw_input = self.raw_input.cpu().detach()
        # self.raw_out = self.raw_out.cpu().detach()
        # del self.raw_out, self.raw_grad