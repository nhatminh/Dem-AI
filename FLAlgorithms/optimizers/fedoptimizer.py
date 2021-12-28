import copy

from torch.optim import Optimizer


class MySGD(Optimizer):
    def __init__(self, params, lr, L_k = 0):
        defaults = dict(lr=lr, L_k = L_k)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                p.data = p.data - group['lr'] * (p.grad.data + group['L_k'] * p.data)
        return loss

class DemSGD(Optimizer):
    def __init__(self, params, lr, L_k = 0):
        defaults = dict(lr=lr, L_k = L_k)
        super(DemSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                p.data = p.data - group['lr'] * (p.grad.data + group['L_k'] * p.data)
        return group['params'],loss

# class DemProx_SGD(Optimizer):
#     def __init__(self, params, lr, mu = 0):
#         defaults = dict(lr=lr, mu = mu)
#         super(DemProx_SGD, self).__init__(params, defaults)
#     #in DemLearn we want the lobal weight update need to be close to the generalized weight
#     def step(self, gen_weights, closure=None):
#         loss = None
#         if closure is not None:
#             loss = closure
#         # prox_weight = prox_weight_updated
#         # if (gen_weights is not None):
#         #     print("Len of gen_weights:",len(gen_weights))
#         for group in self.param_groups:
#             print(group)
#             for p in group['params']:
#                 # prox_terms = p.data - gen_weights[0].next().parameters()
#                 prox_terms = 0
#                 if(gen_weights is not None):
#                     for (w_k,Ng) in gen_weights:
#                         w_t  = next(w_k)
#                         # print(w_t)
#                         prox_terms += 1/Ng *(p.data - w_t.data)  # 1/Ng* (w - w_prox)
#
#                 p.data = p.data - group['lr'] * (p.grad.data + group['mu']* prox_terms)  # w = w - lr * (grad L(w) + mu * prox)
#         return group['params'], loss

class DemProx_SGD(Optimizer):
    def __init__(self, params, lr, mu = 0):
        defaults = dict(lr=lr, mu = mu)
        super(DemProx_SGD, self).__init__(params, defaults)
    #in DemLearn we want the lobal weight update need to be close to the generalized weight
    def step(self, mu_t=0, gen_weights=None,closure=None):
        loss = None
        if closure is not None:
            loss = closure
        # prox_weight = prox_weight_updated
        # if (gen_weights is not None):
        #     print("Len of gen_weights:",len(gen_weights))
        for group in self.param_groups:
            # print(group)
            if (gen_weights is None or mu_t == 0):
                # print("SGD step")
                for p in group['params']:
                    p.data = p.data - group['lr'] * p.grad.data
            else:
                gen_w, fraction = gen_weights
                # print(gen_w)
                # print("fraction:",fraction)
                for p, g_w  in zip(group['params'],gen_w.parameters()):
                    prox_terms = fraction * p.data - g_w.data  # 1/Ng* w - 1/Ng*w_prox)
                    # print(prox_terms)
                    # p.data = p.data - group['lr'] * (p.grad.data + group['mu']* prox_terms)  # w = w - lr * (grad L(w) + mu * prox)
                    p.data = p.data - group['lr'] * (p.grad.data + mu_t* prox_terms)
                # for (p, g_w) in zip(group['params'],gen_w.parameters()):
                #     # print(p.data)
                #     p.data = p.data - group['lr'] * p.grad.data
        return group['params'], loss


class Prox_SGD(Optimizer):
    def __init__(self, params, lr, mu = 0):
        defaults = dict(lr=lr, mu = mu)
        super(Prox_SGD, self).__init__(params, defaults)
    #in DemLearn we want the lobal weight update need to be close to the generalized weight
    def step(self, prox_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        prox_weight = prox_weight_updated
        # print(prox_weight)
        for group in self.param_groups:
            # print(group)
            for p, prox_w in zip( group['params'], prox_weight):
                p.data = p.data - group['lr'] * (p.grad.data + group['mu']* (p.data - prox_w.data))  # w = w - lr * (grad L(w) + mu * (w - w_prox))
        return group['params'], loss

        # for group in self.param_groups:
        #     # print(group)
        #     for p in group['params']:
        #         #implement proximal here
        #         p.data = p.data - group['lr'] * (p.grad.data + group['L_k'] * p.data)
        # return loss


class PerFedAvg(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(PerFedAvg, self).__init__(params, defaults)

    def step(self, closure=None, beta = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(beta != 0):
                    p.data.add_(-beta, d_p)
                else:     
                    p.data.add_(-group['lr'], d_p)
        return loss

class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data = p.data - group['lr'] * \
                         (p.grad.data + group['eta'] * self.server_grads[i] - self.pre_grads[i])
                # p.data.add_(-group['lr'], p.grad.data)
                i += 1
        return loss

class pFedMeOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, L_k=0.1 , mu = 0.001):
        #self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, L_k=L_k, mu = mu)
        super(pFedMeOptimizer, self).__init__(params, defaults)
    
    def step(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = p.data - group['lr'] * (p.grad.data + group['L_k'] * (p.data - localweight.data) + group['mu']*p.data)
        return  group['params'], loss
    
    def update_param(self, local_weight_updated, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        weight_update = local_weight_updated.copy()
        for group in self.param_groups:
            for p, localweight in zip( group['params'], weight_update):
                p.data = localweight.data
        #return  p.data
        return  group['params']


class APFLOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(APFLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None, beta = 1, n_k = 1):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = beta  * n_k * p.grad.data
                p.data.add_(-group['lr'], d_p)
        return loss
