from torchvision import datasets, transforms
from FLAlgorithms.trainmodel.models import *
import torch.nn.functional as F

class KL_Loss(nn.Module):
    def __init__(self, temperature=1):
        super(KL_Loss, self).__init__()
        self.T = temperature

    def forward(self, output_batch, teacher_outputs):
        # output_batch  -> B X num_classes
        # teacher_outputs -> B X num_classes

        # loss_2 = -torch.sum(torch.sum(torch.mul(F.log_softmax(teacher_outputs,dim=1), F.softmax(teacher_outputs,dim=1)+10**(-7))))/teacher_outputs.size(0)
        # print('loss H:',loss_2)

        output_batch = F.log_softmax(output_batch / self.T, dim=1)
        teacher_outputs = F.softmax(teacher_outputs / self.T, dim=1) + 10 ** (-7)

        loss = self.T * self.T * nn.KLDivLoss(reduction='batchmean')(output_batch, teacher_outputs)

        # Same result KL-loss implementation
        # loss = T * T * torch.sum(torch.sum(torch.mul(teacher_outputs, torch.log(teacher_outputs) - output_batch)))/teacher_outputs.size(0)
        return loss

def get_model(args):
    if(args.model == "mclr"):
        if(args.dataset == "human_activity"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "gleam"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = Mclr_Logistic(100,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = Mclr_Logistic(60,10).to(args.device)
        else:#(dataset == "Mnist"):
            model = Mclr_Logistic().to(args.device)

    elif(args.model == "dnn"):
        if(args.dataset == "human_activity"):
            model = DNN(561,100,12).to(args.device)
        elif(args.dataset == "gleam"):
            model = DNN(561,20,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = DNN(100,20,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = DNN(60,20,10).to(args.device)
        else:#(dataset == "Mnist"):
            model = DNN2().to(args.device)
        
    elif(args.model == "cnn"):
        if(args.dataset == "Cifar10"):
            model = CNNCifar().to(args.device)
    else:
        pasexit('Error: unrecognized model')
    return model