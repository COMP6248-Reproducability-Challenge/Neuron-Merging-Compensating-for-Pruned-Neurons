#coding:utf-8
import os
import torch
from torch import optim
import torchbearer
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchbearer import Trial
from LeNet_300_100 import LeNet_300_100
import torch.nn.utils.prune as prune
from decompose import Decompose


def minst_data():
    '''
    This function is used to get the FashionMNIST.
    :return:(trainloader, testloader)
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Lambda(lambda x: x.view(28*28))
    ])

    trainset = FashionMNIST("./data/FashionMNIST", train=True, download=True, transform=transform)
    testset = FashionMNIST("./data/FashionMNIST", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
    testloader = DataLoader(testset, batch_size=128, shuffle=True)

    return (trainloader, testloader)

def train(root_path, model, args, trainloader, testloader):
    #args={"lr":0.1, "momentum":0.9, "weight_decay":5e-4}
    '''

    :param root_path: the program running path
    :param model: model to be trained
    :param args: args={"lr":0.1, "momentum":0.9, "weight_decay":5e-4}
    :param trainloader:
    :param testloader:
    :return: The accuracy of the original model
    '''
    optimiser = optim.SGD(model.parameters(),lr=args['lr'], momentum=args['momentum'], weight_decay=args['weight_decay'])
    loss_function = nn.CrossEntropyLoss()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model, optimiser, loss_function,
                  metrics=['loss', 'accuracy']).to(device)
    trial.with_generators(trainloader, test_generator=testloader)
    trial.run(epochs=10)
    save_path = os.path.join(root_path,'saved_model', 'original.pt')
    torch.save(model, save_path)
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    print(results)
    return (save_path)

def l1_prune_mode(root_path, model_path, amount=0.5):
    '''
    :param root_path: the program running path
    :param model_path: original model path
    :param amount: Pruning rate
    :return: model saved path
    '''
    model = torch.load(model_path)
    for module in [model.fc1, model.fc2]:
        prune.ln_structured(module, name="weight", amount=amount, n=1, dim=0)
    save_path = os.path.join(root_path, 'saved_model', 'l1_prune_'+str(amount)+'.pt')
    torch.save(model, save_path)
    print('complete l1 pruning, amount:', str(amount))
    print(save_path)
    return save_path

def l2_prune_mode(root_path, model_path, amount=0.5):
    '''
    :param root_path: the program running path
    :param model_path: original model path
    :param amount: Pruning rate
    :return: model saved path
    '''
    model = torch.load(model_path)
    for module in [model.fc1, model.fc2]:
        prune.ln_structured(module, name="weight", amount=amount, n=2, dim=0)
    save_path = os.path.join(root_path, 'saved_model', 'l2_prune_'+str(amount)+'.pt')
    torch.save(model, save_path)
    print('complete l2 pruning, amount:', str(amount))
    print(save_path)
    return save_path

def l1_prune_compensating(root_path, model_path, pruning_rate=0.5):
    '''
    Pruning and compensating the model according to l1-norm
    :param model_path:original model path
    :param root_path:the program running path
    :param pruning_rate:Pruning rate
    :return:model saved path
    '''
    parameter = [300, 100]
    parameter_n = []
    for i in parameter:
        parameter_n.append(int(i*(1-pruning_rate)))
    n_model = LeNet_300_100(cfg=parameter_n)
    model = torch.load(model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    decomposed_weight_list = Decompose('LeNet_300_100',
                                model.state_dict(),
                                'l1-norm',
                                0.45,#threshold
                                0.8,#lamda
                                'merge',
                                parameter_n,
                                device).main()
    for layer in n_model.state_dict():
        decomposed_weight = decomposed_weight_list.pop(0)
        n_model.state_dict()[layer].copy_(decomposed_weight)
    save_path = os.path.join(root_path, 'saved_model', 'l1_com_' + str(pruning_rate) + '.pt')
    torch.save(n_model, save_path)
    print('complete l1 prune compensating, pruning rate:', str(pruning_rate))
    print(save_path)
    return save_path

def l2_prune_compensating(root_path, model_path, pruning_rate=0.5):
    '''
    Pruning and compensating the model according to l2-norm
    :param model_path:original model path
    :param root_path:the program running path
    :param pruning_rate:Pruning rate
    :return:model saved path
    '''
    parameter = [300, 100]
    parameter_n = []
    for i in parameter:
        parameter_n.append(int(i*(1-pruning_rate)))
    n_model = LeNet_300_100(cfg=parameter_n)
    model = torch.load(model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    decomposed_weight_list = Decompose('LeNet_300_100',
                                model.state_dict(),
                                'l2-norm',
                                0.45,#threshold
                                0.8,#lamda
                                'merge',
                                parameter_n,
                                device).main()
    for layer in n_model.state_dict():
        decomposed_weight = decomposed_weight_list.pop(0)
        n_model.state_dict()[layer].copy_(decomposed_weight)
    save_path = os.path.join(root_path, 'saved_model', 'l2_com_' + str(pruning_rate) + '.pt')
    torch.save(n_model, save_path)
    print('complete l2 prune compensating, pruning rate:', str(pruning_rate))
    print(save_path)
    return save_path

def test(model_path,testloader):
    '''

    :param model_path: path of test model
    :param testloader: test dataset
    :return: accuracy
    '''
    model = torch.load(model_path)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    trial = Trial(model,metrics=['accuracy']).to(device)
    trial.with_generators(test_generator=testloader)
    trial.eval()
    results = trial.evaluate(data_key=torchbearer.TEST_DATA)
    #print(results)
    return results


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    (trainloader, testloader) = minst_data()
    root_path = os.getcwd()
    model = LeNet_300_100(cfg=None)
    #Train the original model
    args = {"lr": 0.1, "momentum": 0.9, "weight_decay": 5e-4, "parameters": model.parameters()}
    or_path = train(root_path, model, args, trainloader, testloader)
    prung_rate = [0.6, 0.7, 0.8]

    # l1 pruning
    l1_path = []
    for i in prung_rate:
        l1_path.append(l1_prune_mode(root_path, or_path, i))

    # l2 pruning
    l2_path = []
    for i in prung_rate:
            l2_path.append(l2_prune_mode(root_path, or_path, i))

    # l1 prune compensating
    l1_c_path = []
    for i in prung_rate:
        l1_c_path.append(l1_prune_compensating(root_path, or_path, i))

    # l2 prune compensating
    l2_c_path = []
    for i in prung_rate:
        l2_c_path.append(l2_prune_compensating(root_path, or_path, i))

    # test
    acc = test(or_path, testloader)
    print("original model acc: "+ str(acc))

    for i in range(len(prung_rate)):
        acc = test(l1_path[i], testloader)
        print('l1 prune acc: ' + str(acc) +' prune rate: ' + str(prung_rate[i]))

    for i in range(len(prung_rate)):
        acc = test(l2_path[i], testloader)
        print('l2 prune acc: ' + str(acc) +' prune rate: ' + str(prung_rate[i]))

    for i in range(len(prung_rate)):
        acc = test(l1_c_path[i], testloader)
        print('l1 prune compensating acc: ' + str(acc) +' prune rate: ' + str(prung_rate[i]))

    for i in range(len(prung_rate)):
        acc = test(l2_c_path[i], testloader)
        print('l2 prune compensating acc: ' + str(acc) +' prune rate: ' + str(prung_rate[i]))

