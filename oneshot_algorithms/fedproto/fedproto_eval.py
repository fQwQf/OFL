from common_libs import *

def eval_with_proto(model, protos, data_loader, num_classes, device, mode='avg'):
    model.eval()

    test_acc = 0
    test_num = 0

    if protos is not None:
        with torch.no_grad():
            for x, y in data_loader:
                if type(x) == type([]):
                    x[0] = x[0].to(device)
                else:
                    x = x.to(device)
                y = y.to(device)

                if mode == 'avg':
                    rep = model.encoder(x)
                elif mode == 'ensemble':
                    rep = model(x)

                output = float('inf') * torch.ones(y.shape[0], num_classes).to(device)
                for i, r in enumerate(rep):
                    for j, pro in protos.items():
                        if type(pro) != type([]):
                            v = torch.nn.functional.mse_loss(r, pro)
                            output[i, j] = v.item()

                test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                test_num += y.shape[0]
                

        return float(test_acc) / test_num
    else:
        return 0 