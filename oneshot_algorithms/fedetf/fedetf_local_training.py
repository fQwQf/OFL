from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc
from common_libs import *

def balanced_softmax_loss(logits, labels, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = torch.nn.functional.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def fedetf_local_training(model, training_data, test_dataloader, 
                        start_epoch, local_epochs, 
                        optim_name, lr, momentum, 
                        loss_name, device, sample_per_class, num_classes=10):
    
    model.train()
    model.to(device)

    try:
        if hasattr(model, "proto_classifier"):
            proto = getattr(model.proto_classifier, "proto", None)
            if isinstance(proto, torch.Tensor):
                model.proto_classifier.proto = proto.to(device)
    except Exception:
        pass

    if sample_per_class is not None and isinstance(sample_per_class, torch.Tensor):
        sample_per_class = sample_per_class.to(device)
    
    optimizer = init_optimizer(model, optim_name, lr, momentum)
    
    
    for e in range(start_epoch, local_epochs):
        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(training_data):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            
            output_total, feature = model(data)
            
            # output_total = torch.matmul(feature, model.proto_classifier.proto)
            # output_total = model.scaling_train * output_total
            
            if loss_name == 'ce':
                loss_local = torch.nn.functional.cross_entropy(output_total, target)
            elif loss_name == 'balanced':
                loss_local = balanced_softmax_loss(output_total, target, sample_per_class)

            loss_local.backward()
            optimizer.step()
            
            total_loss += loss_local.item()
        
        #　the test function should rewrite to as the training phase
        train_acc = test_acc(model, testloader=test_dataloader, device=device, mode='etf')
        train_set_acc = test_acc(model, testloader=training_data, device=device, mode='etf')
        
        
        logger.info(f"Epoch {e} loss: {total_loss}; train accuracy: {train_set_acc}; test accuracy: {train_acc}")
        
    return model    
        
              
          