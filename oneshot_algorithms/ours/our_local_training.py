from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc, save_best_local_model
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss

from common_libs import *

# 导入AMP所需的库
from torch.cuda.amp import autocast, GradScaler

def ours_local_training(model, training_data, test_dataloader, start_epoch, local_epochs, optim_name, lr, momentum, loss_name, device, num_classes, sample_per_class, aug_transformer, client_model_dir, save_freq=1, use_drcl=False, fixed_anchors=None, lambda_align=1.0, use_progressive_alignment=False, initial_protos=None):
    model.train()
    model.to(device)

    optimizer = init_optimizer(model, optim_name, lr, momentum)
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    contrastive_loss_fn = SupConLoss(temperature=0.07)
    con_proto_feat_loss_fn = Contrastive_proto_feature_loss(temperature=1.0)
    con_proto_loss_fn = Contrastive_proto_loss(temperature=1.0)

    if use_drcl or use_progressive_alignment:
        alignment_loss_fn = torch.nn.MSELoss()

    # 初始化GradScaler用于AMP
    scaler = GradScaler()
    
    # 设置梯度累积的步数
    accumulation_steps = 1

    initial_lambda = lambda_align

    for e in range(start_epoch, start_epoch + local_epochs):
        total_loss = 0
        
        # 梯度累积需要清零一次
        optimizer.zero_grad()

        for batch_idx, (data, target) in enumerate(training_data):
            
            aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
            aug_data = torch.cat([aug_data1, aug_data2], dim=0)
            
            aug_data, target = aug_data.to(device), target.to(device)
            bsz = target.shape[0]
            
            # 使用autocast包裹前向传播和损失计算
            with autocast():
                logits, feature_norm = model(aug_data)
                aug_labels = torch.cat([target, target], dim=0).to(device)            
                cls_loss = cls_loss_fn(logits, aug_labels)
                f1, f2 = torch.split(feature_norm, [bsz, bsz], dim=0)
                features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
                contrastive_loss = contrastive_loss_fn(features, target)
                pro_feat_con_loss = con_proto_feat_loss_fn(feature_norm, model.learnable_proto, aug_labels)
                pro_con_loss = con_proto_loss_fn(model.learnable_proto)
                base_loss = cls_loss + contrastive_loss + pro_con_loss + pro_feat_con_loss
                
                align_loss = 0
                if use_progressive_alignment and initial_protos is not None and fixed_anchors is not None:
                    progress = (e - start_epoch) / local_epochs
                    target_anchor = (1 - progress) * initial_protos + progress * fixed_anchors
                    align_loss = alignment_loss_fn(model.learnable_proto, target_anchor)
                elif use_drcl and fixed_anchors is not None:
                    align_loss = alignment_loss_fn(model.learnable_proto, fixed_anchors)

                # 归一化损失以适应梯度累积
                loss = base_loss
                if use_drcl or use_progressive_alignment:
                    progress = (e - start_epoch) / local_epochs
                    lambda_annealed = initial_lambda * (1 - progress)
                    loss += lambda_annealed * align_loss
                
                loss = loss / accumulation_steps

            # 使用scaler进行反向传播
            scaler.scale(loss).backward()
            
            # 梯度累积
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
    
        train_test_acc = test_acc(model, test_dataloader, device, mode='etf')
        train_set_acc = test_acc(model, training_data, device, mode='etf')

        logger.info(f'Epoch {e} loss: {total_loss}; train accuracy: {train_set_acc}; test accuracy: {train_test_acc}')

        if e % save_freq == 0:
            save_best_local_model(client_model_dir, model, f'epoch_{e}.pth')

    return model