from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc, save_best_local_model
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss

from common_libs import *

def ours_local_training(model, training_data, test_dataloader, start_epoch, local_epochs, optim_name, lr, momentum, loss_name, device, num_classes, sample_per_class, aug_transformer, client_model_dir, save_freq=1, use_drcl=False, fixed_anchors=None, lambda_align=1.0, use_progressive_alignment=False, initial_protos=None):
    model.train()
    model.to(device)

    optimizer = init_optimizer(model, optim_name, lr, momentum)
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    contrastive_loss_fn = SupConLoss(temperature=0.07)
    con_proto_feat_loss_fn = Contrastive_proto_feature_loss(temperature=1.0)
    con_proto_loss_fn = Contrastive_proto_loss(temperature=1.0)

    # 如果使用DRCL，定义对齐损失函数
    if use_drcl:
        alignment_loss_fn = torch.nn.MSELoss()

    initial_lambda = lambda_align

    for e in range(start_epoch, start_epoch + local_epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(training_data):
            
            
            aug_data1, aug_data2 = aug_transformer(data), aug_transformer(data)
            aug_data = torch.cat([aug_data1, aug_data2], dim=0)
            
            aug_data, target = aug_data.to(device), target.to(device)
            bsz = target.shape[0]

            optimizer.zero_grad()
            
            logits, feature_norm = model(aug_data)

            # classification loss
            aug_labels = torch.cat([target, target], dim=0).to(device)            
            cls_loss = cls_loss_fn(logits, aug_labels)
            
            # contrastive loss
            f1, f2 = torch.split(feature_norm, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            contrastive_loss = contrastive_loss_fn(features, target)

            # prototype <--> feature contrastive loss
            pro_feat_con_loss = con_proto_feat_loss_fn(feature_norm, model.learnable_proto, aug_labels)
            
            # prototype self constrastive 
            pro_con_loss = con_proto_loss_fn(model.learnable_proto)

            # 计算基础损失，并根据开关决定是否加入对齐损失
            base_loss = cls_loss + contrastive_loss + pro_con_loss + pro_feat_con_loss

            align_loss = 0

            # 选择对齐策略
            if use_progressive_alignment and initial_protos is not None and fixed_anchors is not None:
                # OursV8 逻辑: 渐进式对齐
                progress = (e - start_epoch) / local_epochs
                # 动态计算插值目标
                target_anchor = (1 - progress) * initial_protos + progress * fixed_anchors
                align_loss = alignment_loss_fn(model.learnable_proto, target_anchor)
            elif use_drcl and fixed_anchors is not None:
                # OursV5, V6, V7 逻辑: 对齐到固定目标
                align_loss = alignment_loss_fn(model.learnable_proto, fixed_anchors)

            if use_drcl and fixed_anchors is not None:
                # 线性衰减：从 initial_lambda 降至 0
                progress = (e - start_epoch) / local_epochs
                lambda_annealed = initial_lambda * (1 - progress)

                # 计算可学习原型与固定锚点之间的对齐损失
                loss = base_loss + lambda_annealed * align_loss
            else:
                loss = base_loss


            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            total_loss += loss.item()
    
        train_test_acc = test_acc(copy.deepcopy(model), test_dataloader, device, mode='etf')
        train_set_acc = test_acc(copy.deepcopy(model), training_data, device, mode='etf')

        logger.info(f'Epoch {e} loss: {total_loss}; train accuracy: {train_set_acc}; test accuracy: {train_test_acc}')

        if e % save_freq == 0:
            save_best_local_model(client_model_dir, model, f'epoch_{e}.pth')


    return model



