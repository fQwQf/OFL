from oneshot_algorithms.utils import init_optimizer, init_loss_fn, test_acc, save_best_local_model
from oneshot_algorithms.ours.unsupervised_loss import SupConLoss, Contrastive_proto_feature_loss, Contrastive_proto_loss

from common_libs import *

def ours_local_training(model, training_data, test_dataloader, start_epoch, local_epochs, optim_name, lr, momentum, loss_name, device, num_classes, sample_per_class, aug_transformer, client_model_dir, save_freq=1):
    model.train()
    model.to(device)

    optimizer = init_optimizer(model, optim_name, lr, momentum)
    cls_loss_fn = torch.nn.CrossEntropyLoss()
    contrastive_loss_fn = SupConLoss(temperature=0.07)
    con_proto_feat_loss_fn = Contrastive_proto_feature_loss(temperature=1.0)
    con_proto_loss_fn = Contrastive_proto_loss(temperature=1.0)

    # 初始化 Memory Bank
    memory_bank = None
    ptr = 0
    if use_memory_bank:
        # 动态获取特征维度
        try:
            # 假设模型有一个名为 encoder 的部分，并且最后是一个线性层
            feature_dim = model.encoder.fc.in_features
        except AttributeError:
            # 如果结构不同，需要提供一个备用方法
            # 我们可以通过一次前向传播来获取
            dummy_input = torch.randn(2, 3, 32, 32).to(device) # 假设输入是 CIFAR10 尺寸
            with torch.no_grad():
                _, dummy_features = model(dummy_input)
            feature_dim = dummy_features.shape[1]
            logger.info(f"Dynamically determined feature dimension: {feature_dim}")

        # 初始化一个足够大的 tensor 来存储特征
        memory_bank = torch.randn(memory_bank_size, feature_dim, device=device).renorm(2, 0, 1) # L2 归一化
        logger.info(f"Client using Memory Bank with size {memory_bank_size} and feature dim {feature_dim}")

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

            if use_memory_bank:
                # 从 memory_bank 中获取负样本
                # 注意：这里直接使用整个 memory_bank 作为负样本集
                # 也可以随机采样一部分，但通常使用全部效果更好
                negative_keys = memory_bank.clone().detach()
                contrastive_loss = contrastive_loss_fn(features, target, negative_keys=negative_keys)

                # 更新 memory_bank (enqueue/dequeue)
                # 使用 f1 特征来更新 bank
                current_features = f1.detach()
                num_keys_to_update = current_features.shape[0]
                
                if ptr + num_keys_to_update >= memory_bank_size:
                    # 如果超出边界，分两部分填充
                    part1_len = memory_bank_size - ptr
                    memory_bank[ptr:, :] = current_features[:part1_len]
                    memory_bank[:num_keys_to_update - part1_len, :] = current_features[part1_len:]
                    ptr = num_keys_to_update - part1_len
                else:
                    memory_bank[ptr:ptr + num_keys_to_update, :] = current_features
                    ptr += num_keys_to_update
            else:
                # 原始算法
                contrastive_loss = contrastive_loss_fn(features, target)

            # prototype <--> feature contrastive loss
            pro_feat_con_loss = con_proto_feat_loss_fn(feature_norm, model.learnable_proto, aug_labels)
            
            # prototype self constrastive 
            pro_con_loss = con_proto_loss_fn(model.learnable_proto)

            loss = cls_loss + contrastive_loss + pro_con_loss + pro_feat_con_loss

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



