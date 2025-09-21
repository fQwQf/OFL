import torch

class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None, negative_keys=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # 如果提供了来自 memory bank 的负样本，将它们拼接到 contrast_feature 中
        if negative_keys is not None:
            full_contrast_feature = torch.cat([contrast_feature, negative_keys], dim=0)
        else:
            full_contrast_feature = contrast_feature

        # compute logits using the full set of contrastive features
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, full_contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)

        # 如果使用了 memory bank，需要扩展 mask 以匹配 full_contrast_feature 的维度
        # 来自 memory bank 的样本永远不应被视为正样本
        if negative_keys is not None:
            # 创建一个全零的 mask 用于 memory bank 的 key
            mem_bank_mask = torch.zeros(mask.shape[0], negative_keys.shape[0], device=device)
            # 拼接 mask
            mask = torch.cat([mask, mem_bank_mask], dim=1)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss
    
class InfoNCELoss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        Computes the InfoNCE loss.
        
        Args:
            features (torch.Tensor): The feature matrix of shape [2 * batch_size, feature_dim], 
                                     where features[:batch_size] are the representations of 
                                     the first set of augmented images, and features[batch_size:] 
                                     are the representations of the second set.
        
        Returns:
            torch.Tensor: The computed InfoNCE loss.
        """
        # Normalize features to have unit norm
        features = torch.nn.functional.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Get batch size
        batch_size = features.shape[0] // 2
        
        # Construct labels where each sample's positive pair is in the other view
        labels = torch.arange(batch_size, device=features.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)

        # Mask out self-similarities by setting the diagonal elements to -inf
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=features.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        # InfoNCE loss
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss  
    
class Contrastive_proto_feature_loss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(Contrastive_proto_feature_loss, self).__init__()
        self.temperature = temperature
        
    def forward(self, feature, proto, labels):
        # Compute similarity matrix
        similarity_matrix = torch.matmul(feature, proto.T) / self.temperature
        
        # same label prototype should similar with the corresponding feature
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss
    
class Contrastive_proto_loss(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(Contrastive_proto_loss, self).__init__()
        self.temperature = temperature
    
    def forward(self, proto):
        proto_len = proto.shape[0]
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(proto, proto.T) / self.temperature
        
        labels = torch.arange(proto_len, device=proto.device)
        
        mask = torch.eye(proto_len, dtype=torch.bool, device=proto.device)
        # similarity_matrix = similarity_matrix.masked_fill(mask, -float('inf'))
        
        loss = torch.nn.functional.cross_entropy(similarity_matrix, labels)
        
        return loss         