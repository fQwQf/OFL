from oneshot_algorithms.utils import prepare_checkpoint_dir, visualize_pic, compute_local_model_variance
from dataset_helper import get_supervised_transform, get_client_dataloader, NORMALIZE_DICT
from models_lib import get_train_models

from common_libs import *

import torch.nn.functional as F

from oneshot_algorithms.ours.our_local_training import ours_local_training

def get_supcon_transform(dataset_name):
    if dataset_name == 'CIFAR10' or dataset_name == 'CIFAR100' or dataset_name == 'SVHN':
        return torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(size=32, scale=(0.2, 1.), antialias=False),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomApply([
                torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            torchvision.transforms.RandomGrayscale(p=0.2),
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(**NORMALIZE_DICT[dataset_name])
        ])                
    else:    
        return torchvision.transforms.Compose([
        torchvision.transforms.RandomResizedCrop(size=64, scale=(0.2, 1.), antialias=False),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomApply([
            torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        torchvision.transforms.RandomGrayscale(p=0.2),
        # torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(**NORMALIZE_DICT[dataset_name])
        ])


def generate_etf_anchors(num_classes, feature_dim, device):
    """
    Generate Equiangular Tight Frame (ETF) anchors based on Neural Collapse theory.
    This creates a set of maximally separated and geometrically optimal prototype targets.
    """
    # 确保特征维度至少不小于类别数，这是构建ETF的常见要求
    if feature_dim < num_classes:
        # 如果维度不够，我们可以退回到正交基，这是一个很好的次优选择
        logger.warning(f"Feature dim ({feature_dim}) is less than num_classes ({num_classes}). Falling back to orthogonal anchors.")
        H = torch.randn(feature_dim, num_classes)
        Q, _ = torch.qr(H)
        return Q.T.to(device)

    # 1. 构造ETF的格拉姆矩阵 M = I - (1/C) * J
    I = torch.eye(num_classes)
    J = torch.ones(num_classes, num_classes)
    M = I - (1 / num_classes) * J

    # 2. 通过Cholesky分解找到 M_sqrt
    # M 是半正定的，可能需要添加一个小的epsilon以保证数值稳定性
    try:
        L = torch.linalg.cholesky(M + 1e-6 * I)
    except torch.linalg.LinAlgError:
        # 如果Cholesky分解失败，使用特征值分解
        eigvals, eigvecs = torch.linalg.eigh(M)
        eigvals[eigvals < 0] = 0 # 消除数值误差导致的负特征值
        L = eigvecs @ torch.diag(torch.sqrt(eigvals))

    # 3. 生成一个随机正交矩阵的“基底”
    H_ortho = torch.randn(feature_dim, num_classes)
    Q, _ = torch.linalg.qr(H_ortho) # Q的列是正交的

    # 4. 将基底与 M_sqrt 相乘，生成最终的ETF矩阵
    # W 的列向量构成了ETF
    W = Q @ L.T
    
    # 我们需要的是 (num_classes, feature_dim) 的原型，所以返回转置
    etf_anchors = W.T.to(device)
    
    # 最终归一化，确保所有锚点都是单位向量
    etf_anchors = torch.nn.functional.normalize(etf_anchors, dim=1)
    
    return etf_anchors


def agg_protos(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos

def collect_protos(model, trainloader, device):
    model.eval()
    protos = defaultdict(list)
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(trainloader):
            if type(data) == type([]):
                data[0] = data[0].to(device)
            else:
                data = data.to(device)
            target = target.to(device) 

            rep = model.encoder(data)

            for i, yy in enumerate(target):
                y_c = yy.item()
                protos[y_c].append(rep[i, :].detach().data)

    local_protos = agg_protos(protos)

    return local_protos  

def generate_sample_per_class(num_classes, local_data, data_num):
    sample_per_class = torch.tensor([0 for _ in range(num_classes)])

    for idx, (data, target) in enumerate(local_data):
        sample_per_class += torch.tensor([sum(target==i) for i in range(num_classes)])

    sample_per_class = sample_per_class / data_num

    return sample_per_class

def aggregate_local_protos(local_protos):

    local_protos = torch.stack(local_protos, dim=0)

    g_protos = torch.mean(local_protos, dim=0).detach()

    inter_client_proto_std = torch.std(local_protos, dim=0).mean().item()

    g_protos_std = torch.std(g_protos).item()

    logger.info(f'g_protos_std (global internal): {g_protos_std:.6f}')
    logger.info(f'inter_client_proto_std (cross-client): {inter_client_proto_std:.6f}')

    return g_protos



class WEnsembleFeatureNoise(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(WEnsembleFeatureNoise, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list
            
    
    def forward(self, x):
        noise = torch.randn_like(x)
        
        dis_noise_list = []
        feature_total = []
        for model, weight in zip(self.models, self.weight_list):
            feature = model.encoder(x) # batchsize, feature_dim
            noise_feature = model.encoder(noise) 
            dis_sim = 1 - torch.nn.functional.cosine_similarity(feature, noise_feature, dim=1) # batchsize,
            dis_noise_list.append(dis_sim) # model_num, batchsize
            feature_total.append(feature) # model_num, batchsize, feature_dim
        
        dis_noise_list = torch.stack(dis_noise_list).mean(dim=0).unsqueeze(-1) # model_num, batchsize, 1
        
        feature_total = torch.stack(feature_total) # model_num, batchsize, feature_dim
        
        feature_total = dis_noise_list * feature_total # model_num, batchsize, feature_dim
        feature_total = feature_total.sum(dim=0) # batchsize, feature_dim
        
        return feature_total


class WEnsembleFeature(torch.nn.Module):
    def __init__(self, model_list, weight_list=None):
        super(WEnsembleFeature, self).__init__()
        self.models = model_list
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list
            
    def forward(self, x):
        feature_total = 0
        for model, weight in zip(self.models, self.weight_list):
            feature = weight * model.encoder(x)
            feature_total += feature
        return feature_total   
    

class TrueSimpleEnsembleServer(torch.nn.Module):

    def __init__(self, model_list, weight_list=None):
        super(TrueSimpleEnsembleServer, self).__init__()
        # Ensure models are in eval mode for inference
        self.models = [model.eval() for model in model_list]
        if weight_list is None:
            self.weight_list = [1.0 / len(model_list) for _ in range(len(model_list))]
        else:
            self.weight_list = weight_list

    def forward(self, x):
        all_probs = []
        with torch.no_grad():
            for model, weight in zip(self.models, self.weight_list):
                # Get logits from the full model (encoder + classifier/proto)
                logits, _ = model(x)
                # Convert to probabilities and apply weight
                probs = F.softmax(logits, dim=1) * weight
                all_probs.append(probs)
        
        # Sum the weighted probabilities
        final_probs = torch.sum(torch.stack(all_probs), dim=0)
        
        # Return the final probability distribution
        return final_probs

    
def eval_with_proto(model, test_loader, device, proto):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            feature = model(data)
            feature = torch.nn.functional.normalize(feature, dim=1)

            logits = torch.matmul(feature, proto.t())
            
            _, predicted = torch.max(logits.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = correct / total
    return acc        

def eval_output_ensemble(model, test_loader, device):
    """
    Evaluates an ensemble model that directly outputs class probabilities.
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # Get final probabilities directly from the model
            final_probs = model(data)
            
            # Get predictions
            _, predicted = torch.max(final_probs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = correct / total
    return acc


def OneshotOurs(trainset, test_loader, client_idx_map, config, device, server_strategy='simple_feature'):
    logger.info('OneshotOurs')
    # get the global model
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    )

    global_model.to(device)
    global_model.train()

    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    # save the config to file
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    # all clients perform local training phrase, the global model is used as the initial model
    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    # get the weight of each client
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    # weight
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])

    # visualization

    supervised_transformer = get_supervised_transform(config['dataset']['data_name'])
    vis_loader = torch.utils.data.DataLoader(copy.deepcopy(test_loader.dataset), config['visualization']['vis_size'], shuffle=False)
    vis_iter = iter(vis_loader)
    vis_data, vis_label = next(vis_iter)
    vis_data = supervised_transformer(vis_data)
    vis_folder = config['visualization']['save_path'] +"/ours/"

    noise_samples = torch.randn_like(vis_data)

    total_rounds = config['server']['num_rounds']


    # sample_per_class
    clients_sample_per_class = []

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
        local_protos = []
        
        for c in range(config['client']['num_clients']):

            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))
                logger.info('generating sample per sample')


            # local training
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
            )
            
            local_models[c] = local_model_c
            
            logger.info(f"Client {c} Finish Local Training--------|")

            # collecting the local prototypes
            # local_proto_c = collect_protos(copy.deepcopy(local_model_c), client_dataloader, device)
            # local_proto_c = local_model_c.get_proto(clients_sample_per_class[c]).detach()
            local_proto_c = local_model_c.get_proto().detach()


            local_protos.append(local_proto_c)
            logger.info(f"Client {c} Collecting Local Prototypes--------|")

            # visualize_pic(local_model_c.encoder, vis_data, target_layers=[local_model_c.encoder.layer4], dataset_name=config['dataset']['data_name'], save_file_name=f'{save_path}/{vis_folder}/local_model_{c}.png', device=device)
                
            # logger.info(f"Visualization of Global model at {save_path}/{vis_folder}/local_model_{c}.png")

            # evaluate the distance between the noise feature and protos

            

        logger.info(f"Round {cr} Finish--------|")
        # some statistics of the current local models
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")


        global_proto = aggregate_local_protos(local_protos)
        
        if server_strategy == 'true_simple_output':
            method_name = 'OursV4+TrueSimpleOutputServer'
            ensemble_model = TrueSimpleEnsembleServer(model_list=local_models, weight_list=weights)
            logger.info("V4 Training | Using TRULY SIMPLE Output-level Server.")
            ens_acc = eval_output_ensemble(ensemble_model, test_loader, device)

        elif server_strategy == 'simple_feature':
            method_name = 'OursV4+SimpleFeatureServer'
            ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V4 Training | Using Simple Feature-level Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)

        elif server_strategy == 'advanced_iffi':
            method_name = 'OursV4+AdvancedIFFIServer'
            ensemble_model = WEnsembleFeatureNoise(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V4 Training | Using Advanced IFFI Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        else:
            raise ValueError(f"Unknown server_strategy: {server_strategy}")

        logger.info(f"The test accuracy of {method_name}: {ens_acc}")


        method_results[method_name].append(ens_acc)


        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


# 新增 OneshotOursV5 (DRCL 版本)
def OneshotOursV5(trainset, test_loader, client_idx_map, config, device, use_simple_server=True):
    logger.info('OneshotOursV5 with Decoupled Representation and Classifier Learning (DRCL)')
    # get the global model
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    )

    global_model.to(device)
    global_model.train()

    # --- 新增逻辑：创建固定的全局原型锚点 ---
    # 获取原型维度
    proto_shape = global_model.learnable_proto.shape
    # 随机初始化，设置 requires_grad=False 使其不可学习
    fixed_anchors = torch.randn(proto_shape, device=device, requires_grad=False)
    # 归一化以保证稳定性
    fixed_anchors = torch.nn.functional.normalize(fixed_anchors, dim=1)
    logger.info(f"Initialized fixed anchors with shape: {fixed_anchors.shape}")
    # ----------------------------------------

    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])

    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))
                logger.info('generating sample per sample')

            # --- 修改本地训练调用 ---
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                # 传入新参数以启用DRCL
                use_drcl=True,
                fixed_anchors=fixed_anchors,
                lambda_align=config.get('lambda_align', 1.0) # 从config获取超参，默认1.0
            )
            # --------------------------
            
            local_models[c] = local_model_c
            logger.info(f"Client {c} Finish Local Training--------|")

            local_proto_c = local_model_c.get_proto().detach()
            local_protos.append(local_proto_c)
            logger.info(f"Client {c} Collecting Local Prototypes--------|")

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        global_proto = aggregate_local_protos(local_protos)
        
        if use_simple_server:
            method_name = 'OneShotOursV5+SimpleServer'
            ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
            logger.info("V5 Training | Using SIMPLE server aggregation.")
        else:
            method_name = 'OneShotOursV5+AdvancedServer'
            ensemble_model = WEnsembleFeatureNoise(model_list=local_models, weight_list=weights)
            logger.info("V5 Training | Using ADVANCED IFFI server aggregation.")
            
        ens_proto_acc = eval_with_proto(copy.deepcopy(ensemble_model), test_loader, device, global_proto)
        logger.info(f"The test accuracy (with prototype) of {method_name}: {ens_proto_acc}")
        method_results[method_name].append(ens_proto_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

# OneshotOursV6 (Lambda Annealing)
def OneshotOursV6(trainset, test_loader, client_idx_map, config, device, use_simple_server=True):
    logger.info('OneshotOursV6 with DRCL and Lambda Annealing')
    # get the global model
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    )

    global_model.to(device)
    global_model.train()
    
    proto_shape = global_model.learnable_proto.shape
    fixed_anchors = torch.randn(proto_shape, device=device, requires_grad=False)
    fixed_anchors = torch.nn.functional.normalize(fixed_anchors, dim=1)
    logger.info(f"Initialized fixed anchors with shape: {fixed_anchors.shape}")

    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])

    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))
                logger.info('generating sample per sample')

            # 调用的是同一个local_training函数，但其内部逻辑会因epoch变化而不同
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                use_drcl=True,
                fixed_anchors=fixed_anchors,
                # lambda_align现在代表初始值
                lambda_align=config.get('lambda_align_initial', 5.0)
            )
            
            local_models[c] = local_model_c
            logger.info(f"Client {c} Finish Local Training--------|")

            local_proto_c = local_model_c.get_proto().detach()
            local_protos.append(local_proto_c)
            logger.info(f"Client {c} Collecting Local Prototypes--------|")

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        global_proto = aggregate_local_protos(local_protos)
        
        if use_simple_server:
            method_name = 'OneShotOursV6+SimpleServer'
            ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
            logger.info("V6 Training | Using SIMPLE server aggregation.")
        else:
            method_name = 'OneShotOursV6+AdvancedServer'
            ensemble_model = WEnsembleFeatureNoise(model_list=local_models, weight_list=weights)
            logger.info("V6 Training | Using ADVANCED IFFI server aggregation.")
            
        ens_proto_acc = eval_with_proto(copy.deepcopy(ensemble_model), test_loader, device, global_proto)
        logger.info(f"The test accuracy (with prototype) of {method_name}: {ens_proto_acc}")
        method_results[method_name].append(ens_proto_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

# OneshotOursV7 (ETF Anchors + Lambda Annealing)
def OneshotOursV7(trainset, test_loader, client_idx_map, config, device, server_strategy='simple_feature'):
    logger.info('OneshotOursV7 with DRCL (ETF Anchors) and Lambda Annealing')
    
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    )
    global_model.to(device)
    global_model.train()

    # --- 核心修改：使用ETF锚点替换随机锚点 ---
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    fixed_anchors = generate_etf_anchors(num_classes, feature_dim, device)
    logger.info(f"Initialized ETF fixed anchors with shape: {fixed_anchors.shape}")

    method_results = defaultdict(list)
    save_path, local_model_dir = prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    if config['server']['aggregated_by_datasize']:
        weights = [i/sum(local_data_size) for i in local_data_size]
    else:
        weights = [1/config['client']['num_clients'] for _ in range(config['client']['num_clients'])]        
    
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])

    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            # 调用与V6完全相同的本地训练函数，但传入的是高质量的ETF锚点
            local_model_c = ours_local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                use_drcl=True,
                fixed_anchors=fixed_anchors,
                lambda_align=config.get('lambda_align_initial', 5.0)
            )
            
            local_models[c] = local_model_c
            logger.info(f"Client {c} Finish Local Training--------|")

            local_proto_c = local_model_c.get_proto().detach()
            local_protos.append(local_proto_c)
            logger.info(f"Client {c} Collecting Local Prototypes--------|")

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        global_proto = aggregate_local_protos(local_protos)
        
        if server_strategy == 'true_simple_output':
            method_name = 'OursV7+TrueSimpleOutputServer'
            ensemble_model = TrueSimpleEnsembleServer(model_list=local_models, weight_list=weights)
            logger.info("V7 Training | Using TRULY SIMPLE Output-level Server.")
            ens_acc = eval_output_ensemble(ensemble_model, test_loader, device)

        elif server_strategy == 'simple_feature':
            method_name = 'OursV7+SimpleFeatureServer'
            ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V7 Training | Using Simple Feature-level Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)

        elif server_strategy == 'advanced_iffi':
            method_name = 'OursV7+AdvancedIFFIServer'
            ensemble_model = WEnsembleFeatureNoise(model_list=local_models, weight_list=weights)
            global_proto = aggregate_local_protos(local_protos)
            logger.info("V7 Training | Using Advanced IFFI Server.")
            ens_acc = eval_with_proto(ensemble_model, test_loader, device, global_proto)
        else:
            raise ValueError(f"Unknown server_strategy: {server_strategy}")

        logger.info(f"The test accuracy of {method_name}: {ens_acc}")


        method_results[method_name].append(ens_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

def OneshotOursV8(trainset, test_loader, client_idx_map, config, device):
    logger.info('OneshotOursV8 - Final Version: Consensus-Start Progressive Alignment')
    
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    )
    global_model.to(device)
    global_model.train()
    
    feature_dim = global_model.learnable_proto.shape[1]
    num_classes = config['dataset']['num_classes']
    fixed_anchors_etf = generate_etf_anchors(num_classes, feature_dim, device)
    logger.info(f"Initialized FINAL ETF anchors with shape: {fixed_anchors_etf.shape}")

    method_results, save_path, local_model_dir = defaultdict(list), *prepare_checkpoint_dir(config)
    if not os.path.exists(save_path + "/config.yaml"):
        save_yaml_config(save_path + "/config.yaml", config) 

    local_models = [copy.deepcopy(global_model) for _ in range(config['client']['num_clients'])]
    
    # 计算一个共同的“共识起点”
    initial_local_protos = [model.get_proto().detach().clone() for model in local_models]
    # 计算所有初始原型的平均值
    consensus_start_protos = torch.stack(initial_local_protos).mean(dim=0)
    logger.info("Calculated a shared CONSENSUS start point for all clients.")

    local_data_size = [len(client_idx_map[c]) for c in range(config['client']['num_clients'])]
    weights = [i/sum(local_data_size) for i in local_data_size] if config['server']['aggregated_by_datasize'] else [1/config['client']['num_clients']] * config['client']['num_clients']
    
    aug_transformer = get_supcon_transform(config['dataset']['data_name'])
    clients_sample_per_class = []

    total_rounds = config['server']['num_rounds']

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        local_protos = []
        
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])

            if cr == 0:
                clients_sample_per_class.append(generate_sample_per_class(config['dataset']['num_classes'], client_dataloader, len(client_idx_map[c])))

            local_model_c = ours_local_training(
                model=local_models[c],
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=cr * config['server']['local_epochs'],
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                device=device,
                num_classes=config['dataset']['num_classes'],
                sample_per_class=clients_sample_per_class[c],
                aug_transformer=aug_transformer,
                client_model_dir=local_model_dir + f"/client_{c}",
                total_rounds=total_rounds,
                save_freq=config['checkpoint']['save_freq'],
                use_drcl=True,
                use_progressive_alignment=True,
                initial_protos=consensus_start_protos,
                fixed_anchors=fixed_anchors_etf,
                lambda_align=config.get('lambda_align_initial', 5.0)
            )
            
            local_models[c] = local_model_c
            logger.info(f"Client {c} Finish Local Training--------|")

            local_protos.append(local_model_c.get_proto().detach())

        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")

        global_proto = aggregate_local_protos(local_protos)
        
        method_name = 'OneShotOursV8+Ensemble'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        ens_proto_acc = eval_with_proto(copy.deepcopy(ensemble_model), test_loader, device, global_proto)
        logger.info(f"The test accuracy (with prototype) of {method_name}: {ens_proto_acc}")
        method_results[method_name].append(ens_proto_acc)

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)