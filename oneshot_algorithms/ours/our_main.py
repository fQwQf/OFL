from oneshot_algorithms.utils import prepare_checkpoint_dir, visualize_pic, compute_local_model_variance
from dataset_helper import get_supervised_transform, get_client_dataloader, NORMALIZE_DICT
from models_lib import get_train_models

from common_libs import *

from oneshot_algorithms.ours.our_local_training import ours_local_training

import torch.nn.functional as F

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

    g_protos = torch.stack(local_protos)
    g_protos = torch.mean(g_protos, dim=0).detach()
    # g_protos = torch.median(g_protos, dim=0).values.detach()

    g_protos_std= torch.std(g_protos)
    logger.info(f'g_protos_std: {g_protos_std}')

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



def OneshotOurs(trainset, test_loader, client_idx_map, config, device):
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


    public_feature_bank = None
    # Check if the feature bank strategy is enabled in the config
    if config.get('ours_v5_params', {}).get('use_public_feature_bank', False):
        logger.info("Public feature bank strategy is ENABLED.")
        
        # We need the public dataset, which should have been split off earlier.
        # This requires modifying the main test.py script. Let's assume it's passed in.
        # For now, we will add the logic to generate it from a placeholder.
        # The correct way is to modify test.py to call the modified get_fl_dataset.
        
        # We need the public_trainset from the modified get_fl_dataset
        _, _, _, public_trainset = get_fl_dataset(
            config["dataset"]["data_name"],
            config["dataset"]["root_path"],
            config['client']['num_clients'],
            config['distribution']['type'],
            config['distribution']['label_num_per_client'],
            config['distribution']['alpha'],
            public_ratio=config['ours_v5_params']['public_data_ratio']
        )
        
        if public_trainset and len(public_trainset) > 0:
            logger.info("Generating public feature bank from public dataset...")
            
            # Use a fresh model instance for feature extraction to avoid data leakage
            bank_model = get_train_models(
                model_name=config['server']['model_name'],
                num_classes=config['dataset']['num_classes'],
                mode='our'
            ).to(device)
            bank_model.eval()

            public_loader = torch.utils.data.DataLoader(public_trainset, batch_size=config['dataset']['test_batch_size'], shuffle=False)
            
            all_features = []
            with torch.no_grad():
                for data, _ in public_loader:
                    data = data.to(device)
                    features = bank_model.encoder(data).detach()
                    all_features.append(features)
            
            public_feature_bank = torch.cat(all_features, dim=0)
            
            # Normalize features and optionally truncate to a max size
            public_feature_bank = F.normalize(public_feature_bank, dim=1)
            
            bank_size = config['ours_v5_params'].get('feature_bank_size', 4096)
            if len(public_feature_bank) > bank_size:
                perm = torch.randperm(len(public_feature_bank))
                public_feature_bank = public_feature_bank[perm[:bank_size]]

            logger.info(f"Public feature bank created with size: {public_feature_bank.shape}")
        else:
            logger.warning("Public feature bank enabled, but no public data was provided.")
    else:
        logger.info("Public feature bank strategy is DISABLED.")

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
                public_feature_bank=public_feature_bank, 
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
        
        method_name = 'OneShotOurs+Ensemble'
        ensemble_model = WEnsembleFeature(model_list=local_models, weight_list=weights)
        ens_proto_acc = eval_with_proto(copy.deepcopy(ensemble_model), test_loader, device, global_proto)
        logger.info(f"The test accuracy (with prototype) of {method_name}: {ens_proto_acc}")
        method_results[method_name].append(ens_proto_acc)


        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)


