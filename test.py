from dataset_helper import get_fl_dataset

from models_lib.models import get_model

from models_lib import get_train_models
from args import args_parser

from common_libs import *

from oneshot_algorithms import *

config_args = args_parser()
config = load_yaml_config(config_args.cfp)
logger.info(f"config: {config}")

setup_seed(config['seed'])

public_ratio = 0.0
# 根据配置决定是否启用公共池以及公共数据的比例
if config_args.algo == 'OursV5':
    if 'ours_v5_params' not in config:
        config['ours_v5_params'] = {}
    config['ours_v5_params']['use_public_feature_bank'] = True
    public_ratio = config['ours_v5_params'].get('public_data_ratio', 0.1)

# get_fl_dataset 现在返回4个值: 私有训练集, 测试集, 客户端索引图, 公共数据集
# trainset 现在是私有部分, public_set 是公共部分
trainset, testset, client_idx_map, public_set = get_fl_dataset(
    config["dataset"]["data_name"], 
    config["dataset"]["root_path"], 
    config['client']['num_clients'], 
    config['distribution']['type'], 
    config['distribution']['label_num_per_client'], 
    config['distribution']['alpha'],
    public_ratio=public_ratio
)

test_loader = torch.utils.data.DataLoader(testset, batch_size=config['dataset']['test_batch_size'], shuffle=True)

# If you want to use the FedAvg model, uncomment the following lines
# global_model = get_model(
#     model_name=config['server']['model_name'],
#     num_classes=config['dataset']['num_classes'],
#     channels=config['dataset']['channels'],
# )

global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='our'
    )

device = config['device']

if config_args.algo == 'FedAvg':
    OneshotFedAvg(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'Ensemble':
    OneshotEnsemble(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'OTFusion':
    OTFusion(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'FedProto':
    OneshotFedProto(trainset, test_loader, client_idx_map, config, device)
elif config_args.algo == 'FedETF':
    OneshotFedETF(trainset, test_loader, client_idx_map, config, device)
# elif config_args.algo == 'OursV1':
#     FedBCD(trainset, test_loader, client_idx_map, config, global_model, device)
# elif config_args.algo == 'OursV2':
#     FedBCD2(trainset, test_loader, client_idx_map, config, global_model, device)
# elif config_args.algo == 'OursV3':
#     FedBCD3(trainset, test_loader, client_idx_map, config, global_model, device)
elif config_args.algo == 'OursV4':
    OneshotOurs(trainset, test_loader, client_idx_map, config, device, public_set=None)
elif config_args.algo == 'OursV5': 
    OneshotOurs(trainset, test_loader, client_idx_map, config, device, public_set=public_set)


else:
    raise NotImplementedError(f"Algorithm {config_args.algo} is not implemented.")   