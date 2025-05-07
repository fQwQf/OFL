from oneshot_algorithms.utils import prepare_checkpoint_dir, local_training, test_acc, compute_local_model_variance, visualize_pic
from dataset_helper import get_client_dataloader, get_supervised_transform


from common_libs import *
from models_lib import get_train_models

from oneshot_algorithms.otfusion.aggregation import get_otfusion_model

def OTFusion(trainset, test_loader, client_idx_map, config, device):
    
    logger.info('OTFusion')
    # get the global model
    global_model = get_train_models(
        model_name=config['server']['model_name'],
        num_classes=config['dataset']['num_classes'],
        mode='ot'
    )

    global_model.to(device)
    global_model.train()


    method_results = defaultdict(list)
    save_path, _ = prepare_checkpoint_dir(config)
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


    # visualization
    supervised_transformer = get_supervised_transform(config['dataset']['data_name'])
    vis_loader = torch.utils.data.DataLoader(copy.deepcopy(test_loader.dataset), config['visualization']['vis_size'], shuffle=False)
    vis_iter = iter(vis_loader)
    vis_data, vis_label = next(vis_iter)
    vis_data = supervised_transformer(vis_data)
    vis_folder = config['visualization']['save_path'] +"/otfusion/"

    for cr in trange(config['server']['num_rounds']):
        logger.info(f"Round {cr} starts--------|")
        for c in range(config['client']['num_clients']):
            logger.info(f"Client {c} Starts Local Trainning--------|")
            client_dataloader = get_client_dataloader(client_idx_map[c], trainset, config['dataset']['train_batch_size'])
            # local training
            local_model_c = local_training(
                model=copy.deepcopy(local_models[c]),
                training_data=client_dataloader,
                test_dataloader=test_loader,
                start_epoch=0,
                local_epochs=config['server']['local_epochs'],
                optim_name=config['server']['optimizer'],
                lr=config['server']['lr'],
                momentum=config['server']['momentum'],
                loss_name=config['server']['loss_name'],
                history_acc_list=[],
                best_acc=-1,
                best_epoch=-1,
                client_model_dir=None,
                device=device,
                num_classes=config['dataset']['num_classes'],
                save_freq=config['checkpoint']['save_freq']
            )
            
            local_models[c] = local_model_c
            logger.info(f"Client {c} Finish Local Training--------|")

            visualize_pic(local_model_c, vis_data, target_layers=[local_model_c.layer4], dataset_name=config['dataset']['data_name'], save_file_name=f'{save_path}/{vis_folder}/local_model_{c}.png', device=device)
                
            logger.info(f"Visualization of Global model at {save_path}/{vis_folder}/local_model_{c}.png")
        logger.info(f"Round {cr} Finish--------|")
        model_var_m, model_var_s = compute_local_model_variance(local_models)
        logger.info(f"Model variance: mean: {model_var_m}, sum: {model_var_s}")


        # OTFusion
        method_name = 'OTFusion'
        otfusion_model = get_otfusion_model(weight=weights, model_name=config['server']['model_name'], net_glob=global_model, models=local_models, test_loader=test_loader, device=device)

        acc = test_acc(otfusion_model, test_loader, device)
        logger.info(f"The test accuracy of {method_name}: {acc}")

        save_yaml_config(save_path + "/baselines_" + method_name +"_" + config['checkpoint']['result_file'], method_results)

