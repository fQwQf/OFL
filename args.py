import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--cfp', type=str, default='./configs/SVHN_alpha0.1.yaml', help='Name of the experiment configuration path')
    parser.add_argument('--algo', type=str, default='OursV4', help='Name of the algorithm', choices=['FedAvg', 'Ensemble', 'OTFusion', 'FedProto', 'FedETF', 'OursV1', 'OursV2', 'OursV3', 'OursV4', 'OursV5', 'OursV6', 'OursV7', 'OursV8'])

    args = parser.parse_args()

    return args

