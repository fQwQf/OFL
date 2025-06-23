# python test.py --cfp ./configs/CIFAR10.yaml --algo FedAvg
# python test.py --cfp ./configs/CIFAR10.yaml --algo Ensemble
# python test.py --cfp ./configs/CIFAR10.yaml --algo OTFusion
# python test.py --cfp ./configs/CIFAR10.yaml --algo FedProto
# python test.py --cfp ./configs/CIFAR10.yaml --algo FedETF
# python test.py --cfp ./configs/CIFAR10.yaml --algo OursV4

# nohup python test.py --cfp ./configs/CIFAR10.yaml --algo OursV4 >/dev/null 2>&1 &

# nohup python test.py --cfp ./configs/Tiny_alpha0.3.yaml --algo OursV4 >/dev/null 2>&1 &
# nohup python test.py --cfp ./configs/Tiny_alpha0.1.yaml --algo OursV4 >/dev/null 2>&1 &
# nohup python test.py --cfp ./configs/Tiny_alpha0.05.yaml --algo OursV4 >/dev/null 2>&1 &

# nohup python test.py --cfp ./configs/SVHN_alpha0.1.yaml --algo OursV4 >/dev/null 2>&1 &
# nohup python test.py --cfp ./configs/SVHN_alpha0.05.yaml --algo OursV4 >/dev/null 2>&1 &

# python test.py --cfp ./configs/observation.yaml --algo FedAvg

# python test.py --cfp ./configs/CIFAR10_client10.yaml --algo OursV4

# python intra_model_inconsistency.py --cfp ./configs/observation.yaml

# python intra_visual.py --cfp ./configs/Tiny_alpha0.5.yaml


# nohup python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo OursV4 >/dev/null 2>&1 &


python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo OursV4

# 运行FedAvg
python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo FedAvg

# 运行Ensemble
python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo Ensemble

# 运行OTFusion
python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo OTFusion

# 运行FedProto
python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo FedProto

# 运行FedETF
python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo FedETF



# 在CIFAR-100上运行
python test.py --cfp ./configs/CIFAR100_alpha0.1.yaml --algo OursV4

# 在SVHN上运行
python test.py --cfp ./configs/SVHN_alpha0.1.yaml --algo OursV4

# 在Tiny-ImageNet上运行
python test.py --cfp ./configs/Tiny_alpha0.1.yaml --algo OursV4



# CIFAR-10，不同alpha值
python test.py --cfp ./configs/CIFAR10_alpha0.5.yaml --algo OursV4
python test.py --cfp ./configs/CIFAR10_alpha0.3.yaml --algo OursV4
python test.py --cfp ./configs/CIFAR10_alpha0.1.yaml --algo OursV4
python test.py --cfp ./configs/CIFAR10_alpha0.05.yaml --algo OursV4