# CIFAR-10 (Original)
# python main.py --niid --partition dir --rg 1 --num_clients 100 --alpha 0.1 --clean_reg --dataset cifar10 --num_classes 10 --modelseed 1 --pretrained

# CIFAR-100 with Continual Learning Setup
# 10 clients, 4 tasks per client, 20 classes per task
python main.py --continual --rg 1 --clean_reg --dataset cifar100 --num_classes 100 --modelseed 1 --pretrained

# CIFAR-100 (Original for comparison)
# python main.py --niid --partition dir --rg 1 --num_clients 100 --alpha 0.1 --clean_reg --dataset cifar100 --num_classes 100 --modelseed 1 --pretrained

# Tiny-ImageNet (Original)
# python main.py --niid --partition dir --rg 1 --num_clients 100 --alpha 0.1 --clean_reg --dataset tinyimagenet --num_classes 200 --modelseed 1 --pretrained