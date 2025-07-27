#!/bin/bash

# EMNIST-letters with Continual Learning Setup
# 8 clients, 6 tasks per client, 4 classes per task
python main.py --continual --rg 1 --clean_reg --dataset emnist --modelseed 1 --pretrained

# Additional configurations for EMNIST-letters (if needed)
# Without continual learning (for comparison)
# python main.py --niid --partition dir --rg 1 --num_clients 8 --alpha 0.1 --clean_reg --dataset emnist --num_classes 26 --modelseed 1 --pretrained

# CIFAR-100 with Continual Learning Setup (original)
# python main.py --continual --rg 1 --clean_reg --dataset cifar100 --num_classes 100 --modelseed 1 --pretrained

# CIFAR-10 (Original)
# python main.py --niid --partition dir --rg 1 --num_clients 100 --alpha 0.1 --clean_reg --dataset cifar10 --num_classes 10 --modelseed 1 --pretrained

# Tiny-ImageNet (Original)
# python main.py --niid --partition dir --rg 1 --num_clients 100 --alpha 0.1 --clean_reg --dataset tinyimagenet --num_classes 200 --modelseed 1 --pretrained