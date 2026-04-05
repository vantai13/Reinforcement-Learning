# check_setup.py
import torch
import gymnasium as gym
import matplotlib
import numpy as np

print("=== Kiểm tra cài đặt ===")
print(f"PyTorch version  : {torch.__version__}")
print(f"Gymnasium version: {gym.__version__}")
print(f"NumPy version    : {np.__version__}")
print(f"Matplotlib ver   : {matplotlib.__version__}")

# Kiểm tra GPU
if torch.cuda.is_available():
    print(f"GPU             : {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    print("GPU             : Apple MPS (Mac)")
else:
    print("GPU             : Không có — dùng CPU (vẫn chạy được)")

# Thử tạo môi trường CartPole
env = gym.make("CartPole-v1")
state, _ = env.reset()
print(f"\nCartPole state shape: {state.shape}")  # Phải ra (4,)
print(f"CartPole action space: {env.action_space.n}")  # Phải ra 2
print("\nCài đặt thành công!")
env.close()