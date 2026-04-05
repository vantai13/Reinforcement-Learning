# main.py
# Entry point — chạy file này để bắt đầu training

import torch
import matplotlib.pyplot as plt

from train import train
import config


def main():
    print("=" * 60)
    print("DQN CartPole — bắt đầu training")
    print("=" * 60)

    # Train và nhận lại agent đã học
    agent = train()

    # ── Lưu model sau khi train xong ──────────────────────────
    # Tại sao save state_dict thay vì save cả model?
    # state_dict chỉ lưu trọng số (nhỏ, portable)
    # Load lại chỉ cần biết kiến trúc mạng — linh hoạt hơn nhiều
    save_path = "policy_net.pth"
    torch.save(agent.policy_net.state_dict(), save_path)
    print(f"\nModel đã lưu tại: {save_path}")
    print("Để load lại:\n"
          "  from model import DQN\n"
          f"  net = DQN(4, 2).to(config.DEVICE)\n"
          f"  net.load_state_dict(torch.load('{save_path}'))")

    # Giữ đồ thị mở
    plt.ioff()
    
    plt.show()


if __name__ == "__main__":
    main()