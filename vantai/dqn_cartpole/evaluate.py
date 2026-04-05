# evaluate.py
# Load model đã train và xem nó chơi CartPole

import torch
import gymnasium as gym
import time

import config
from model import DQN


def evaluate(model_path: str = "policy_net.pth",
             n_episodes: int =100,
             render: bool = True):
    """
    Chạy agent đã train, in kết quả từng episode.
    """

    # Load môi trường với render để nhìn thấy
    render_mode = "human" if render else None
    env = gym.make("CartPole-v1", render_mode=render_mode)

    # Tạo mạng và load trọng số đã lưu
    net = DQN(
        n_observations=env.observation_space.shape[0],
        n_actions=env.action_space.n
    ).to(config.DEVICE)

    net.load_state_dict(
        torch.load(model_path, map_location=config.DEVICE)
    )
    net.eval()   # tắt dropout, batchnorm — chế độ inference

    print(f"Model loaded: {model_path}")
    print(f"Chạy {n_episodes} episodes...\n")

    durations = []

    for ep in range(n_episodes):
        state, _ = env.reset()
        state = torch.tensor(
            state, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)

        total_reward = 0

        for t in range(500):
            # Luôn exploit — không explore khi evaluate
            with torch.no_grad():
                action = net(state).max(1).indices.view(1, 1)

            obs, reward, terminated, truncated, _ = \
                env.step(action.item())
            total_reward += reward

            if terminated or truncated:
                print(f"  Episode {ep+1}: {t+1} steps | "
                      f"reward={total_reward:.0f} "
                      f"{'✓ Đạt 500!' if t+1 >= 500 else ''}")
                durations.append(t + 1)
                break

            state = torch.tensor(
                obs, dtype=torch.float32, device=config.DEVICE
            ).unsqueeze(0)

            if render:
                time.sleep(0.02)   # chậm lại để nhìn rõ

    avg = sum(durations) / len(durations)
    print(f"\nTrung bình: {avg:.1f} steps / episode")
    print("Agent tốt nếu avg ≥ 450 steps")
    env.close()


if __name__ == "__main__":
    evaluate()