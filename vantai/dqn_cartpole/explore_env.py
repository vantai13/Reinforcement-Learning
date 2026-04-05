# explore_env.py
# Mục đích: hiểu CartPole trước khi dạy AI — chạy thủ công để xem dữ liệu thực sự trông như thế nào

import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")

print("=" * 50)
print("THÔNG TIN MÔI TRƯỜNG")
print("=" * 50)

# Observation space: không gian trạng thái
# Box(4,) có nghĩa là 4 số thực liên tục
print(f"Observation space : {env.observation_space}")
print(f"  Shape           : {env.observation_space.shape}")  # (4,)
print(f"  Giới hạn thấp   : {env.observation_space.low}")
print(f"  Giới hạn cao    : {env.observation_space.high}")

# Action space: không gian hành động
# Discrete(2) có nghĩa là 2 hành động rời rạc: 0 và 1
print(f"\nAction space      : {env.action_space}")
print(f"  Số actions      : {env.action_space.n}")
print(f"  Action 0        : đẩy trái")
print(f"  Action 1        : đẩy phải")

print("\n" + "=" * 50)
print("THỬ CHẠY 1 EPISODE VỚI ACTION NGẪU NHIÊN")
print("=" * 50)

# Reset trả về (state, info)
# state là numpy array [x, x_dot, theta, theta_dot]
state, info = env.reset(seed=42)  # seed để kết quả lặp lại được
print(f"\nState ban đầu   : {state}")
print(f"  x (vị trí)    : {state[0]:.4f}  (gần 0 = ở giữa ray)")
print(f"  ẋ (vận tốc)   : {state[1]:.4f}  (gần 0 = đứng yên)")
print(f"  θ (góc)       : {state[2]:.4f}  (gần 0 = thẳng đứng)")
print(f"  θ̇ (v.tốc góc): {state[3]:.4f}  (gần 0 = không xoay)")

total_reward = 0
print(f"\n{'Step':>4} | {'Action':>6} | {'x':>8} | {'ẋ':>8} | {'θ':>8} | {'θ̇':>8} | {'Reward':>6} | Done")
print("-" * 72)

for step in range(100):  # chạy tối đa 20 bước
    action = env.action_space.sample()  # chọn ngẫu nhiên: 0 hoặc 1

    # env.step() trả về 5 giá trị
    # next_state: trạng thái mới
    # reward    : phần thưởng (+1 nếu còn sống)
    # terminated: True nếu cột ngã / xe ra ngoài
    # truncated : True nếu đã đủ 500 bước
    # info      : thông tin phụ (thường bỏ qua)
    next_state, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    done = terminated or truncated

    print(f"{step+1:>4} | {action:>6} | {next_state[0]:>8.4f} | {next_state[1]:>8.4f} | "
          f"{next_state[2]:>8.4f} | {next_state[3]:>8.4f} | {reward:>6.1f} | {done}")

    state = next_state

    if done:
        print(f"\nEpisode kết thúc ở bước {step+1}!")
        break

print(f"\nTổng reward: {total_reward}")
print("\nKết luận: action ngẫu nhiên => cột ngã rất nhanh")
print("Mục tiêu DQN: học cách giữ cột thẳng càng lâu càng tốt (tối đa 500 bước)")

env.close()
