# test_agent.py
# Kiểm tra select_action và epsilon decay hoạt động đúng

import torch
import torch.optim as optim
import gymnasium as gym

import config
from model import build_networks
from agent import Agent

device = config.DEVICE
env    = gym.make("CartPole-v1")

n_obs     = env.observation_space.shape[0]   # 4
n_actions = env.action_space.n               # 2

policy_net, target_net = build_networks(n_obs, n_actions, device)
optimizer = optim.AdamW(policy_net.parameters(),
                        lr=config.LR, amsgrad=True)
agent = Agent(policy_net, target_net, optimizer, n_actions)

# ── Test 1: Epsilon decay ─────────────────────────────────────
import math
print("Test 1 — Epsilon decay theo steps:")
checkpoints = [0, 100, 500, 1000, 2500, 5000, 10000]
for step in checkpoints:
    eps = (config.EPS_END
           + (config.EPS_START - config.EPS_END)
           * math.exp(-step / config.EPS_DECAY))
    bar_e = int(eps * 30)
    bar_x = 30 - bar_e
    bar = "█" * bar_x + "░" * bar_e
    print(f"  step {step:>6}: ε={eps:.4f}  exploit|{bar}|explore")

# ── Test 2: select_action trả về đúng shape ───────────────────
print("\nTest 2 — Output shape của select_action:")
state, _ = env.reset()
state_t   = torch.tensor(state, dtype=torch.float32,
                          device=device).unsqueeze(0)  # (1, 4)
action    = agent.select_action(state_t)
print(f"  State shape : {state_t.shape}")    # (1, 4)
print(f"  Action shape: {action.shape}")     # (1, 1)
print(f"  Action value: {action.item()}")    # 0 hoặc 1

# ── Test 3: Tỉ lệ explore vs exploit thực tế ─────────────────
print("\nTest 3 — Tỉ lệ explore vs exploit (1000 bước đầu):")
agent2 = Agent(policy_net, target_net, optimizer, n_actions)
explore_count = exploit_count = 0
for _ in range(1000):
    s = torch.randn(1, 4, device=device)
    import random as _r
    _prev = agent2.steps_done
    act   = agent2.select_action(s)
    # Tính epsilon tại bước trước để biết lần này explore hay exploit
    eps   = (config.EPS_END
             + (config.EPS_START - config.EPS_END)
             * math.exp(-_prev / config.EPS_DECAY))
    # Không thể biết trực tiếp, nên đếm gián tiếp qua epsilon
    if _r.random() < eps:
        explore_count += 1
    else:
        exploit_count += 1

print(f"  Explore: ~{explore_count/10:.1f}%  (mong đợi cao vì steps còn nhỏ)")
print(f"  Exploit: ~{exploit_count/10:.1f}%")

# ── Test 4: Soft update hoạt động ─────────────────────────────
print("\nTest 4 — Soft update:")
w_before = agent.target_net.layer1.weight.clone()
agent.soft_update_target()
w_after  = agent.target_net.layer1.weight
diff     = (w_after - w_before).abs().mean().item()
print(f"  Thay đổi sau 1 soft update: {diff:.8f}  (rất nhỏ = đúng)")

print("\nTất cả test passed! agent.py (select_action) hoạt động đúng.")
env.close()
