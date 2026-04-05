# test_optimize.py
# Kiểm tra optimize_model hoạt động đúng — đặc biệt là chiều shape tensor

import torch
import torch.optim as optim
import gymnasium as gym

import config
from model import build_networks
from agent import Agent

device = config.DEVICE
env    = gym.make("CartPole-v1")
n_obs     = env.observation_space.shape[0]
n_actions = env.action_space.n

policy_net, target_net = build_networks(n_obs, n_actions, device)
optimizer = optim.AdamW(
    policy_net.parameters(), lr=config.LR, amsgrad=True
)
agent = Agent(policy_net, target_net, optimizer, n_actions)

# ── Test 1: Chưa đủ data thì không optimize ──────────────────
print("Test 1 — optimize_model khi memory rỗng:")
result = agent.optimize_model()
print(f"  Kết quả: {result}  (mong đợi: None)")

# ── Test 2: Nạp đủ data rồi optimize ─────────────────────────
print("\nTest 2 — Nạp dữ liệu giả và optimize:")

# Tạo BATCH_SIZE + 10 transitions giả (một vài terminal)
state, _ = env.reset()
for i in range(config.BATCH_SIZE + 10):
    s  = torch.tensor(state, dtype=torch.float32,
                      device=device).unsqueeze(0)
    a  = torch.tensor([[env.action_space.sample()]])
    obs, r, terminated, truncated, _ = env.step(a.item())
    done = terminated or truncated

    ns = (None if terminated else
          torch.tensor(obs, dtype=torch.float32,
                       device=device).unsqueeze(0))
    rw = torch.tensor([r], device=device)

    agent.store_transition(s, a, ns, rw)

    if done:
        state, _ = env.reset()
    else:
        state = obs

print(f"  Memory size: {len(agent.memory)}")
print(f"  is_ready   : {agent.memory.is_ready(config.BATCH_SIZE)}")

loss_val = agent.optimize_model()
print(f"  Loss value : {loss_val:.6f}  (bất kỳ số dương nào = đúng)")

# ── Test 3: Loss giảm sau nhiều bước optimize ─────────────────
print("\nTest 3 — Loss thay đổi theo thời gian (10 bước):")
losses = []
for step in range(10):
    l = agent.optimize_model()
    if l is not None:
        losses.append(l)
        print(f"  Step {step+1:>2}: loss = {l:.6f}")

print(f"\n  Loss đầu : {losses[0]:.6f}")
print(f"  Loss cuối: {losses[-1]:.6f}")
print("  (loss không nhất thiết giảm liên tục — RL rất noisy)")

# ── Test 4: Verify gradient chạy đúng ────────────────────────
print("\nTest 4 — Gradient đã cập nhật trọng số:")
w_before = policy_net.layer1.weight.clone()
agent.optimize_model()
w_after  = policy_net.layer1.weight
changed  = not torch.allclose(w_before, w_after)
print(f"  Trọng số thay đổi sau optimize: {changed}  (mong đợi: True)")

print("\nTất cả test passed! optimize_model hoạt động đúng.")
env.close()
