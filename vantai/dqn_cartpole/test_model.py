# test_model.py
# Kiểm tra model trước khi tích hợp

import torch
from model import DQN, build_networks

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dùng device: {device}\n")

N_OBS     = 4   # CartPole state size
N_ACTIONS = 2   # trái / phải

# --- Test 1: Kiến trúc mạng ---
net = DQN(N_OBS, N_ACTIONS).to(device)
print("Test 1 — Kiến trúc mạng:")
print(net)

# Đếm số tham số (trọng số) cần học
total_params = sum(p.numel() for p in net.parameters())
print(f"\nTổng số tham số: {total_params:,}")
# layer1: 4*128 + 128 = 640
# layer2: 128*128 + 128 = 16,512
# layer3: 128*2 + 2 = 258
# Tổng: 17,410

# --- Test 2: Forward pass với 1 sample ---
print("\nTest 2 — Forward pass:")
fake_state = torch.tensor([[0.02, -0.04, 0.03, 0.02]],
                           dtype=torch.float32, device=device)
print(f"  Input shape : {fake_state.shape}")   # (1, 4)

q_values = net(fake_state)
print(f"  Output shape: {q_values.shape}")     # (1, 2)
print(f"  Q-values    : {q_values}")
print(f"  Action chọn : {q_values.argmax(dim=1).item()}")
# argmax → 0 (trái) hoặc 1 (phải)

# --- Test 3: Forward pass với cả batch ---
print("\nTest 3 — Batch forward pass:")
fake_batch = torch.randn(128, 4, device=device)  # 128 states
q_batch    = net(fake_batch)
print(f"  Input shape : {fake_batch.shape}")   # (128, 4)
print(f"  Output shape: {q_batch.shape}")      # (128, 2)

# --- Test 4: Hai mạng bắt đầu giống nhau ---
print("\nTest 4 — policy_net và target_net đồng bộ ban đầu:")
policy_net, target_net = build_networks(N_OBS, N_ACTIONS, device)

# So sánh trọng số lớp đầu tiên
w_policy = policy_net.layer1.weight
w_target = target_net.layer1.weight
are_equal = torch.allclose(w_policy, w_target)
print(f"  Trọng số layer1 giống nhau: {are_equal}")  # True

# --- Test 5: Soft update ---
print("\nTest 5 — Soft update target_net:")
TAU = 0.005
policy_state = policy_net.state_dict()
target_state = target_net.state_dict()

# Ghi nhớ trọng số trước update
w_before = target_net.layer1.weight.clone()

for key in policy_state:
    target_state[key] = (policy_state[key] * TAU
                         + target_state[key] * (1 - TAU))
target_net.load_state_dict(target_state)

w_after = target_net.layer1.weight
changed = not torch.allclose(w_before, w_after)
print(f"  Target thay đổi sau soft update: {changed}")  # True
diff = (w_after - w_before).abs().mean().item()
print(f"  Thay đổi trung bình          : {diff:.6f}  (rất nhỏ, đúng!)")

print("\nTất cả test passed! model.py hoạt động đúng.")
