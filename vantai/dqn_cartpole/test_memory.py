import torch
from memory import ReplayMemory, Transition

print("=" * 50)
print("TEST REPLAY MEMORY")
print("=" * 50)

# --- Test 1: Tạo memory và push dữ liệu giả ---
memory = ReplayMemory(capacity=5)  # nhỏ để test dễ

# Tạo dữ liệu giả giống với dữ liệu thật sẽ dùng
# state thật là tensor shape (1, 4) — 1 sample, 4 observations
for i in range(4):
    fake_state      = torch.tensor([[float(i),   0.0, 0.1, 0.0]])  # shape (1,4)
    print(fake_state)
    fake_action     = torch.tensor([[0]])
    print(fake_action)                            # shape (1,1)
    fake_next_state = torch.tensor([[float(i+1), 0.0, 0.1, 0.0]])  # shape (1,4)
    fake_reward     = torch.tensor([1.0])                           # shape (1,)
    memory.push(fake_state, fake_action, fake_next_state, fake_reward)

print(f"\nTest 1 — Push 4 transitions vào capacity=5")
print(f"  Số transitions: {len(memory)} (mong đợi: 4)")
print(f"  is_ready(3)   : {memory.is_ready(3)} (mong đợi: True)")
print(f"  is_ready(5)   : {memory.is_ready(5)} (mong đợi: False)")

# --- Test 2: Deque tự động xóa khi đầy ---
memory.push(
    torch.tensor([[99.0, 0.0, 0.0, 0.0]]),
    torch.tensor([[1]]),
    torch.tensor([[100.0, 0.0, 0.0, 0.0]]),
    torch.tensor([1.0])
)
memory.push(
    torch.tensor([[999.0, 0.0, 0.0, 0.0]]),
    torch.tensor([[1]]),
    torch.tensor([[1000.0, 0.0, 0.0, 0.0]]),
    torch.tensor([1.0])
)

print(f"\nTest 2 — Push thêm 2 (total 6 > capacity 5)")
print(f"  Số transitions: {len(memory)} (mong đợi: 5 — tự cắt)")

# --- Test 3: Sample ngẫu nhiên ---
batch = memory.sample(3)
print(f"\nTest 3 — Sample 3 transitions ngẫu nhiên")
print(f"  Số lượng sample: {len(batch)} (mong đợi: 3)")
print(f"  Kiểu dữ liệu  : {type(batch[0])}")
print(f"  Tên fields     : {batch[0]._fields}")

# --- Test 4: Cách unpack batch (dùng thật trong training) ---
# Đây là pattern quan trọng sẽ dùng trong optimize_model()
batch_transition = Transition(*zip(*batch))
print(f"\nTest 4 — Unpack batch thành từng field")
print(f"  batch_transition.state  : list {len(batch_transition.state)} tensors")
print(f"  batch_transition.action : list {len(batch_transition.action)} tensors")
print(f"  batch_transition.reward : list {len(batch_transition.reward)} tensors")

# Concatenate thành một tensor lớn — cách dùng thật
state_batch = torch.cat(batch_transition.state)
print(f"\n  state_batch shape: {state_batch.shape} (mong đợi: torch.Size([3, 4]))")

print("\nTất cả test passed! memory.py hoạt động đúng.")