# config.py
# Tất cả hyperparameter tập trung ở đây — chỉ sửa file này khi tune

import torch

# ── Replay Memory ─────────────────────────────────────────────
MEMORY_CAPACITY = 10_000   # lưu tối đa 10,000 transitions

# ── Training ──────────────────────────────────────────────────
BATCH_SIZE = 128    # số transitions mỗi lần học
GAMMA      = 0.99   # discount factor: coi trọng reward tương lai
                    # 0.99 = rất coi trọng; 0.5 = chỉ quan tâm gần

# ── Epsilon-Greedy ────────────────────────────────────────────
EPS_START = 0.9     # xác suất random lúc đầu (90% thử ngẫu nhiên)
EPS_END   = 0.01    # xác suất random tối thiểu (luôn giữ 1% explore)
EPS_DECAY = 2500    # càng lớn = decay càng chậm
                    # ở bước 2500: epsilon ≈ EPS_END + 0.37*(EPS_START-EPS_END)

# ── Target Network ────────────────────────────────────────────
TAU = 0.005         # tốc độ soft update target_net
                    # θ_target = TAU*θ_policy + (1-TAU)*θ_target
                    # 0.005 = target_net thay đổi rất chậm → ổn định

# ── Optimizer ─────────────────────────────────────────────────
LR = 3e-4           # learning rate cho AdamW (0.0003)
                    # nhỏ hơn default 1e-3 để training ổn định hơn

# ── Training Loop ─────────────────────────────────────────────
NUM_EPISODES = 600  # số episodes nếu có GPU
NUM_EPISODES_CPU = 500  # số episodes nếu chỉ có CPU

# ── Device (tự động chọn) ─────────────────────────────────────
DEVICE = torch.device(
    "cuda"  if torch.cuda.is_available()        else
    "mps"   if torch.backends.mps.is_available() else
    "cpu"
)