# agent.py
# Chứa logic ra quyết định (select_action) và tối ưu hóa (optimize_model)
# Bước này chỉ làm select_action — optimize_model sẽ thêm ở bước 6

import math
import random
import torch
import torch.nn as nn
import torch.optim as optim

import config
from memory import ReplayMemory, Transition


class Agent:
    """
    Agent DQN: đưa ra quyết định và học từ kinh nghiệm.

    Tách Agent thành class riêng vì:
    - Gom trạng thái (steps_done, memory) vào một chỗ
    - Dễ tái sử dụng cho môi trường khác trong đồ án
    - Dễ lưu/load checkpoint
    """

    def __init__(self, policy_net, target_net, optimizer, n_actions: int):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer  = optimizer
        self.n_actions  = n_actions
        self.memory     = ReplayMemory(config.MEMORY_CAPACITY)

        # Đếm tổng số bước đã thực hiện — dùng để tính epsilon
        # Không đặt trong episode loop vì epsilon decay liên tục
        # xuyên suốt toàn bộ training, không reset mỗi episode
        self.steps_done = 0

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Chọn action theo chiến lược ε-Greedy.

        state: tensor shape (1, n_observations) — trạng thái hiện tại

        Trả về: tensor shape (1, 1) — action được chọn (0 hoặc 1)

        Tại sao trả về shape (1,1) thay vì scalar?
        Vì sau này memory.push() cần tensor để torch.cat() hoạt động
        khi unpack batch trong optimize_model().
        """
        sample = random.random()   # số ngẫu nhiên [0, 1)

        # Tính epsilon hiện tại theo công thức decay mũ
        # Ở bước 0   : eps ≈ EPS_START = 0.9
        # Ở bước 2500: eps ≈ EPS_END + 0.37*(EPS_START-EPS_END) ≈ 0.34
        # Ở bước 10000: eps ≈ EPS_END = 0.01
        eps_threshold = (
            config.EPS_END
            + (config.EPS_START - config.EPS_END)
            * math.exp(-1.0 * self.steps_done / config.EPS_DECAY)
        )
        self.steps_done += 1

        if sample > eps_threshold:
            # ── EXPLOIT: dùng model để chọn action tốt nhất ──
            # torch.no_grad() vì đây là inference, không cần gradient
            # Tắt gradient tiết kiệm bộ nhớ và tăng tốc ~20%
            with torch.no_grad():
                # policy_net(state) trả về tensor shape (1, 2)
                # ví dụ: tensor([[0.312, -0.184]])
                # .max(1) tìm max theo chiều action (dim=1)
                # .indices lấy index của max → action có Q-value cao nhất
                # .view(1, 1) đảm bảo shape đúng (1, 1)
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            # ── EXPLORE: chọn ngẫu nhiên ──
            # env.action_space.sample() cũng được, nhưng dùng
            # randint trực tiếp để không cần truyền env vào agent
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=config.DEVICE,
                dtype=torch.long
            )

    def store_transition(self, state, action, next_state, reward):
        """
        Lưu transition vào memory — wrapper gọn hơn.
        next_state = None nếu episode kết thúc (terminal state).
        """
        self.memory.push(state, action, next_state, reward)

    def soft_update_target(self):
        """
        Cập nhật target_net từng bước nhỏ từ policy_net.

        Công thức: θ_target = TAU*θ_policy + (1-TAU)*θ_target
        TAU=0.005 nghĩa là target chỉ di chuyển 0.5% về phía policy
        mỗi bước — cực kỳ chậm và ổn định.

        Tại sao không copy thẳng (hard update)?
        Hard update tạo ra "cú nhảy" đột ngột trong target mỗi N bước.
        Soft update tạo ra gradient mượt mà hơn, training ổn định hơn.
        """
        policy_state = self.policy_net.state_dict()
        target_state = self.target_net.state_dict()

        for key in policy_state:
            target_state[key] = (
                policy_state[key] * config.TAU
                + target_state[key] * (1 - config.TAU)
            )
        self.target_net.load_state_dict(target_state)

    # ── optimize_model: phần mới ──────────────────────────────
    def optimize_model(self):
        """
        Thực hiện một bước gradient descent để cải thiện policy_net.

        Quy trình:
          1. Lấy ngẫu nhiên BATCH_SIZE transitions từ memory
          2. Tính Q-value dự đoán (từ policy_net)
          3. Tính Q-value mục tiêu (từ target_net + Bellman)
          4. Tính Huber loss giữa dự đoán và mục tiêu
          5. Backprop + cập nhật trọng số policy_net
        """

        # ── Bước 0: Chưa đủ dữ liệu thì bỏ qua ──────────────
        # Phải có ít nhất BATCH_SIZE transitions mới sample được
        if not self.memory.is_ready(config.BATCH_SIZE):
            return

        # ── Bước 1: Sample batch ngẫu nhiên ──────────────────
        transitions = self.memory.sample(config.BATCH_SIZE)

        # Chuyển list of Transition → Transition of lists
        # Xem giải thích chi tiết ở bước 3 (test_memory.py)
        batch = Transition(*zip(*transitions))

        # ── Bước 2: Xác định terminal states ─────────────────
        # Terminal state (done=True) có next_state = None
        # Cần mask để không tính Q(s') cho những state cuối
        # vì Q(terminal) = 0 theo định nghĩa
        #
        # non_final_mask: tensor bool shape (BATCH_SIZE,)
        # True  = transition này chưa kết thúc → có s' hợp lệ
        # False = transition này là terminal   → s' = None
        non_final_mask = torch.tensor(
            tuple(s is not None for s in batch.next_state),
            device=config.DEVICE,
            dtype=torch.bool
        )

        # Gom tất cả next_state không phải None thành một batch
        # shape: (số_non_terminal, 4)
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        # Gom state, action, reward thành batch tensor
        state_batch  = torch.cat(batch.state)    # (128, 4)
        action_batch = torch.cat(batch.action)   # (128, 1)
        reward_batch = torch.cat(batch.reward)   # (128,)

        # ── Bước 3: Tính Q(s, a) — predicted ─────────────────
        # policy_net(state_batch) → shape (128, 2)
        # Mỗi hàng là [Q(s, left), Q(s, right)]
        #
        # .gather(1, action_batch) chọn Q-value của đúng action
        # đã thực hiện trong transition đó
        #
        # Ví dụ hàng 0: Q-values = [0.3, 0.7], action = 1
        # → gather lấy 0.7 (Q của action "phải")
        #
        # Kết quả shape: (128, 1)
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch
        )

        # ── Bước 4: Tính V(s') = max_a' Q(s', a') — target ───
        # Khởi tạo tất cả bằng 0 (terminal states giữ 0)
        next_state_values = torch.zeros(
            config.BATCH_SIZE, device=config.DEVICE
        )

        # Chỉ tính Q(s') cho non-terminal states
        # Dùng target_net (không phải policy_net!) để ổn định
        # torch.no_grad() vì target chỉ cung cấp nhãn, không học
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )

        # Bellman: target = r + γ * max_a' Q(s', a')
        # shape: (128,)
        expected_state_action_values = (
            next_state_values * config.GAMMA
        ) + reward_batch

        # ── Bước 5: Tính Huber Loss ───────────────────────────
        # state_action_values  shape: (128, 1) — predicted
        # expected_...unsqueeze(1) shape: (128, 1) — target
        #
        # SmoothL1Loss chính là Huber Loss trong PyTorch
        # β=1: hoạt động như MSE khi |δ|<1, MAE khi |δ|>1
        criterion = nn.SmoothL1Loss()
        loss = criterion(
            state_action_values,
            expected_state_action_values.unsqueeze(1)
        )

        # ── Bước 6: Backpropagation ───────────────────────────
        self.optimizer.zero_grad()
        # zero_grad() xóa gradient cũ từ bước trước
        # (PyTorch cộng dồn gradient, không xóa tự động)

        loss.backward()
        # Tính gradient của loss theo mọi tham số của policy_net

        # Clip gradient để tránh "exploding gradient"
        # Giới hạn gradient ở [-100, 100] — tránh bước nhảy quá lớn
        torch.nn.utils.clip_grad_value_(
            self.policy_net.parameters(), 100
        )

        self.optimizer.step()
        # Cập nhật trọng số: θ = θ − lr * gradient

        return loss.item()   # trả về để theo dõi quá trình học