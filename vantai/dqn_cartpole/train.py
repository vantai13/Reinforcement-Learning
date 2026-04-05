# train.py
# Vòng lặp huấn luyện chính — ghép tất cả module lại

import math
import torch
import torch.optim as optim
import gymnasium as gym

import config
from model import build_networks
from agent import Agent
from plot_utils import (record_episode, plot_training,
                        print_progress)


def train():
    """
    Hàm training chính.
    Trả về agent đã train để có thể dùng tiếp (evaluate, save...).
    """

    # ── Khởi tạo môi trường ───────────────────────────────────
    env = gym.make("CartPole-v1")
    n_obs     = env.observation_space.shape[0]   # 4
    n_actions = env.action_space.n               # 2

    print(f"Device    : {config.DEVICE}")
    print(f"Env       : CartPole-v1")
    print(f"n_obs     : {n_obs}  |  n_actions: {n_actions}")
    print(f"Episodes  : {config.NUM_EPISODES}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print("-" * 60)

    # ── Khởi tạo networks, optimizer, agent ───────────────────
    policy_net, target_net = build_networks(
        n_obs, n_actions, config.DEVICE
    )
    optimizer = optim.AdamW(
        policy_net.parameters(),
        lr=config.LR,
        amsgrad=True
        # amsgrad=True: biến thể ổn định hơn của Adam
        # lưu giá trị max gradient để tránh learning rate giảm quá nhanh
    )
    agent = Agent(policy_net, target_net, optimizer, n_actions)

    # Số episodes tùy theo có GPU không
    num_episodes = (config.NUM_EPISODES
                    if config.DEVICE.type != 'cpu'
                    else config.NUM_EPISODES_CPU)

    # ── Vòng lặp episode ──────────────────────────────────────
    for i_episode in range(num_episodes):

        # Reset môi trường, lấy state ban đầu
        state, _ = env.reset()

        # Chuyển numpy array → tensor PyTorch
        # unsqueeze(0): thêm chiều batch → shape (1, 4)
        # Tại sao cần batch dim? Vì policy_net.forward()
        # luôn mong đợi input shape (batch, features)
        state = torch.tensor(
            state, dtype=torch.float32, device=config.DEVICE
        ).unsqueeze(0)

        step_losses = []   # theo dõi loss trong episode này

        # ── Vòng lặp step (1 episode) ─────────────────────────
        # itertools.count() = 0, 1, 2, 3, ... vô hạn
        # vòng lặp tự dừng khi gặp break (done=True)
        from itertools import count
        for t in count():

            # 1. Chọn action
            action = agent.select_action(state)

            # 2. Thực thi action trong môi trường
            observation, reward, terminated, truncated, _ = \
                env.step(action.item())
            # .item() chuyển tensor scalar → Python int

            reward_t = torch.tensor(
                [reward], device=config.DEVICE
            )
            done = terminated or truncated

            # 3. Xác định next_state
            # Nếu terminated (cột ngã/xe ra ngoài):
            #   next_state = None → optimize_model biết đây là terminal
            # Nếu truncated (đủ 500 steps, thắng!):
            #   next_state vẫn hợp lệ về mặt kỹ thuật
            #   nhưng ta cũng kết thúc episode
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation,
                    dtype=torch.float32,
                    device=config.DEVICE
                ).unsqueeze(0)

            # 4. Lưu transition vào memory
            agent.store_transition(state, action,
                                   next_state, reward_t)

            # 5. Chuyển sang state mới
            state = next_state

            # 6. Một bước optimize + soft update target
            loss = agent.optimize_model()
            if loss is not None:
                step_losses.append(loss)

            agent.soft_update_target()

            # 7. Kiểm tra kết thúc episode
            if done:
                duration = t + 1
                avg_loss = (sum(step_losses) / len(step_losses)
                            if step_losses else 0.0)

                record_episode(duration, avg_loss)

                # Tính epsilon hiện tại để in ra
                current_eps = (
                    config.EPS_END
                    + (config.EPS_START - config.EPS_END)
                    * math.exp(-agent.steps_done / config.EPS_DECAY)
                )

                # In tiến độ mỗi 10 episodes
                if (i_episode + 1) % 10 == 0:
                    print_progress(
                        i_episode + 1, duration,
                        avg_loss, current_eps
                    )

                # Cập nhật đồ thị mỗi 20 episodes
                if (i_episode + 1) % 20 == 0:
                    plot_training()

                break   # thoát vòng lặp step, sang episode mới

    # ── Kết thúc training ─────────────────────────────────────
    print("\n" + "=" * 60)
    print("Training hoàn tất!")
    from plot_utils import episode_durations, recent_durations
    if episode_durations:
        print(f"  Duration tốt nhất : {max(episode_durations)}")
        print(f"  Avg 100 ep cuối   : "
              f"{sum(list(recent_durations)[-100:]) / min(100, len(recent_durations)):.1f}")
    print("=" * 60)

    # Vẽ kết quả cuối cùng
    plot_training(show_result=True)

    env.close()
    return agent