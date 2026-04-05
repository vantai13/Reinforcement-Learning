# plot_utils.py
# Công cụ vẽ đồ thị theo dõi quá trình training

import matplotlib
import matplotlib.pyplot as plt
import torch
from collections import deque

# Dùng backend không cần GUI khi chạy trên server/colab
# Nếu chạy local có thể bỏ dòng này
# matplotlib.use('Agg')

episode_durations = []       # lưu duration từng episode
episode_losses    = []       # lưu loss trung bình từng episode
recent_durations  = deque(maxlen=100)   # cửa sổ 100 episode gần nhất

# Kiểm tra có đang chạy trong Jupyter/IPython không
IS_IPYTHON = 'inline' in matplotlib.get_backend()
if IS_IPYTHON:
    from IPython import display

plt.ion()   # bật interactive mode — đồ thị tự cập nhật


def record_episode(duration: int, avg_loss: float = None):
    """
    Ghi lại kết quả một episode vừa kết thúc.
    Gọi sau mỗi episode trong training loop.
    """
    episode_durations.append(duration)
    recent_durations.append(duration)
    if avg_loss is not None:
        episode_losses.append(avg_loss)


def plot_training(show_result: bool = False):
    """
    Vẽ đồ thị duration và đường trung bình 100 episode.

    show_result=False : vẽ trong lúc training (cập nhật liên tục)
    show_result=True  : vẽ kết quả cuối cùng
    """
    plt.figure(1)
    plt.clf()   # xóa figure cũ để vẽ lại

    durations_t = torch.tensor(episode_durations, dtype=torch.float)

    if show_result:
        plt.title('Kết quả training hoàn tất')
    else:
        ep   = len(episode_durations)
        best = int(max(episode_durations)) if episode_durations else 0
        avg  = int(sum(recent_durations) / len(recent_durations)) if recent_durations else 0
        plt.title(f'Training... | Episode: {ep} | Best: {best} | Avg100: {avg}')

    plt.xlabel('Episode')
    plt.ylabel('Duration (steps)')
    plt.ylim(0, 520)

    # Đường duration thô — xanh nhạt
    plt.plot(durations_t.numpy(), color='#85B7EB', alpha=0.6,
             linewidth=0.8, label='Duration')

    # Đường trung bình 100 episode — xanh đậm
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        # unfold: tạo cửa sổ trượt 100 episode
        # mean(1): lấy trung bình mỗi cửa sổ
        # Thêm 99 số 0 ở đầu để căn chỉnh với trục x
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy(), color='#185FA5',
                 linewidth=2, label='Avg 100 ep')
        plt.legend(loc='upper left', fontsize=9)

    # Đường mục tiêu 500 steps
    plt.axhline(y=500, color='#3B6D11', linewidth=1,
                linestyle='--', alpha=0.7, label='Target (500)')

    plt.tight_layout()
    plt.pause(0.001)

    if IS_IPYTHON:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def print_progress(episode: int, duration: int,
                   avg_loss: float, eps: float):
    """
    In tiến độ ra terminal — dễ theo dõi khi không có GUI.
    """
    avg100 = (sum(recent_durations) / len(recent_durations)
              if recent_durations else 0)
    bar_len  = 20
    prog     = min(avg100 / 500, 1.0)
    filled   = int(bar_len * prog)
    bar      = '█' * filled + '░' * (bar_len - filled)

    print(f"Ep {episode:>4} | dur={duration:>3} | "
          f"avg100={avg100:>6.1f} | "
          f"loss={avg_loss:>8.5f} | "
          f"eps={eps:.3f} | [{bar}] {prog*100:.0f}%")