# memory.py
# Chứa toàn bộ logic lưu trữ kinh nghiệm của agent

from collections import namedtuple, deque
import random


# ---------------------------------------------------------------
# Transition: đơn vị dữ liệu nhỏ nhất trong Replay Memory
# namedtuple giống như một struct đơn giản — truy cập bằng tên
# thay vì index, giúp code dễ đọc hơn nhiều.
#
# Ví dụ: t.state thay vì t[0]
#         t.reward thay vì t[3]
# ---------------------------------------------------------------
Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory:
    """
    Cyclic buffer lưu trữ các Transition đã trải qua.

    Tại sao dùng deque thay vì list thông thường?
    - deque(maxlen=N) tự động xóa phần tử cũ nhất khi đầy
    - Không cần viết logic "nếu đầy thì ghi đè" thủ công
    - Thao tác append() ở deque nhanh hơn list khi có maxlen
    """

    def __init__(self, capacity: int):
        # maxlen đảm bảo bộ nhớ không vượt quá capacity
        # Khi memory đầy, transition cũ nhất tự động bị xóa
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """
        Lưu một transition vào memory.

        Tại sao dùng *args thay vì truyền trực tiếp?
        Transition(*args) giúp unpacking linh hoạt hơn —
        bạn có thể gọi push(s, a, ns, r) gọn hơn là
        push(Transition(s, a, ns, r))
        """
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size: int):
        """
        Lấy ngẫu nhiên batch_size transitions để training.

        Tại sao random.sample thay vì lấy batch liên tiếp?
        Đây là trái tim của Replay Memory! Lấy ngẫu nhiên phá vỡ
        sự tương quan giữa các bước liên tiếp. Nếu lấy liên tiếp
        (step 100, 101, 102...) thì mạng sẽ học một chiều duy nhất.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """
        Trả về số transition hiện có.

        Cần thiết vì trong training loop ta kiểm tra:
        if len(memory) < BATCH_SIZE: return  # chưa đủ dữ liệu
        """
        return len(self.memory)

    def is_ready(self, batch_size: int) -> bool:
        """
        Kiểm tra đã đủ dữ liệu để bắt đầu training chưa.
        Hàm tiện ích thêm vào cho code sạch hơn.
        """
        return len(self) >= batch_size