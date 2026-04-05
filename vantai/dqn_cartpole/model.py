# model.py
# Định nghĩa kiến trúc mạng DQN và cách khởi tạo hai mạng

import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """
    Mạng Q-Network: nhận state, trả về Q-value cho mỗi action.

    Tại sao kế thừa nn.Module?
    nn.Module là lớp cơ sở của mọi mạng trong PyTorch.
    Kế thừa nó giúp ta được miễn phí:
      - Tự động track parameters (để optimizer biết cần cập nhật gì)
      - .to(device) chuyển cả mạng sang GPU/CPU
      - .state_dict() / .load_state_dict() để save/load model
      - .train() / .eval() để chuyển chế độ
    """

    def __init__(self, n_observations: int, n_actions: int):
        """
        n_observations: số chiều của state vector (CartPole = 4)
        n_actions     : số actions có thể thực hiện (CartPole = 2)
        """
        super(DQN, self).__init__()

        # 3 lớp Linear (fully connected)
        # Linear(in, out) tạo ma trận trọng số shape (in, out)
        # và vector bias shape (out,)
        self.layer1 = nn.Linear(n_observations, 128)
        # 4 → 128: học các pattern cơ bản từ state

        self.layer2 = nn.Linear(128, 128)
        # 128 → 128: học các pattern phức tạp hơn

        self.layer3 = nn.Linear(128, n_actions)
        # 128 → 2: output Q-value cho từng action
        # KHÔNG có activation — Q-value là số thực tự do

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: tính Q-values từ state.

        x có shape (batch_size, n_observations)
        ví dụ: (128, 4) khi training, hoặc (1, 4) khi chọn action

        Tại sao F.relu thay vì nn.ReLU()?
        Cả hai đều cho kết quả giống nhau.
        F.relu là functional (không có tham số học),
        dùng trong forward() gọn hơn vì không cần khai báo trong __init__
        """
        x = F.relu(self.layer1(x))
        # shape: (batch, 4) → (batch, 128), các âm thành 0

        x = F.relu(self.layer2(x))
        # shape: (batch, 128) → (batch, 128), lọc tiếp

        return self.layer3(x)
        # shape: (batch, 128) → (batch, 2)
        # KHÔNG relu ở đây — Q-value có thể âm


def build_networks(n_observations: int, n_actions: int, device: torch.device):
    """
    Tạo policy_net và target_net, đồng bộ trọng số ban đầu.

    Tại sao tách ra hàm riêng?
    Dễ tái sử dụng, dễ test, và nhắc nhở rõ ràng rằng
    hai mạng phải bắt đầu với trọng số GIỐNG HỆT nhau.

    Tại sao target_net phải copy từ policy_net?
    Nếu không, ngay bước đầu tiên target đã khác policy —
    loss sẽ tính sai ngay từ đầu.
    """
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)

    # Copy TOÀN BỘ trọng số từ policy sang target
    # state_dict() trả về dict {tên_layer: tensor_trọng_số}
    target_net.load_state_dict(policy_net.state_dict())

    # target_net KHÔNG bao giờ tự học qua backprop
    # Nó chỉ nhận soft update thủ công từ policy_net
    # eval() tắt dropout/batchnorm (dù mạng này không có,
    # nhưng là thói quen tốt để đánh dấu "mạng này không train")
    target_net.eval()

    return policy_net, target_net