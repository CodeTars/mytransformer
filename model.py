import torch
import torch.nn.functional as F

# transformer流程
# 1. 输入嵌入：将输入的prompt分词、转换为带隐式维度向量
# 2. 位置编码：暂时实现绝对位置编码
# 3. n * TransformerBlock
#   3.a 单头注意力
#
# 暂时不考虑输入嵌入和绝对位置编码，从3开始

B = 10
prompt_len = 100
d = 100
d_k = 100
n = 24


class LayerNorm:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.eps = eps
        self.gamma = torch.ones(dim)
        self.beta = torch.zeros(dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, Lq, d)
        mean = X.mean(dim=-1, keepdim=True)  # (B, Lq, 1)
        var = ((X - mean) ** 2).mean(dim=-1, keepdim=True)  # (B, Lq, 1)
        X_hat = (X - mean) / torch.sqrt(var + self.eps)
        return self.gamma * X_hat + self.beta


class FFN:
    # 两层MLP，先升维再降维度, y = GELU(x * W1 + b1) * W2 + b2
    # x: (d,), x * W1: (d_up,), W1: (d, d_up), b1: (d_up,)
    # GELU(...): (d_up,), W2: (d_up, d), b2: (d,)
    def __init__(self):
        self.d_up = d * 4
        self.W1 = torch.randn(d, self.d_up)
        self.b1 = torch.randn(self.d_up)
        self.W2 = torch.randn(self.d_up, d)
        self.b2 = torch.randn(d)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, L, d)
        return F.gelu(X @ self.W1 + self.b1) @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self):
        # 暂时实现单头注意力
        self.W_Q = torch.randn(d, d_k)
        self.W_K = torch.randn(d, d_k)
        self.W_V = torch.randn(d, d_k)
        self.W_O = torch.randn(d_k, d)
        self.layer_norm1 = LayerNorm(d)
        self.layer_norm2 = LayerNorm(d)
        self.ffn = FFN()
        self.reset_cache()

    def reset_cache(self):
        # KV cache，序列维初始长度为 0
        self.K = torch.zeros(B, 0, d_k)
        self.V = torch.zeros(B, 0, d_k)

    def _causal_mask(self, q_len: int, k_len: int) -> torch.Tensor:
        # query 绝对位置: [k_len - q_len, ..., k_len - 1]
        # 位置 i 只能看见 j <= i
        # prefill (q_len == k_len): 标准上三角
        # decode  (q_len == 1)   : 可看见全部已缓存 key，等价于不遮挡
        offset = k_len - q_len
        return torch.triu(
            torch.full((q_len, k_len), float("-inf")),
            diagonal=offset + 1,
        )

    def attention(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, Lq, d)
        # self.K/V 已包含当前步（含 cache）
        Q = X @ self.W_Q  # (B, Lq, d_k)
        Lq, Lk = Q.shape[-2], self.K.shape[-2]

        scores = Q @ self.K.transpose(-2, -1) / (d_k ** 0.5)  # (B, Lq, Lk)
        scores = scores + self._causal_mask(Lq, Lk)
        A = torch.softmax(scores, dim=-1)  # (B, Lq, Lk)
        Y = A @ self.V  # (B, Lq, d_k)
        return Y @ self.W_O  # (B, Lq, d)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, Lq, d)  —— Pre-LN
        X_norm = self.layer_norm1.forward(X)

        # 投影并追加到 KV cache, (B, Lk, d)，满足 1 <= Lq <= Lk
        self.K = torch.cat([self.K, X_norm @ self.W_K], dim=-2)
        self.V = torch.cat([self.V, X_norm @ self.W_V], dim=-2)

        H = X + self.attention(X_norm) # (B, Lq, d)
        Y = H + self.ffn.forward(self.layer_norm2.forward(H)) # (B, Lq, d)
        return Y


def main():
    X = 0.02 * torch.randn(B, prompt_len, d)  # 99.7% 落在 [-0.06, 0.06]
    blocks = [TransformerBlock() for _ in range(n)]

    # prefill：整段 prompt 一次过完
    for block in blocks:
        X = block.forward(X)
    print("prefill:", X.shape)

    # decode：每次 1 个 token，KV cache 继续增长
    token = 0.02 * torch.randn(B, 1, d)
    for block in blocks:
        token = block.forward(token)
    print("decode:", token.shape)


if __name__ == "__main__":
    main()
