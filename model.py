import torch

# transformer流程   
# 1. 输入嵌入：将输入的prompt分词、转换为带隐式维度向量
# 2. 位置编码：暂时实现绝对位置编码
# 3. n * TransformerBlock
#   3.a 单头注意力
#
# 暂时不考虑输入嵌入和绝对位置编码，输入3开始

B = 10
prompt_len = 100
d = 100
d_k = 100
n = 24


class LayerNorm:
    def __init__(self, eps, gamma, beta):
        self.eps = eps
        self.gamma = gamma
        self.beta = beta
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, L, d)
        mean = X.mean(dim = -1, keepdim = True) # (B, L, 1)
        var = ((X - mean) ** 2).mean(dim = -1, keepdim = True) # (B, L, 1)
        X_hat = (X - mean) / torch.sqrt(var + self.eps) # (B, L, d)
        Y = self.gamma * X_hat + self.beta  # (B, L, d)
        return Y


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
        return torch.nn.functional.gelu(X @ self.W1 + self.b1) @ self.W2 + self.b2


class TransformerBlock:
    def __init__(self):
        # 暂时实现单头注意力
        self.W_Q = torch.randn(d, d_k)
        self.W_K = torch.randn(d, d_k)
        self.W_V = torch.randn(d, d_k)
        self.W_O = torch.randn(d_k, d)
        self.layer_norm1 = LayerNorm(eps = 1e-5, gamma = torch.randn(d), beta = torch.randn(d))
        self.layer_norm2 = LayerNorm(eps = 1e-5, gamma = torch.randn(d), beta = torch.randn(d))
        self.ffn = FFN()
        self.K = torch.zeros(B, 0, d_k)
        self.V = torch.zeros(B, 0, d_k)
    
    def attention(self, X: torch.Tensor) -> torch.Tensor:
        # X: prefill: (B, L, d), decode: (B, 1, d)
        Q = X @ self.W_Q    # prefill: (B, L, d_k), decode: (B, 1, d_k)

        scores = Q @ self.K.transpose(-2, -1) / (d_k ** 0.5) # prefill: (B, L, L), decode: (B, 1, L)
        # 因果掩码
        q_len = Q.shape[-2], k_len = self.K.shape[-2]
        mask = torch.triu(torch.full((q_len, k_len), float("-inf")), diagonal = 1)
        A = torch.softmax(scores + mask, dim = -1) # 注意力权重矩阵 prefill: (B, L, L)
        Y = A @ self.V   # 注意力输出，根据注意力权重交换不同token的信息 prefill: (B, L, d_k), decode: (B, 1, d_k)

        return Y @ self.W_O # prefill: (B, L, d), decode: (B, 1, d)
        
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # X: (B, L, d) or (B, 1, d)
        # 做输入张量归一化
        X_norm = self.layer_norm1.forward(X)
        # K V投影
        k = X_norm @ self.W_K    # (B, L, d_k) or (B, 1, d_k)
        v = X_norm @ self.W_V    # (B, L, d_k) or (B, 1, d_k)
        self.K = torch.cat([self.K, k], dim=-2)
        self.V = torch.cat([self.V, v], dim=-2)
        # Self-Attention
        attention_result = self.attention(X_norm) # (B, L, d) or (B, 1, d)
        # 残差连接
        H = X + attention_result # (B, L, d)
        # 第二次张量归一化
        H_norm = self.layer_norm2.forward(H) # (B, L, d)
        # 做一次FFN
        ffn_result = self.ffn.forward(H_norm) # (B, L, d)
        # 残差连接
        Y = H + ffn_result # (B, L, d)
        return Y


def main():
    X = 0.02 * torch.randn(B, prompt_len, d)  # 99.7% 落在[-0.06, 0.06]
    blocks = [TransformerBlock() for i in range(0, n)]


if __name__ == "__main__":
    main()