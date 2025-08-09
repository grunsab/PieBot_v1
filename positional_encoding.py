import math
import torch
import torch.nn as nn


class ChessSquareEmbedding(nn.Module):
    """
    Simple learnable square embeddings that combine file/rank/diagonals.
    Useful as a drop-in when fancier encodings are disabled.
    """

    def __init__(self, d_model, max_seq_len=64):
        super().__init__()
        self.square = nn.Parameter(torch.randn(1, max_seq_len, d_model))

    def forward(self, x):
        return x + self.square[:, : x.size(1), :]


class RoPE2D(nn.Module):
    """
    2D Rotary Positional Embedding for an 8x8 chess grid.
    Applies rotation to query/key pairs per head across file and rank axes.

    Reference: Su et al. 2021; extended to 2D by composing file/rank rotations.
    """

    def __init__(self, dim_head, seq_len=64, board_size=8):
        super().__init__()
        assert dim_head % 4 == 0, "dim_head must be divisible by 4 for 2D RoPE"
        self.dim_head = dim_head
        self.board_size = board_size
        self.seq_len = seq_len

        half = dim_head // 2
        inv_freq = 1.0 / (10000 ** (torch.arange(0, half, 2).float() / half))

        # Precompute for files and ranks separately
        files = torch.arange(board_size).float()
        ranks = torch.arange(board_size).float()
        freqs_file = torch.einsum('i,j->ij', files, inv_freq)  # [8, half/2]
        freqs_rank = torch.einsum('i,j->ij', ranks, inv_freq)  # [8, half/2]

        # Convert to cos/sin and register buffers
        self.register_buffer('cos_file', torch.cos(freqs_file).repeat_interleave(2, dim=1))
        self.register_buffer('sin_file', torch.sin(freqs_file).repeat_interleave(2, dim=1))
        self.register_buffer('cos_rank', torch.cos(freqs_rank).repeat_interleave(2, dim=1))
        self.register_buffer('sin_rank', torch.sin(freqs_rank).repeat_interleave(2, dim=1))

    @staticmethod
    def _rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).flatten(-2)

    def apply_rope(self, t, files, ranks):
        """Apply 2D RoPE to tensor t shaped [b, h, n, d]."""
        d = t.size(-1)
        half = d // 2
        t_file, t_rank = t[..., :half], t[..., half:]

        cos_f = self.cos_file[files]  # [n, half]
        sin_f = self.sin_file[files]
        cos_r = self.cos_rank[ranks]
        sin_r = self.sin_rank[ranks]

        def rope_1d(x, cos, sin):
            return (x * cos) + (self._rotate_half(x) * sin)

        t_file = rope_1d(t_file, cos_f, sin_f)
        t_rank = rope_1d(t_rank, cos_r, sin_r)
        return torch.cat([t_file, t_rank], dim=-1)

    def forward(self, q, k):
        # q, k: [b, h, n, d]
        n = q.size(2)
        # map linear 0..63 -> file/rank
        idx = torch.arange(n, device=q.device)
        files = (idx % self.board_size).long()
        ranks = (idx // self.board_size).long()
        return self.apply_rope(q, files, ranks), self.apply_rope(k, files, ranks)


class ALiBi2D(nn.Module):
    """
    2D ALiBi bias for attention on chess boards.
    Provides linear penalties proportional to file/rank distance per head.
    """

    def __init__(self, num_heads, max_offset=7, board_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.board_size = board_size

        # Different slopes per head (as in original ALiBi), shared for file/rank
        def head_slopes(n):
            # from the ALiBi paper
            def get_slopes_power_of_2(n):
                start = 2 ** (-2 ** -(math.log2(n) - 3))
                ratio = start
                return [start * (ratio ** i) for i in range(n)]

            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power = 2 ** math.floor(math.log2(n))
                return (
                    get_slopes_power_of_2(closest_power)
                    + head_slopes(2 * closest_power)[0::2][: n - closest_power]
                )

        slopes = torch.tensor(head_slopes(num_heads), dtype=torch.float32)
        self.register_buffer('slopes', slopes)

    def forward(self, seq_len):
        idx = torch.arange(seq_len, device=self.slopes.device)
        f = (idx % self.board_size).unsqueeze(0)
        r = (idx // self.board_size).unsqueeze(0)

        df = (f.t() - f).abs().float()  # [n, n]
        dr = (r.t() - r).abs().float()

        dist = df + dr  # Manhattan distance on board
        # [h, n, n]
        bias = -self.slopes.view(-1, 1, 1) * dist.unsqueeze(0).to(self.slopes.device)
        return bias.unsqueeze(0)  # [1, h, n, n]
