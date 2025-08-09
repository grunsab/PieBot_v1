# Luna Transformer (Chess)

Luna is a transformer-based network for chess with chess-aware positional encodings and attention biases.

Highlights:
- Input planes configurable (16 classic or 112 enhanced).
- Optional 2D RoPE and 2D ALiBi in attention for board spatial bias.
- Relative file/rank bias and smolgen-style global mixing.
- Policy head outputs 64×72 logits (flattened to 4608) to match existing pipeline.
- Training supports CCRL (supervised), RL (self-play), or mixed.

Flags of interest in train_luna.py:
- --input-planes {16|112}
- --use-rope, --use-alibi
- --mode {supervised|rl|mixed}
- --mixed-ratio
 - --entropy-coef <float> (entropy bonus for RL policy; typically small like 0.01–0.05)

 Recommended combos:
 - Supervised pretrain: --mode supervised --input-planes 112 --use-rope --use-alibi --mixed-precision
 - RL finetune: --mode rl --use-rope --use-alibi --swa --entropy-coef 0.02

Compatibility:
- Policy target shape 4608 and legal move masks (72,8,8) remain unchanged.
- Mixed datasets with different plane counts are padded/truncated to --input-planes at runtime.

Future work:
- Add piece-aware bias tables, check detection head, and auxiliary heads for material/king safety.
- Add test coverage for encoder plane swap and mask correctness.
