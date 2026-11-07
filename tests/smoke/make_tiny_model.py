"""Build a tiny, random-weight Qwen2.5-VL (and a Qwen2-VL "student") for CPU smoke tests.

Nothing here is used for real results. It exists so every stage of the pipeline
(train, merge, GPTQ/AWQ, distill, GGUF export, llama.cpp eval) can be exercised
end to end on a laptop/CI box without downloading 7 GB of weights.

The tokenizer is a small byte-level BPE trained locally, with Qwen's special
tokens appended in Qwen's order, and Qwen2.5-VL's real chat template.
"""

import argparse
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, Regex
from transformers import (
    PreTrainedTokenizerFast,
    Qwen2VLImageProcessor,
    Qwen2_5_VLConfig,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLProcessor,
    Qwen2VLConfig,
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
)

QWEN_SPECIALS = [
    "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>",
    "<|box_start|>", "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>",
    "<|vision_end|>", "<|vision_pad|>", "<|image_pad|>", "<|video_pad|>",
]

# NB: vision depth >= 3 because llama.cpp's tensor map only maps merger.mlp.{bid} for bid < depth.

# Qwen2.5-VL-Instruct chat template (as shipped in chat_template.json).
CHAT_TEMPLATE = (
    "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}"
    "{% for message in messages %}{% if loop.first and message['role'] != 'system' %}"
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}"
    "<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}"
    "{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}"
    "{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}"
    "{% set image_count.value = image_count.value + 1 %}"
    "{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}"
    "<|vision_start|><|image_pad|><|vision_end|>"
    "{% elif content['type'] == 'video' or 'video' in content %}"
    "{% set video_count.value = video_count.value + 1 %}"
    "{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}"
    "<|vision_start|><|video_pad|><|vision_end|>"
    "{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n"
    "{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)

QWEN_PRETOKENIZE_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)

CORPUS = [
    "What color is the square? red blue green yellow white black orange purple",
    "How many circles are there? one two three four five 1 2 3 4 5 6 7 8 9 0",
    "Is there a circle in the image? yes no Answer the question using a single word or phrase.",
    "You are a helpful assistant. system user assistant shape color count left right top bottom",
] * 50


def build_tokenizer(vocab_size: int = 1200) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(QWEN_PRETOKENIZE_REGEX), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer=trainer)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>",
        additional_special_tokens=QWEN_SPECIALS, clean_up_tokenization_spaces=False,
    )
    fast.add_special_tokens({"additional_special_tokens": QWEN_SPECIALS})
    fast.chat_template = CHAT_TEMPLATE
    fast.padding_side = "left"
    return fast


def ids(tok, *names):
    return [tok.convert_tokens_to_ids(n) for n in names]


def text_cfg(tok, hidden, layers, inter):
    head_dim = 32
    return dict(
        vocab_size=len(tok), hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
        num_attention_heads=hidden // head_dim, num_key_value_heads=2, max_position_embeddings=4096,
        rms_norm_eps=1e-6, rope_parameters={"rope_type": "default", "rope_theta": 1e6, "mrope_section": [4, 6, 6]},
        bos_token_id=ids(tok, "<|endoftext|>")[0], eos_token_id=ids(tok, "<|im_end|>")[0],
        pad_token_id=ids(tok, "<|endoftext|>")[0], tie_word_embeddings=False,
    )


def pretrain(model, proc, data_jsonl, steps=500):
    """Give the random tiny model a real skill (colors only) so later stages have signal to move.
    Counting is left for LoRA fine-tuning / distillation to add."""
    import random as _r
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from vlmc.data import load_items
    from vlmc.hf import build_train_batch
    items = [it for it in load_items(data_jsonl) if it["answer_type"] == "other"]
    proc.image_processor.min_pixels, proc.image_processor.max_pixels = 3136, 12544
    proc.image_processor.size = {"shortest_edge": 3136, "longest_edge": 12544}
    proc.tokenizer.padding_side = "right"
    model.float().train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    rng = _r.Random(0)
    for s in range(steps):
        loss = model(**build_train_batch(proc, rng.sample(items, 16), "Answer the question using a single word or phrase.")).loss
        loss.backward()
        opt.step()
        opt.zero_grad()
    print(f"[tiny] pretrain final loss {loss.item():.3f}")
    proc.tokenizer.padding_side = "left"
    model.eval()


def build(out: Path, kind: str, tok, seed: int, pretrain_data=None):
    torch.manual_seed(seed)
    img_proc = Qwen2VLImageProcessor(min_pixels=28 * 28 * 4, max_pixels=28 * 28 * 16)
    image_id, video_id, vs_id, ve_id = ids(tok, "<|image_pad|>", "<|video_pad|>", "<|vision_start|>", "<|vision_end|>")
    common = dict(image_token_id=image_id, video_token_id=video_id, vision_start_token_id=vs_id,
                  vision_end_token_id=ve_id, tie_word_embeddings=False)
    if kind == "teacher":
        cfg = Qwen2_5_VLConfig(
            text_config=text_cfg(tok, hidden=128, layers=2, inter=256),
            vision_config=dict(depth=4, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=128,
                               fullatt_block_indexes=[1, 3], window_size=112, patch_size=14, spatial_merge_size=2),
            **common)
        model = Qwen2_5_VLForConditionalGeneration(cfg)
        from transformers import Qwen2VLVideoProcessor
        proc = Qwen2_5_VLProcessor(image_processor=img_proc, tokenizer=tok, chat_template=CHAT_TEMPLATE,
                                   video_processor=Qwen2VLVideoProcessor())
    else:  # student: Qwen2-VL architecture, same tokenizer (mirrors Qwen2-VL-2B vs Qwen2.5-VL-3B)
        cfg = Qwen2VLConfig(
            text_config=text_cfg(tok, hidden=128, layers=1, inter=256),
            vision_config=dict(depth=3, embed_dim=64, hidden_size=128, mlp_ratio=2, num_heads=4,
                               patch_size=14, spatial_merge_size=2),
            **common)
        model = Qwen2VLForConditionalGeneration(cfg)
        from transformers import Qwen2VLVideoProcessor
        proc = Qwen2VLProcessor(image_processor=img_proc, tokenizer=tok, chat_template=CHAT_TEMPLATE,
                                video_processor=Qwen2VLVideoProcessor())
    if pretrain_data:
        pretrain(model, proc, pretrain_data)
    model = model.to(torch.bfloat16)
    model.generation_config.eos_token_id = ids(tok, "<|im_end|>")[0]
    model.generation_config.pad_token_id = ids(tok, "<|endoftext|>")[0]
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    proc.save_pretrained(out)
    n = sum(p.numel() for p in model.parameters())
    print(f"[tiny] {kind}: {n/1e6:.2f}M params -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="smoke_artifacts/models")
    ap.add_argument("--pretrain-data", help="synthetic train.jsonl; teaches the tiny models colors only")
    a = ap.parse_args()
    tok = build_tokenizer()
    build(Path(a.out) / "tiny-qwen25vl", "teacher", tok, 0, a.pretrain_data)
    build(Path(a.out) / "tiny-qwen2vl-student", "student", tok, 1, a.pretrain_data)
    json.dump({"note": "random weights, smoke test only"}, open(Path(a.out) / "README.json", "w"))
