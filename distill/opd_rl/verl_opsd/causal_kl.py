"""
causal_kl.py — OPSD-inspired causal KL computation on INSIGHT tokens.

Computes KL(teacher || student) on insight token positions (Phase 4).
Teacher = same model with correct answer + teacher insight prepended as privileged info.

Key difference from previous version:
- KL is computed on Phase 4 (insight) tokens, NOT Phase 5 (answer) tokens
- Teacher prompt includes teacher's consensus_insight as reference, not just answer letter
- This measures: "Does knowing the correct answer + teacher insight change the model's insight generation?"

If insight quality is high → teacher and student insight distributions similar → low KL → high reward
If insight quality is low → knowing answer significantly changes insight → high KL → low reward
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from typing import List, Tuple


def find_insight_positions(input_ids, response_ids, tokenizer, batch_size=None):
    """Find token positions of Phase 4 (insight) in each response.

    Returns: (batch, response_length) mask where 1 = insight token, 0 = non-insight
    """
    if batch_size is None:
        batch_size = input_ids.size(0)
    response_length = response_ids.size(1)

    insight_masks = torch.zeros(batch_size, response_length, dtype=torch.long,
                                 device=response_ids.device)

    for i in range(batch_size):
        rids = response_ids[i]

        # Actual (non-padded) response length: responses are right-padded, so
        # char-ratio positions must be mapped onto the real token count, not
        # the padded tensor width (otherwise the mask lands in padding).
        pad_id = getattr(tokenizer, "pad_token_id", None)
        actual_len = response_length
        if pad_id is not None:
            actual_len = int((rids != pad_id).sum().item())
        if pad_id != 0:
            actual_len = min(actual_len, int((rids != 0).sum().item()))
        actual_len = max(1, min(actual_len, response_length))

        resp_text = tokenizer.decode(rids, skip_special_tokens=True)

        # Find Phase 4 start
        phase4_idx = resp_text.find("Phase 4")
        if phase4_idx < 0:
            # No Phase 4 found, use middle 40% as insight
            start = int(actual_len * 0.3)
            end = int(actual_len * 0.7)
            insight_masks[i, start:end] = 1
            continue

        # Find Phase 5 start (end of insight)
        phase5_idx = resp_text.find("Phase 5", phase4_idx)
        if phase5_idx < 0:
            phase5_idx = len(resp_text)

        # Convert character positions to approximate token positions
        char_ratio_start = phase4_idx / max(len(resp_text), 1)
        char_ratio_end = phase5_idx / max(len(resp_text), 1)

        token_start = int(char_ratio_start * actual_len)
        token_end = int(char_ratio_end * actual_len)

        insight_masks[i, token_start:token_end] = 1

    return insight_masks


def construct_teacher_inputs(input_ids, attention_mask, position_ids,
                              ground_truths, teacher_insights, tokenizer, response_length,
                              options_list=None):
    """Construct teacher inputs with correct answer + option content + teacher insight prepended.

    Teacher input: [prompt] + "The correct answer is: {gt} ({content}). Based on this answer, what insight most directly leads to this conclusion? Reference insight: {insight}" + [response]
    """
    batch_size = input_ids.size(0)

    teacher_ids_list = []
    teacher_mask_list = []

    for i in range(batch_size):
        gt = ground_truths[i]
        insight = teacher_insights[i] if i < len(teacher_insights) else ""

        # Truncate teacher insight to avoid overly long prompts
        if len(insight) > 500:
            insight = insight[:500]

        # Get option content for the correct answer
        option_content = ""
        if options_list and i < len(options_list):
            opts = options_list[i]
            if isinstance(opts, dict):
                option_content = opts.get(gt, "")
            elif isinstance(opts, str):
                try:
                    opts_dict = json.loads(opts)
                    option_content = opts_dict.get(gt, "")
                except Exception:
                    pass

        if option_content:
            hint_text = f"The correct answer is: {gt} ({option_content}). Based on this answer, what insight most directly leads to this conclusion? Reference insight: {insight}. Now provide your own analysis. "
        else:
            hint_text = f"The correct answer is: {gt}. Based on this answer, what insight most directly leads to this conclusion? Reference insight: {insight}. Now provide your own analysis. "
        hint_ids = tokenizer.encode(hint_text, add_special_tokens=False)

        orig_ids = input_ids[i]
        orig_mask = attention_mask[i]
        prompt_len = orig_ids.size(0) - response_length

        prompt_ids = orig_ids[:prompt_len]
        response_ids = orig_ids[prompt_len:]
        prompt_mask = orig_mask[:prompt_len]
        response_mask = orig_mask[prompt_len:]

        hint_tensor = torch.tensor(hint_ids, dtype=orig_ids.dtype, device=orig_ids.device)
        hint_mask = torch.ones_like(hint_tensor)

        t_ids = torch.cat([prompt_ids, hint_tensor, response_ids])
        t_mask = torch.cat([prompt_mask, hint_mask, response_mask])

        teacher_ids_list.append(t_ids)
        teacher_mask_list.append(t_mask)

    # Left-pad to same length
    max_len = max(t.size(0) for t in teacher_ids_list)
    pad_id = tokenizer.pad_token_id or 0

    padded_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long,
                            device=input_ids.device)
    padded_mask = torch.zeros((batch_size, max_len), dtype=torch.long,
                              device=input_ids.device)

    for i in range(batch_size):
        t_len = teacher_ids_list[i].size(0)
        padded_ids[i, -t_len:] = teacher_ids_list[i]
        padded_mask[i, -t_len:] = teacher_mask_list[i]

    # Position IDs from attention mask
    t_pos = padded_mask.cumsum(-1) - 1
    t_pos = t_pos.clamp(min=0)

    # Handle mrope (3D position_ids for qwen2vl/qwen3vl)
    if position_ids.dim() == 3:
        t_pos = t_pos.unsqueeze(1).expand(-1, position_ids.size(1), -1)

    return padded_ids, padded_mask, t_pos


def extract_ground_truths(data):
    """Extract ground truth answers from DataProto."""
    ground_truths = []
    for ei in data.non_tensor_batch.get("extra_info", []):
        if isinstance(ei, str):
            ei = json.loads(ei)
        gt = ei.get("ground_truth", "") if isinstance(ei, dict) else ""
        ground_truths.append(str(gt).strip())
    return ground_truths


def extract_teacher_insights(data):
    """Extract teacher consensus_insight from DataProto."""
    teacher_insights = []
    for ei in data.non_tensor_batch.get("extra_info", []):
        if isinstance(ei, str):
            ei = json.loads(ei)
        insight = ei.get("teacher_insight", "") if isinstance(ei, dict) else ""
        teacher_insights.append(insight)
    return teacher_insights


def extract_options(data):
    """Extract options dict for each sample from DataProto extra_info."""
    options_list = []
    for ei in data.non_tensor_batch.get("extra_info", []):
        if isinstance(ei, str):
            ei = json.loads(ei)
        opts = ei.get("options", {}) if isinstance(ei, dict) else {}
        options_list.append(opts)
    return options_list


def compute_kl_per_token(student_log_probs, teacher_log_probs, insight_mask):
    """k1 per-token estimate: mean of (log pi_T(y_t) - log pi_S(y_t)) over insight tokens.
    """
    diff = teacher_log_probs - student_log_probs
    token_count = insight_mask.sum(dim=-1).clamp(min=1)
    return (diff * insight_mask).sum(dim=-1) / token_count


def compute_kl_full_vocab(student_logits, teacher_logits, insight_mask):
    """Full-vocab KL(teacher || student) on insight positions.
    """
    log_pt = F.log_softmax(teacher_logits.float(), dim=-1)
    log_ps = F.log_softmax(student_logits.float(), dim=-1)
    kl_per_pos = (log_pt.exp() * (log_pt - log_ps)).sum(dim=-1)
    token_count = insight_mask.sum(dim=-1).clamp(min=1)
    return (kl_per_pos * insight_mask).sum(dim=-1) / token_count


def compute_kl_topk(student_logits, teacher_logits, insight_mask, topk=64):
    """Optimized top-k KL(teacher || student) using logsumexp trick.
    """
    lse_t = torch.logsumexp(teacher_logits.float(), dim=-1)   # (B, T)
    lse_s = torch.logsumexp(student_logits.float(), dim=-1)   # (B, T)

    topk_logits_t, topk_idx = teacher_logits.float().topk(topk, dim=-1)  # (B, T, k)
    topk_log_pt = topk_logits_t - lse_t.unsqueeze(-1)
    topk_pt = topk_log_pt.exp()

    topk_logits_s = student_logits.float().gather(-1, topk_idx)  # (B, T, k)
    topk_log_ps = topk_logits_s - lse_s.unsqueeze(-1)

    kl_per_pos = (topk_pt * (topk_log_pt - topk_log_ps)).sum(dim=-1)
    token_count = insight_mask.sum(dim=-1).clamp(min=1)
    return (kl_per_pos * insight_mask).sum(dim=-1) / token_count


def compute_kl_bucket(student_logits, teacher_logits, insight_mask, topk=64):
    """Bucket KL(teacher || student): top-k tokens explicitly + tail as one bucket.
    """
    lse_t = torch.logsumexp(teacher_logits.float(), dim=-1)   # (B, T)
    lse_s = torch.logsumexp(student_logits.float(), dim=-1)   # (B, T)

    topk_logits_t, topk_idx = teacher_logits.float().topk(topk, dim=-1)  # (B, T, k)
    topk_log_pt = topk_logits_t - lse_t.unsqueeze(-1)
    topk_pt = topk_log_pt.exp()

    topk_logits_s = student_logits.float().gather(-1, topk_idx)  # (B, T, k)
    topk_log_ps = topk_logits_s - lse_s.unsqueeze(-1)
    topk_qs = topk_log_ps.exp()

    kl_topk = (topk_pt * (topk_log_pt - topk_log_ps)).sum(dim=-1)  # (B, T)

    p_R = (1.0 - topk_pt.sum(dim=-1)).clamp(min=1e-8)  # (B, T)
    q_R = (1.0 - topk_qs.sum(dim=-1)).clamp(min=1e-8)  # (B, T)

    log_ratio = torch.log(p_R / q_R)
    kl_bucket = p_R * log_ratio  # (B, T)

    kl_per_pos = kl_topk + kl_bucket
    token_count = insight_mask.sum(dim=-1).clamp(min=1)
    return (kl_per_pos * insight_mask).sum(dim=-1) / token_count


def compute_kl_full_iv(student_logits, teacher_logits, insight_mask):
    """Full-vocab reverse KL(student || teacher) on insight positions.
    """
    log_ps = F.log_softmax(student_logits.float(), dim=-1)
    log_pt = F.log_softmax(teacher_logits.float(), dim=-1)
    kl_per_pos = (log_ps.exp() * (log_ps - log_pt)).sum(dim=-1)
    token_count = insight_mask.sum(dim=-1).clamp(min=1)
    return (kl_per_pos * insight_mask).sum(dim=-1) / token_count


def compute_kl_bucket_iv(student_logits, teacher_logits, insight_mask, topk=64):
    """Reverse bucket KL(student || teacher): student's top-k + tail bucket.
    """
    lse_s = torch.logsumexp(student_logits.float(), dim=-1)   # (B, T)
    lse_t = torch.logsumexp(teacher_logits.float(), dim=-1)   # (B, T)

    topk_logits_s, topk_idx = student_logits.float().topk(topk, dim=-1)  # (B, T, k)
    topk_log_qs = topk_logits_s - lse_s.unsqueeze(-1)
    topk_qs = topk_log_qs.exp()

    topk_logits_t = teacher_logits.float().gather(-1, topk_idx)  # (B, T, k)
    topk_log_pt = topk_logits_t - lse_t.unsqueeze(-1)
    topk_ps = topk_log_pt.exp()

    kl_topk = (topk_qs * (topk_log_qs - topk_log_pt)).sum(dim=-1)  # (B, T)

    q_R = (1.0 - topk_qs.sum(dim=-1)).clamp(min=1e-8)  # (B, T)
    p_R = (1.0 - topk_ps.sum(dim=-1)).clamp(min=1e-8)  # (B, T)

    log_ratio = torch.log(q_R / p_R)
    kl_bucket = q_R * log_ratio  # (B, T)

    kl_per_pos = kl_topk + kl_bucket
    token_count = insight_mask.sum(dim=-1).clamp(min=1)
    return (kl_per_pos * insight_mask).sum(dim=-1) / token_count
