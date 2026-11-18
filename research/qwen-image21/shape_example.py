"""Source-aligned shape arithmetic only; no tensors, weights or GPU allocation.

W x H convention. 2048 = effective non-image DiT tokens AFTER template/drop.
Image slot count assumes the matching processor, one slot per 2x2 VAE latents.
"""
import json
import math


def example():
    original_w, original_h = 1024, 1204
    resolution = 1024
    ratio = original_w / original_h
    w_unrounded = math.sqrt(resolution**2 * ratio)
    w = round(w_unrounded / 32) * 32
    h = round((w_unrounded / ratio) / 32) * 32
    ref_tokens = (h // 16) * (w // 16)
    text_tokens = 2048
    target_tokens = (1024 // 16)**2
    ref_slots = ref_tokens // 4
    vlm_length = text_tokens + ref_slots
    prefix = text_tokens + ref_tokens
    joint = prefix + target_tokens
    cache_tensor = [1, prefix, 32, 128]
    pair_bytes = 2 * math.prod(cache_tensor) * 2
    result = dict(
        assumptions='B=1; no TP; one reference; effective non-image tokens=2048; matching processor; BF16; constructor defaults 32 layers/32 heads/128 head dim',
        reference_original_wh=[original_w, original_h], reference_resized_wh=[w, h],
        reference_pixels=[1,4,1,h,w], reference_vae=[1,64,1,h//16,w//16],
        reference_flat=[1,ref_tokens,64], target_flat=[1,target_tokens,64],
        reference_slots=ref_slots, prompt_embeds=[1,vlm_length,4096],
        img_mask_with_target_slots=[1,vlm_length+target_tokens//4],
        expanded_joint=[1,joint,4096], prefix_tokens=prefix,
        prefill_qkv_each=[1,joint,32,128], cache_k_or_v=cache_tensor,
        cached_new_qkv_each=[1,target_tokens,32,128],
        cached_concat_kv_each=[1,joint,32,128],
        target_attention_scores_logical=[1,32,target_tokens,joint],
        target_prediction=[1,target_tokens,64],
        vae_decode_input=[1,64,1,64,64], decoded_rgba=[1,4,1,1024,1024],
        layer_cache_pair_mib=pair_bytes/2**20, all_layers_cache_gib=pair_bytes*32/2**30,
    )
    assert (w,h)==(960,1120)
    assert (ref_tokens,ref_slots,vlm_length,prefix,joint)==(4200,1050,3098,6248,10344)
    assert vlm_length + target_tokens//4 + 3*(ref_slots+target_tokens//4)==joint
    return result


if __name__=='__main__':
    print(json.dumps(example(),ensure_ascii=False,indent=2))
