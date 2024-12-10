import torch

from dataclasses import dataclass
from typing import Optional
from einops import rearrange, repeat

from flash_attn.bert_padding import pad_input, unpad_input

def round_up(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple

@dataclass
class TestTensors:
    """Tensors for benchmarking attention implementations."""
    q: torch.Tensor
    "q shape: (batch_size, seqlen, nheads, head_dim)"
    k: torch.Tensor
    "k shape: (batch_size, seqlen, nheads, head_dim)"
    v: torch.Tensor
    "v shape: (batch_size, seqlen, nheads, head_dim)"
    q_unpad: torch.Tensor
    "q_unpad shape: (total_q, nheads, head_dim)"
    k_unpad: torch.Tensor
    "k_unpad shape: (total_k, nheads, head_dim)"
    v_unpad: torch.Tensor
    "v_unpad shape: (total_k, nheads, head_dim)"
    cu_seqlens_q: torch.Tensor
    "cu_seqlens_q shape: (batch_size + 1)"
    cu_seqlens_k: torch.Tensor
    "cu_seqlens_k shape: (batch_size + 1)"
    seqused_q: torch.Tensor
    "seqused_q shape: (batch_size)"
    seqused_k: torch.Tensor
    "seqused_k shape: (batch_size)"
    max_seqlen_q: int
    max_seqlen_k: int
    query_padding_mask: torch.Tensor
    "query_padding_mask shape: (batch_size, seqlen_q)"
    key_padding_mask: torch.Tensor
    "key_padding_mask shape: (batch_size, seqlen_kv)"
    indices_q: torch.Tensor
    "indices shape: (total_nnz)"
    
    ### For FP8 ###
    descale_q: Optional[torch.Tensor] = None
    "descale_q shape: (1, 1, 1, 1), for FP8"
    descale_k: Optional[torch.Tensor] = None
    "descale_k shape: (1, 1, 1, 1), for FP8"
    descale_v: Optional[torch.Tensor] = None
    "descale_v shape: (1, 1, 1, 1), for FP8"
    descale_s: Optional[torch.Tensor] = None
    "descale_s shape: (1, 1, 1, 1), for FP8"
    scale_s: Optional[torch.Tensor] = None
    "scale_s shape: (1, 1, 1, 1), for FP8"
    scale_o: Optional[torch.Tensor] = None
    "scale_o shape: (1, 1, 1, 1), for FP8"
    
    ### For Paged Attention ###
    k_paged: Optional[torch.Tensor] = None
    "k_paged shape: (total_pages, page_size, nheads_kv, head_dim)"
    v_paged: Optional[torch.Tensor] = None
    "v_paged shape: (total_pages, page_size, nheads_kv, head_dim)"
    page_table: Optional[torch.Tensor] = None
    "page_table shape: (batch_size, max_pages_per_batch)"
    
    @property
    def batch_size(self) -> int:
        return self.q.shape[0]
    
    @property
    def softmax_scale(self) -> torch.Tensor:
        return self.q.shape[-1] ** (-0.5)
    
    @staticmethod 
    def generate(
        dtype: torch.dtype,
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_kv: int,
        nheads_q: int,
        nheads_kv: int,
        headdim: int,
        device: str,
        query_padding_mask_mode: str="full", # opt "full", "random", "third"
        key_padding_mask_mode: str="full", # opt "full", "random", "third"
        page_size: Optional[int]=None,
        paged_kv_cache_size: Optional[int]=None,
        page_order: str = "sequential", # opt "sequential", "random", "reverse"
    ) -> 'TestTensors':
        rand_dtype = torch.float16 if dtype == torch.float8_e4m3fn else dtype
        common_args = dict(
            device=device, dtype=rand_dtype, requires_grad=True)
        q_shape = (batch_size, max_seqlen_q, nheads_q, headdim)
        kv_shape = (batch_size, max_seqlen_kv, nheads_kv, headdim)
        q = torch.randn(*q_shape, **common_args).to(dtype) # type: ignore
        k = torch.randn(*kv_shape, **common_args).to(dtype) # type: ignore
        v = torch.randn(*kv_shape, **common_args).to(dtype) # type: ignore
        
        def generate_padding_mask(
            max_seqlen, batch_size, device, mode, zero_lengths=False
        ) -> torch.Tensor:
            assert mode in ["full", "random", "third"]
            if mode == "full":
                lengths = torch.full(
                    (batch_size, 1), max_seqlen, 
                    device=device, dtype=torch.int32)
            elif mode == "random":
                lengths = torch.randint(
                    max(0 if zero_lengths else 1, max_seqlen - 20), 
                    max_seqlen + 1, (batch_size, 1), device=device
                )
            elif mode == "third":
                lengths = torch.randint(max_seqlen // 3, max_seqlen + 1, 
                                        (batch_size, 1), device=device)

            if zero_lengths:
                # Generate zero-lengths every 5 batches and the last batch.
                for i in range(batch_size):
                    if i % 5 == 0:
                        lengths[i] = 0
                lengths[-1] = 0
        
            return repeat(
                torch.arange(max_seqlen, device=device), 
                "s -> b s", b=batch_size) < lengths

        query_padding_mask = generate_padding_mask(
            max_seqlen_q, batch_size, device, query_padding_mask_mode)
        key_padding_mask = generate_padding_mask(
            max_seqlen_kv, batch_size, device, key_padding_mask_mode)

        def unpad_input_(input, padding_mask):
            if input.dtype == torch.float8_e4m3fn:
                input_unpad, indices, cu_seqlens, max_seqlen, seqused =\
                    unpad_input(input.view(dtype=torch.int8), padding_mask)
                return input_unpad.view(dtype=input.dtype), indices, cu_seqlens, max_seqlen, seqused
            else:
                return unpad_input(input, padding_mask)

        q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, seqused_q =\
            unpad_input_(q, query_padding_mask)
        k_unpad, _, cu_seqlens_k, max_seqlen_k, seqused_k =\
            unpad_input_(k, key_padding_mask)
        v_unpad, _, _, _, _ = unpad_input_(v, key_padding_mask)

        default_scale = torch.ones(
            1, 1, 1, 1, dtype=torch.float32, device=device)
        
        if page_size:
            paged_kv_cache_size = paged_kv_cache_size or k.shape[1]
            paged_kv_cache_size = round_up(paged_kv_cache_size, page_size)
            assert paged_kv_cache_size is not None
            
            k_padded = torch.zeros(
                k.shape[0], paged_kv_cache_size, *k.shape[2:],  
                device=device, dtype=k.dtype, requires_grad=False).detach()
            v_padded = torch.zeros(
                v.shape[0], paged_kv_cache_size, *v.shape[2:],
                device=device, dtype=v.dtype, requires_grad=False).detach()
            k_padded[:, :max_seqlen_kv, ...] = k
            v_padded[:, :max_seqlen_kv, ...] = v
            k_paged, v_paged = [
                rearrange(x, "b (n p) h d -> (b n) p h d", p=page_size) 
                for x in [k_padded, v_padded]]
            total_pages = k_paged.shape[0]
            
            k_paged = k_paged.detach().requires_grad_(False)
            v_paged = v_paged.detach().requires_grad_(False)
            
            page_table = rearrange(
                torch.arange(total_pages, device=device, dtype=torch.int32),
                "(b s) -> b s", s=paged_kv_cache_size // page_size)
            
            if page_order == "reverse":
                perm = torch.arange(total_pages - 1, -1, -1, device=device)
            if page_order == "random":
                perm = torch.randperm(total_pages, device=device)
            if page_order in ["reverse", "random"]:
                if k_paged.dtype == torch.float8_e4m3fn:
                    assert v_paged.dtype == torch.float8_e4m3fn
                    k_paged = k_paged.view(dtype=torch.uint8)[perm, ...].view(dtype=k_padded.dtype)
                    v_paged = v_paged.view(dtype=torch.uint8)[perm, ...].view(dtype=v_padded.dtype)
                else:
                    k_paged = k_paged[perm, ...]
                    v_paged = v_paged[perm, ...]
                inv_perm = torch.argsort(perm)
                page_table_flat = page_table.flatten()
                page_table_flat = inv_perm[page_table_flat].to(torch.int32)
                page_table = page_table_flat.reshape(page_table.shape)
        else:
            k_paged = None
            v_paged = None
            page_table = None

        return TestTensors(
            q,
            k,
            v,
            q_unpad.detach(),
            k_unpad.detach(),
            v_unpad.detach(),
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            max_seqlen_q,
            max_seqlen_k,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            indices_q=indices_q,
            descale_q=default_scale.clone(),
            descale_k=default_scale.clone(),
            descale_v=default_scale.clone(),
            descale_s=default_scale.clone(),
            scale_s=default_scale.clone(),
            scale_o=default_scale.clone(),
            k_paged=k_paged,
            v_paged=v_paged,
            page_table=page_table,
        )
