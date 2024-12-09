import torch
import math

from dataclasses import dataclass
from einops import rearrange, repeat

from utils.test_tensors import TestTensors

import flash_attn_interface as flash_attn_3
import flash_attn.flash_attn_interface as flash_attn_2

@dataclass
class RunConfig:
    causal: bool
    dropout_p: float


def is_cudnn_available():
    try:
        import cudnn
        return True
    except ImportError:
        return False


def is_triton_available():
    try:
        import triton_fused_attention
        return True
    except ImportError:
        return False


ATTN_IMPL_FACTORIES = {}
def register_attn_factory(name, supported_dtypes, available=True):
    def decorator(fn):
        global ATTN_IMPL_FACTORIES
        if available:
            ATTN_IMPL_FACTORIES[name] = (fn, supported_dtypes)
        return fn
    return decorator


@register_attn_factory("Pytorch", (torch.float16, torch.bfloat16))
def create_attn_pytorch_fn(ts: TestTensors, c: RunConfig):
    def attention_pytorch(
        q,
        k,
        v,
        query_padding_mask=None,
        key_padding_mask=None,
        dropout_p=0.0,
        dropout_mask=None,
        causal=False,
        softcap=0.0,
        upcast=True,
        reorder_ops=False,
    ):
        """
        Arguments:
            q: (batch_size, seqlen_q, nheads, head_dim)
            k: (batch_size, seqlen_k, nheads_k, head_dim)
            v: (batch_size, seqlen_k, nheads_k, head_dim)
            query_padding_mask: (batch_size, seqlen_q)
            key_padding_mask: (batch_size, seqlen_k)
            dropout_p: float
            dropout_mask: (batch_size, nheads, seqlen_q, seqlen_k)
            causal: whether to apply causal masking
            upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
                output back to fp16/bf16.
        Output:
            output: (batch_size, seqlen_q, nheads, head_dim)
        """
        batch_size, seqlen_q, nheads_q, d = q.shape
        _, _, nheads_kv, _ = k.shape
        q = rearrange(q, 'b t h d -> (b h) t d')
        
        if nheads_q != nheads_kv:
            k = repeat(k, "b s h d -> (b h g) d s", g=nheads_q // nheads_kv)
            v = repeat(v, "b s h d -> b s (h g) d", g=nheads_q // nheads_kv)
        else:
            k = rearrange(k, 'b s h d -> (b h) d s')

        softmax_scale = 1.0 / math.sqrt(d)
        # Preallocate attn_weights for `baddbmm`
        scores = torch.empty(batch_size * nheads_q, seqlen_q, seqlen_q, 
                            dtype=q.dtype, device=q.device)
        scores = rearrange(torch.baddbmm(scores, q, k, beta=0, alpha=softmax_scale),
                        '(b h) t s -> b h t s', h=nheads_q)
        if causal:
            # "triu_tril_cuda_template" not implemented for 'BFloat16'
            # So we have to construct the mask in float
            causal_mask = torch.triu(torch.full((seqlen_q, seqlen_q), -10000.0, device=scores.device), 1)
            # TD [2022-09-30]: Adding is faster than masked_fill_ (idk why, just better kernel I guess)
            scores = scores + causal_mask.to(dtype=scores.dtype)
        attention = torch.softmax(scores, dim=-1)
        attention_drop = F.dropout(attention, dropout_p)
        output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
        return output.to(dtype=q.dtype)
    
    q, k, v = ts.q, ts.k, ts.v 
    if q.dtype not in [torch.float16, torch.bfloat16]:
        q, k, v = q.half(), k.half(), v.half()
    return lambda: attention_pytorch(
        q, k, v,
        query_padding_mask=ts.query_padding_mask,
        key_padding_mask=ts.key_padding_mask,
        dropout_p=c.dropout_p, causal=c.causal
    )


def _create_attn_flashX_fn(flash_module, ts: TestTensors, c: RunConfig):
    softmax_scale = ts.q.shape[-1] ** (-0.5)
    return lambda: flash_module._flash_attn_varlen_forward(
        ts.q, ts.k, ts.v, 
        softmax_scale=softmax_scale,
        causal=c.causal,
        window_size=(-1, -1))[0]


def _create_attn_flashX_kvcache_paged_fn(flash_module, ts: TestTensors, c: RunConfig):
    softmax_scale = ts.q.shape[-1] ** (-0.5)
    return lambda: flash_module.flash_attn_with_kvcache(
        ts.q, ts.k_paged, ts.v_paged, 
        softmax_scale=softmax_scale,
        page_table=ts.page_table,
        causal=c.causal,
        window_size=(-1, -1),
        num_splits=1)


def _create_attn_flashX_varlen_paged_fn(flash_module, ts: TestTensors, c: RunConfig):
    softmax_scale = ts.q.shape[-1] ** (-0.5)
    bs, seqlen_q, nheads, head_dim = ts.q.shape
    return lambda: flash_module._flash_attn_varlen_forward(
        ts.q.reshape(-1, nheads, head_dim),
        ts.k_paged,
        ts.v_paged, 
        cu_seqlens_q=ts.cu_seqlens_q,
        cu_seqlens_k=ts.cu_seqlens_k,
        max_seqlen_q=ts.max_seqlen_q,
        max_seqlen_k=ts.max_seqlen_k,
        block_table=ts.page_table,
        dropout_p=c.dropout_p,
        softmax_scale=softmax_scale,
        causal=c.causal)[0]\
            .reshape(bs, seqlen_q, nheads, head_dim)


def _create_attn_flashX_varlen_fn(flash_module, ts: TestTensors, c: RunConfig):
    softmax_scale = ts.q.shape[-1] ** (-0.5)
    bs, seqlen_q, nheads, head_dim = ts.q.shape
    return lambda: flash_module._flash_attn_varlen_forward(
        ts.q.reshape(-1, nheads, head_dim),
        ts.k.reshape(-1, nheads, head_dim),
        ts.v.reshape(-1, nheads, head_dim), 
        cu_seqlens_q=ts.cu_seqlens_q,
        cu_seqlens_k=ts.cu_seqlens_k,
        max_seqlen_q=ts.max_seqlen_q,
        max_seqlen_k=ts.max_seqlen_k,
        dropout_p=c.dropout_p,
        softmax_scale=softmax_scale,
        causal=c.causal)[0]\
            .reshape(bs, seqlen_q, nheads, head_dim)

@register_attn_factory("Flash2", (torch.float16, torch.bfloat16))
def create_flash2_attention_fn(ts: TestTensors, c: RunConfig):
    return _create_attn_flashX_fn(flash_attn_2, ts, c)

@register_attn_factory("Flash3", (torch.float16, torch.bfloat16))
def create_flash3_attention_fn(ts: TestTensors, c: RunConfig):
    return _create_attn_flashX_fn(flash_attn_3, ts, c)

@register_attn_factory("Flash3_kvcache_paged", (torch.float16, torch.bfloat16))
def create_attn_flash3_kvcache_paged_fn(ts: TestTensors, c: RunConfig):
    return _create_attn_flashX_kvcache_paged_fn(flash_attn_3, ts, c)

@register_attn_factory("Flash2_varlen_paged", (torch.float16, torch.bfloat16))
def create_attn_flash2_varlen_paged_fn(ts: TestTensors, c: RunConfig):
    return _create_attn_flashX_varlen_paged_fn(flash_attn_2, ts, c)

@register_attn_factory("Flash3_varlen_paged", (torch.float16, torch.bfloat16))
def create_attn_flash3_varlen_paged_fn(ts: TestTensors, c: RunConfig):
    return _create_attn_flashX_varlen_paged_fn(flash_attn_3, ts, c)

@register_attn_factory("Flash2_varlen", (torch.float16, torch.bfloat16))
def create_attn_flash2_varlen_fn(ts: TestTensors, c: RunConfig):
    return _create_attn_flashX_varlen_fn(flash_attn_2, ts, c)

@register_attn_factory("Flash3_varlen", 
                       (torch.float16, torch.bfloat16, torch.float8_e4m3fn))
def create_attn_flash3_varlen_fn(ts: TestTensors, c: RunConfig):
    return _create_attn_flashX_varlen_fn(flash_attn_3, ts, c)

@register_attn_factory("triton", (torch.float16, torch.bfloat16), 
                       is_triton_available())
def create_attn_triton_fn(ts: TestTensors, c: RunConfig):
    from triton_fused_attention import attention as attention_triton
    
    # TODO (Lucas): Figure out how to test this, i.d. where
    #   triton_fused_attention comes from
    scale = 1 / math.sqrt(ts.q.shape[-1])
    q_transposed = ts.q.transpose(1, 2).contiguous()
    k_transposed = ts.k.transpose(1, 2).contiguous()
    v_transposed = ts.v.transpose(1, 2).contiguous().permute(0, 1, 3, 2)
    fn = lambda: attention_triton(
        q_transposed,
        k_transposed,
        v_transposed.permute(0, 1, 3, 2),
        c.causal,
        scale,
    )
    fn() # Run once to compile
    return fn

@register_attn_factory(
    "cuDNN", (torch.float16, torch.bfloat16, torch.float8_e4m3fn), 
    is_cudnn_available())
def create_attn_cudnn_fn(ts: TestTensors, c: RunConfig):
    import cudnn
    
    def convert_to_cudnn_type(torch_type):
        mapping = {
            torch.float16: cudnn.data_type.HALF,
            torch.bfloat16: cudnn.data_type.BFLOAT16,
            torch.float32: cudnn.data_type.FLOAT,
            torch.int32: cudnn.data_type.INT32,
            torch.int64: cudnn.data_type.INT64,
            torch.float8_e4m3fn: cudnn.data_type.FP8_E4M3,
        }
        if torch_type in mapping:
            return mapping[torch_type]
        else:
            raise ValueError("Unsupported tensor data type.")


    
    if ts.q.dtype == torch.float8_e4m3fn:
        # "cat_cuda" not implemented for 'Float8_e4m3fn'
        qkv = torch.stack([ts.q.view(dtype=torch.uint8), 
                           ts.k.view(dtype=torch.uint8), 
                           ts.v.view(dtype=torch.uint8)], 
                          dim=2).view(dtype=torch.float8_e4m3fn)
    else: 
        qkv = torch.stack([ts.q, ts.k, ts.v], dim=2)
    seqlen_q = ts.q.shape[1]
    seqlen_k = ts.k.shape[1]
    b, _, _, nheads, headdim = qkv.shape
    assert cudnn is not None, 'CUDNN is not available'
    o_gpu = torch.zeros(b, seqlen_q, nheads, headdim, dtype=qkv.dtype, device=qkv.device)
    o_gpu_transposed = torch.as_strided(
        o_gpu,
        [b, nheads, seqlen_q, headdim],
        [nheads * seqlen_q * headdim, headdim, nheads * headdim, 1],
    )
    graph = cudnn.pygraph(
        io_data_type=convert_to_cudnn_type(qkv.dtype),
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    new_q = torch.as_strided(
        qkv,
        [b, nheads, seqlen_q, headdim],
        [seqlen_q * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=0,
    )
    q = graph.tensor(
        name="Q",
        dim=list(new_q.shape),
        stride=list(new_q.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_k = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim,
    )
    k = graph.tensor(
        name="K",
        dim=list(new_k.shape),
        stride=list(new_k.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )
    new_v = torch.as_strided(
        qkv,
        [b, nheads, seqlen_k, headdim],
        [seqlen_k * nheads * headdim * 3, headdim, headdim * nheads * 3, 1],
        storage_offset=nheads * headdim * 2,
    )
    v = graph.tensor(
        name="V",
        dim=list(new_v.shape),
        stride=list(new_v.stride()),
        data_type=convert_to_cudnn_type(qkv.dtype)
    )

    is_fp8 = qkv.dtype == torch.float8_e4m3fn

    variant_pack = {
        q: new_q,
        k: new_k,
        v: new_v,
    }

    common_sdpa_args = dict(
        q=q,
        k=k,
        v=v,
        is_inference=True,
        attn_scale=1.0 / math.sqrt(headdim),
        use_causal_mask=c.causal,
        name="sdpa",
    )

    if is_fp8:
        descale_q = graph.tensor(dim=[1,1,1,1], stride=[1,1,1,1], data_type=cudnn.data_type.FLOAT)
        descale_k = graph.tensor(dim=[1,1,1,1], stride=[1,1,1,1], data_type=cudnn.data_type.FLOAT)
        descale_v = graph.tensor(dim=[1,1,1,1], stride=[1,1,1,1], data_type=cudnn.data_type.FLOAT)
        descale_s = graph.tensor(dim=[1,1,1,1], stride=[1,1,1,1], data_type=cudnn.data_type.FLOAT)
        scale_s = graph.tensor(dim=[1,1,1,1], stride=[1,1,1,1], data_type=cudnn.data_type.FLOAT)
        scale_o = graph.tensor(dim=[1,1,1,1], stride=[1,1,1,1], data_type=cudnn.data_type.FLOAT)
        amax_s_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)
        amax_o_gpu = torch.empty(1, 1, 1, 1, dtype=torch.float32, device=qkv.device)

        o, _, amax_s, amax_o = graph.sdpa_fp8(
            **common_sdpa_args,
            descale_q=descale_q,
            descale_k=descale_k,
            descale_v=descale_v,
            descale_s=descale_s,
            scale_s=scale_s,
            scale_o=scale_o,
            use_padding_mask=False,
        )

        amax_s.set_output(False).set_dim(amax_s_gpu.shape).set_stride(amax_s_gpu.stride())
        amax_o.set_output(False).set_dim(amax_o_gpu.shape).set_stride(amax_o_gpu.stride())

        variant_pack.update({
            descale_q: ts.descale_q,
            descale_k: ts.descale_k,
            descale_v: ts.descale_v,
            descale_s: ts.descale_s,
            scale_s: ts.scale_s,
            scale_o: ts.scale_o,
            amax_o: amax_o_gpu,
            amax_s: amax_s_gpu,
        })
    else:
        o, _ = graph.sdpa(
            **common_sdpa_args,
        )

    o.set_output(True).set_dim(o_gpu_transposed.shape).set_stride(o_gpu_transposed.stride())
    variant_pack[o] = o_gpu_transposed

    graph.validate()
    graph.build_operation_graph()
    graph.create_execution_plans([cudnn.heur_mode.A, cudnn.heur_mode.FALLBACK])
    graph.check_support()
    graph.build_plans()

    workspace = torch.empty(graph.get_workspace_size(), device="cuda", dtype=torch.uint8)

    def run():
        graph.execute(variant_pack, workspace)
        return o_gpu

    return run
