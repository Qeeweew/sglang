"""Public operator API. Class imports are lazy; package import has no device effects."""

from importlib import import_module

_EXPORTS = {
    "RMSNorm": "elementwise",
    "ClippedSwiGLU": "elementwise",
    "PackedClippedSwiGLU": "elementwise",
    "WindowAttention": "attention",
    "SparseAttention": "attention",
    "DraftAttention": "attention",
    "DenseProjection": "projection",
    "WoAProjection": "projection",
    "repack_linear": "projection",
    "repack_wo_a": "projection",
    "QLIPlan": "qli",
    "PositionQuant": "position_quant",
    "RopeTables": "position_quant",
    "ProjectedCompressor": "compressor",
    "CompressorState": "compressor",
    "Provisional": "compressor",
    "MHC": "mhc",
    "Triplet": "mhc",
    "Workspace": "mhc",
    "PreResult": "mhc",
    "EngramTable": "engram",
    "HostBuffer": "engram",
    "EngramOps": "engram_ops",
    "HashParameters": "engram_ops",
    "FP8Weight": "fp8_weight",
    "Router": "router",
    "CED": "ced",
    "TailMetadata": "ced",
    "tail_capacity": "ced",
    "DSparkSchedule": "dspark_schedule",
    "DraftPageMetadata": "page_metadata",
    "MXFP4LocalExperts": "quant_grouped_gemm",
    "MoERoutes": "moe_routes",
    "MXFP4WeightPacker": "mxfp4_weight",
    "NativeGroupedGemm": "quant_grouped_gemm",
    "W8A16": "w8a16",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value
