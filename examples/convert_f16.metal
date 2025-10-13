#include <metal_stdlib>
using namespace metal;

kernel void convert_f32_to_f16(
    device const float*      in  [[buffer(0)]],
    device half*             out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = half(in[gid]);
}

kernel void convert_f16_to_f32(
    device const half*       in  [[buffer(0)]],
    device float*            out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    out[gid] = float(in[gid]);
}

