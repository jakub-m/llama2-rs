// vim: ft=c
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

kernel void convert_f32_to_bf16(
  device const float*      in  [[buffer(0)]],
  device ushort*           out [[buffer(1)]],
  uint gid [[thread_position_in_grid]]
) {
  // BF16 is essentially the upper 16 bits of FP32
  // Simple rounding: truncate the lower 16 bits
  uint bits = as_type<uint>(in[gid]);
  out[gid] = ushort(bits >> 16);
}

kernel void convert_f32_to_bf16_rne(
    device const float*      in  [[buffer(0)]],
    device ushort*           out [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
  // BF16 with round-to-nearest-even (RNE) rounding adds a rounding bias
  // before truncation to minimize rounding errors, which is generally
  // preferred for ML workloads.
  uint bits = as_type<uint>(in[gid]);
  uint rounding_bias = 0x7FFF + ((bits >> 16) & 1);
  out[gid] = ushort((bits + rounding_bias) >> 16);
}

kernel void convert_bf16_to_f32(
  device const ushort*     in  [[buffer(0)]],
  device float*            out [[buffer(1)]],
  uint gid [[thread_position_in_grid]]
) {
  // Convert BF16 back to FP32 by shifting to upper 16 bits
  uint bits = uint(in[gid]) << 16;
  out[gid] = as_type<float>(bits);
}


