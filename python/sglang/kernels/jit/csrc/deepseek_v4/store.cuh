#include <sgl_kernel/tensor.h>
#include <sgl_kernel/utils.h>

#include <sgl_kernel/math.cuh>
#include <sgl_kernel/type.cuh>
#include <sgl_kernel/utils.cuh>
#include <sgl_kernel/vec.cuh>
#include <sgl_kernel/warp.cuh>

#include <sgl_kernel/deepseek_v4/fp8_utils.cuh>

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <bit>
#include <cstdint>
#include <cuda_fp8.h>

namespace {

using deepseek_v4::fp8::cast_to_ue8m0;
using deepseek_v4::fp8::inv_scale_ue8m0;
using deepseek_v4::fp8::pack_fp8;

SGL_DEVICE uint8_t quant_int4_symmetric(float x, float inv_scale) {
  const float scaled = x * inv_scale;
  int32_t q = scaled >= 0.0f ? static_cast<int32_t>(floorf(scaled + 0.5f))
                             : static_cast<int32_t>(ceilf(scaled - 0.5f));
  q = q < -7 ? -7 : (q > 7 ? 7 : q);
  return static_cast<uint8_t>(q) & 0x0f;
}

SGL_DEVICE uint8_t pack_int4_symmetric(float x, float y, float inv_scale) {
  return quant_int4_symmetric(x, inv_scale) | (quant_int4_symmetric(y, inv_scale) << 4);
}

struct FusedStoreCacheParam {
  const void* __restrict__ input;
  void* __restrict__ cache;
  const void* __restrict__ indices;
  uint32_t num_tokens;
};

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL, bool kInt4Store>
__global__ void fused_store_flashmla_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 584 = 576 + 8
  constexpr int64_t kPageBytes =
      kInt4Store ? (368ll << kPageBits) : host::div_ceil(584 << kPageBits, 576) * 576;

  // each warp handles 64 elements, 8 warps, each block handles 1 row
  const auto& [input, cache, indices, num_tokens] = param;
  const uint32_t bid = blockIdx.x;
  const uint32_t tid = threadIdx.x;
  const uint32_t wid = tid / 32;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[bid];
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  const auto elems = static_cast<const Float2*>(input)[tid + bid * 256];
  if (index < 0) return;
  if (wid != 7) {
    const auto [x, y] = cast<fp32x2_t>(elems);
    const auto abs_max = warp::reduce_max(fmaxf(fabs(x), fabs(y)));
    const int32_t page = index >> kPageBits;
    const int32_t offset = index & ((1 << kPageBits) - 1);
    const auto page_ptr = pointer::offset(cache, page * kPageBytes);
    if constexpr (kInt4Store) {
      const auto scale_bf16 = cast<bf16_t>(abs_max == 0.0f ? 1.0f : abs_max / 7.0f);
      const auto inv_scale = 1.0f / cast<float>(scale_bf16);
      const auto token_ptr = pointer::offset(page_ptr, offset * 368);
      static_cast<uint8_t*>(token_ptr)[tid] = pack_int4_symmetric(x, y, inv_scale);
      if ((tid & 31) == 0)
        reinterpret_cast<bf16_t*>(static_cast<uint8_t*>(token_ptr) + 352)[wid] = scale_bf16;
    } else {
      const auto scale_raw = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
      const auto scale_ue8m0 = cast_to_ue8m0(scale_raw);
      const auto inv_scale = inv_scale_ue8m0(scale_ue8m0);
      const auto result = pack_fp8(x * inv_scale, y * inv_scale);
      const auto value_ptr = pointer::offset(page_ptr, offset * 576);
      const auto scale_ptr = pointer::offset(page_ptr, 576 << kPageBits, offset * 8);
      static_cast<fp8x2_e4m3_t*>(value_ptr)[tid] = result;
      static_cast<uint8_t*>(scale_ptr)[wid] = scale_ue8m0;
    }
  } else {
    const auto result = cast<bf16x2_t>(elems);
    const int32_t page = index >> kPageBits;
    const int32_t offset = index & ((1 << kPageBits) - 1);
    const auto page_ptr = pointer::offset(cache, page * kPageBytes);
    const auto value_ptr = kInt4Store ? pointer::offset(page_ptr, offset * 368, 224)
                                     : pointer::offset(page_ptr, offset * 576, 448);
    static_cast<bf16x2_t*>(value_ptr)[tid - 7 * 32] = result;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageBits, bool kUsePDL, bool kInt4Store>
__global__ void fused_store_indexer_cache(const __grid_constant__ FusedStoreCacheParam param) {
  using namespace device;

  /// NOTE: 132 = 128 + 4
  constexpr int64_t kPageBytes = (kInt4Store ? 72ll : 132ll) << kPageBits;

  // each warp handles 128 elements, 1 warp, each block handles multiple rows
  const auto& [input, cache, indices, num_tokens] = param;
  const auto global_tid = blockIdx.x * blockDim.x + threadIdx.x;
  const auto global_wid = global_tid / 32;
  const auto lane_id = threadIdx.x % 32;

  if (global_wid >= num_tokens) return;

  PDLWaitPrimary<kUsePDL>();

  // prefetch the index
  const auto index = static_cast<const IndicesT*>(indices)[global_wid];
  // always load the value from input (don't store if invalid)
  using Float2 = packed_t<Float>;
  using InStorage = AlignedVector<Float2, 2>;
  using OutStorage = AlignedVector<fp8x2_e4m3_t, 2>;
  const auto elems = static_cast<const InStorage*>(input)[global_tid];
  if (index < 0) return;
  const auto [x0, x1] = cast<fp32x2_t>(elems[0]);
  const auto [y0, y1] = cast<fp32x2_t>(elems[1]);
  const auto local_max = fmaxf(fmaxf(fabs(x0), fabs(x1)), fmaxf(fabs(y0), fabs(y1)));
  const auto abs_max = kInt4Store ? warp::reduce_max<8>(local_max) : warp::reduce_max(local_max);
  const int32_t page = index >> kPageBits;
  const int32_t offset = index & ((1 << kPageBits) - 1);
  const auto page_ptr = pointer::offset(cache, page * kPageBytes);
  if constexpr (kInt4Store) {
    const auto scale_bf16 = cast<bf16_t>(abs_max == 0.0f ? 1.0f : abs_max / 7.0f);
    const auto inv_scale = 1.0f / cast<float>(scale_bf16);
    const auto value_ptr = pointer::offset(page_ptr, offset * 64);
    const auto scale_ptr = pointer::offset(page_ptr, 64 << kPageBits, offset * 8);
    const uint16_t packed = static_cast<uint16_t>(pack_int4_symmetric(x0, x1, inv_scale)) |
                            (static_cast<uint16_t>(pack_int4_symmetric(y0, y1, inv_scale)) << 8);
    reinterpret_cast<uint16_t*>(value_ptr)[lane_id] = packed;
    if ((lane_id & 7) == 0) reinterpret_cast<bf16_t*>(scale_ptr)[lane_id >> 3] = scale_bf16;
  } else {
    // use normal fp32 scale
    const auto scale = fmaxf(1e-4f, abs_max) / kFP8E4M3Max;
    const auto inv_scale = 1.0f / scale;
    const auto value_ptr = pointer::offset(page_ptr, offset * 128);
    const auto scale_ptr = pointer::offset(page_ptr, 128 << kPageBits, offset * 4);
    OutStorage result;
    result[0] = pack_fp8(x0 * inv_scale, x1 * inv_scale);
    result[1] = pack_fp8(y0 * inv_scale, y1 * inv_scale);
    static_cast<OutStorage*>(value_ptr)[lane_id] = result;
    static_cast<float*>(scale_ptr)[0] = scale;
  }

  PDLTriggerSecondary<kUsePDL>();
}

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL, bool kInt4Store = false>
struct FusedStoreCacheFlashMLAKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes =
      kInt4Store ? 368 * kPageSize : host::div_ceil(584 * kPageSize, 576) * 576;
  static constexpr auto kernel = fused_store_flashmla_cache<Float, IndicesT, kLogSize, kUsePDL, kInt4Store>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 512})  // input
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 256;
    const auto num_blocks = num_tokens;
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

template <typename Float, typename IndicesT, uint32_t kPageSize, bool kUsePDL, bool kInt4Store = false>
struct FusedStoreCacheIndexerKernel {
  static constexpr int32_t kLogSize = std::countr_zero(kPageSize);
  static constexpr int64_t kPageBytes = (kInt4Store ? 72 : 132) * kPageSize;
  static constexpr auto kernel = fused_store_indexer_cache<Float, IndicesT, kLogSize, kUsePDL, kInt4Store>;

  static_assert(std::has_single_bit(kPageSize), "kPageSize must be a power of 2");
  static_assert(1 << kLogSize == kPageSize);

  static void run(tvm::ffi::TensorView input, tvm::ffi::TensorView cache, tvm::ffi::TensorView indices) {
    using namespace host;

    auto N = SymbolicSize{"num_tokens"};
    auto device_ = SymbolicDevice{};
    device_.set_options<kDLCUDA>();
    TensorMatcher({N, 128})  // input
        .with_dtype<Float>()
        .with_device(device_)
        .verify(input);
    TensorMatcher({-1, -1})  // cache
        .with_strides({kPageBytes, 1})
        .with_dtype<uint8_t>()
        .with_device(device_)
        .verify(cache);
    TensorMatcher({N})  // indices
        .with_dtype<IndicesT>()
        .with_device(device_)
        .verify(indices);
    const auto num_tokens = static_cast<uint32_t>(N.unwrap());
    const auto params = FusedStoreCacheParam{
        .input = input.data_ptr(),
        .cache = cache.data_ptr(),
        .indices = indices.data_ptr(),
        .num_tokens = num_tokens,
    };
    const auto kBlockSize = 128;
    const auto num_blocks = div_ceil(num_tokens * 32, kBlockSize);
    LaunchKernel(num_blocks, kBlockSize, device_.unwrap()).enable_pdl(kUsePDL)(kernel, params);
  }
};

}  // namespace
