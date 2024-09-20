/***************************************************************************************************
 * Copyright (c) 2023 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cutlass/arch/mma.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/sm70_mma_twostage.hpp"

#include "cute/atom/mma_atom.hpp"
#include "cute/atom/copy_atom.hpp"
/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {

/////////////////////////////////////////////////////////////////////////////////////////////////


///////////////////////////////////////////////////////////////////////////////
//config A和B的copy范式
namespace detail {


using TiledMma_16x16x16 = TiledMMA<MMA_Atom<GFX928_16x16x16_F32F16F16F32_NT>,
                                  Layout<Shape<_2,_2,_1>>,
                                  Layout<Shape<_1,_1,_1>>>; 
                                  
using TiledMma_32x32x16 = TiledMMA<MMA_Atom<GFX928_32x32x16_F32F16F16F32_NT_ALT>,
                                  Layout<Shape<_2,_2,_1>>, 
                                  Layout<Shape<_1,_1,_1>>>; 

//to do根据shape进行选择tile mma的大小设置
template <
  class ElementA,
  class ElementB,
  class ElementC,
  class TileShape_MNK,
  auto... Args                       
>
CUTE_HOST_DEVICE constexpr
auto
mmac_op_selector()
{
  static_assert(is_static<TileShape_MNK>::value, "TileShape_MNK must be static.");
  static_assert(rank(TileShape_MNK{}) == 3, "TileShape_MNK must be rank 3.");
  //最小的mnk中m应该为32
  static_assert(size<0>(TileShape_MNK{}) % 32 == 0, "Tile_M must be a multiple of 32.");
  auto Tile_N = size<1>(TileShape_MNK{});

  // FP16 accumulator
  if constexpr (is_same_v<ElementC, half_t>) {
    static_assert(is_same_v<ElementA, half_t>, "Element types for AB must be half if ElementC is half.");
    static_assert(is_same_v<ElementB, half_t>, "Element types for AB must be half if ElementC is half.");
    static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");

  }

  // FP32 accumulator
  else if constexpr (is_same_v<ElementC, float>) {

    // FP16 inputs
    if constexpr (is_same_v<ElementA, half_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
      if constexpr (Tile_N % 128 == 0) {
        return GFX928_16x16x16_F32F16F16F32_NT{};
        // return TiledMma_32x32x16;
      }
      else if constexpr (Tile_N % 32 == 0) {
        return GFX928_16x16x16_F32F16F16F32_NT{};
        // return TiledMma_16x16x16;
      }
      // else {
      //   static_assert(Tile_N % 32 == 0, "Tile_N must be a multiple of 8.");
      // }
    }

    // BF16 inputs
    else if constexpr (is_same_v<ElementA, bfloat16_t>) {
      static_assert(is_same_v<ElementA, ElementB>, "ElementA and ElementB must be the same type for this config.");
      static_assert(size<2>(TileShape_MNK{}) % 16 == 0, "Tile_K must be a multiple of 16.");
    }

    // TF32 inputs
    else if constexpr (is_same_v<ElementA, tfloat32_t>) {

    }
    else {
      static_assert(sizeof(ElementA) == 0, "No eligible GMMA operator for request configuration.");
    }
  }
  // Unknown accumulator type
  else {
    static_assert(sizeof(ElementC) == 0, "Unknown ElementC accumulator type.");
  }
}
//
// F16
//

// Generates the most efficient possible TiledCopy with cp  atom given a set of parameters.
template<int ThreadCount, class Element, int Alignment, class StrideType, class TileMN, class TileK>
constexpr auto
make_cp_gmem_tiled_copy() {
  using AlignmentType = cute::uint_byte_t<static_cast<int>(sizeof(Element)) * Alignment>;
  constexpr int TileSizeMN  = cute::size(TileMN{});
  constexpr int TileSizeK   = cute::size(TileK{});

  // Maximize the number of threads along the gmem major mode to promote coalesced reads
  // While making sure our thread layout tiles the threadblock tile evenly

  if constexpr (cutlass::gemm::detail::is_k_major<StrideType>()) {
    // K major thread layout for K major gmem
    constexpr int threads_major = TileSizeK   / Alignment;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeMN % threads_minor == 0));
    return make_tiled_copy(
      Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
      Layout<Shape <Int<threads_minor>,Int<threads_major>>,
             Stride<Int<threads_major>,                _1>>{},
      Layout<Shape<_1,Int<Alignment>>>{});
  }
  else if constexpr (cutlass::gemm::detail::is_mn_major<StrideType>()) {
    // MN major thread layout for MN major gmem
    constexpr int threads_major = TileSizeMN  / Alignment;
    constexpr int threads_minor = ThreadCount / threads_major;
    static_assert(threads_major > 0);
    static_assert(ThreadCount % threads_major == 0);
    static_assert(threads_minor == 0 || (TileSizeK % threads_minor == 0));
    return make_tiled_copy(
        Copy_Atom<UniversalCopy<AlignmentType>, Element>{},
        Layout<Shape <Int<threads_major>,Int<threads_minor>>,
              Stride<                _1,Int<threads_major>>>{},
        Layout<Shape<Int<Alignment>,_1>>{});
  }
  else {
    static_assert(cute::is_void_v<Element>, "Unsupported gmem layout for automatic gmem tiled copy builder.");
  }
}

template <class StrideType, class ElementType, class BLK_MN, class BLK_K>
CUTE_HOST_DEVICE constexpr
auto
tiled_smem_selector()
{
  auto BLK_MN0 = size<0>(BLK_MN{});
  auto BLK_K0  = size<0>(BLK_K{});

  static_assert(BLK_MN0 % 16 == 0, "BLK_MN0 must be a multiple of 8.");
  static_assert(BLK_K0 % 16 == 0,  "BLK_K0 must be a multiple of 8.");

  if constexpr (cutlass::gemm::detail::is_mn_major<StrideType>()) {
    // if constexpr (BLK_MN0 % size<0>(GMMA::Layout_MN_SW128_Atom<ElementType>{}) == 0) {
    //   return GMMA::Layout_MN_SW128_Atom<ElementType>{};
    // }
    return composition(Swizzle<3,3,3>{},Layout<Shape < _64,_16>,Stride<_1, _64>>{});
  }
  else if constexpr (cutlass::gemm::detail::is_k_major<StrideType>()) {
    return composition(Swizzle<3,3,3>{},Layout<Shape < _64,_16>,Stride<_1, _64>>{});
  }
}
}




/////////////////////////////////////////////////////////////////////////////////////////////////

// MainloopSm70TwoStage
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm75,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<
      cute::is_same_v<KernelScheduleType, KernelMultistage>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  using TileShape = TileShape_MNK;

  using DispatchPolicy = MainloopSm70TwoStage;

  using AtomLayoutMNK = Layout<Shape<_2,_2,_1>>;

  // using TiledMma = detail::TiledMma_16x16x16;
  using TiledMma = decltype(cute::make_tiled_mma(detail::mmac_op_selector<
      ElementA, ElementB, ElementAccumulator, TileShape_MNK>(), AtomLayoutMNK{}));


  // A
  using GmemTiledCopyA = decltype(detail::make_cp_gmem_tiled_copy<
      256, ElementA, AlignmentA, TagToStrideA_t<GmemLayoutA>,decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());
  
  // B
  using GmemTiledCopyB = decltype(detail::make_cp_gmem_tiled_copy<
      256, ElementB, AlignmentB, TagToStrideB_t<GmemLayoutB>,decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomA = decltype(detail::tiled_smem_selector<
      TagToStrideA_t<GmemLayoutA>, ElementA, decltype(cute::get<0>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  using SmemLayoutAtomB = decltype(detail::tiled_smem_selector<
      TagToStrideB_t<GmemLayoutB>, ElementB, decltype(cute::get<1>(TileShape_MNK{})), decltype(cute::get<2>(TileShape_MNK{}))>());

  // GFX928_DS_READ_DS_M32x16_B16_ALT only support M/N major
  using SmemCopyAtomA = Copy_Atom<cute::DefaultCopy, ElementA>;

  using SmemCopyAtomB = Copy_Atom<cute::DefaultCopy, ElementB>;
  
  // Mainloop
  using CollectiveOp = collective::CollectiveMma<
      MainloopSm70TwoStage, TileShape,
      half_t, 
      TagToStrideA_t<GmemLayoutA>,
      half_t, 
      TagToStrideB_t<GmemLayoutB>,
      TiledMma,
      GmemTiledCopyA, 
      SmemLayoutAtomA, 
      SmemCopyAtomA, 
      cute::identity,  // A
      GmemTiledCopyB, 
      SmemLayoutAtomB, 
      SmemCopyAtomB, 
      cute::identity   // B
    >;
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// 目前默认走一个 multistage 后续直接走sm70的two stage pipline 
// GMMA auto kernel schedule
template <
  class ElementA,
  class GmemLayoutA,
  int AlignmentA,
  class ElementB,
  class GmemLayoutB,
  int AlignmentB,
  class ElementAccumulator,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class StageCountType,
  class KernelScheduleType
>
struct CollectiveBuilder<
    arch::Sm75,
    arch::OpClassTensorOp,
    ElementA,
    GmemLayoutA,
    AlignmentA,
    ElementB,
    GmemLayoutB,
    AlignmentB,
    ElementAccumulator,
    TileShape_MNK,
    ClusterShape_MNK,
    StageCountType,
    KernelScheduleType,
    cute::enable_if_t<cute::is_same_v<KernelScheduleType, KernelScheduleAuto>>
> {
  static_assert(is_static<TileShape_MNK>::value);
  static_assert(is_static<ClusterShape_MNK>::value);

  using CollectiveOp = typename CollectiveBuilder<
      arch::Sm75,
      arch::OpClassTensorOp,
      ElementA,
      GmemLayoutA,
      AlignmentA,
      ElementB,
      GmemLayoutB,
      AlignmentB,
      ElementAccumulator,
      TileShape_MNK,
      ClusterShape_MNK,
      StageCountType,
      KernelMultistage
    >::CollectiveOp;
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
