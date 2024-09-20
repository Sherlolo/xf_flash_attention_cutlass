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

#include <cute/arch/mma_gfx928.hpp>

#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>

namespace cute
{
/////////////////////////////////////v_mmac_16x16x8_f32//////////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x8_F32F32F32F32_NT>
{
  using ElementDVal = float;
  using ElementAVal = float;
  using ElementBVal = float;
  using ElementCVal = float;

  using Shape_MNK = Shape<_16,_16,_8>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,_2>,
                  Stride<Stride<_1,_32>,_16>>;
  
  using BLayout = Layout<Shape <Shape <_16, _4>,_2>,
                         Stride<Stride< _1,_32>,_16>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};

/////////////////////////////////////v_mmac_f32_16x16x8_tf32//////////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x8_F32TF32TF32F32_NT>
     : MMA_Traits<GFX928_16x16x8_F32F32F32F32_NT>
{
  using ElementDVal = float;
  using ElementAVal = tfloat32_t;
  using ElementBVal = tfloat32_t;
  using ElementCVal = float;
};
/////////////////////////////////////v_mmac_f32_16x16x16_f16//////////////////////////////////////////

template <>
struct MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT>
{
  using ElementDVal = float;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = float;


  using Shape_MNK = Shape<_16,_16,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,_4>,
                          Stride<Stride<_1,_64>,_16>>;
  
  using BLayout = Layout<Shape <Shape <_16, _4>,_4>,
                         Stride<Stride< _1,_64>,_16>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};

template <>
struct MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT_FOR_GEMM1>
{
  using ElementDVal = float;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = float;


  using Shape_MNK = Shape<_16,_16,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
  
  using BLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};
template <>
struct MMA_Traits<GFX928_16x16x16_F32BF16BF16F32_NT_FOR_GEMM1>
      : MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT_FOR_GEMM1>
{
  using ElementDVal = float;
  using ElementAVal = bfloat16_t;
  using ElementBVal = bfloat16_t;
  using ElementCVal = float;
};

template <>
struct MMA_Traits<GFX928_32x32x16_F32F16F16F32_NT>
{
  using ElementDVal = float;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = float;


  using Shape_MNK = Shape<_32,_32,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,Shape<_2,_4>>,
                          Stride<Stride<_1,_128>,Stride<_16,_32>>>;
  
  using BLayout = Layout<Shape<Shape<_16, _4>,Shape<_2,_4>>,
                          Stride<Stride<_1,_128>,Stride<_16,_32>>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, Shape<Shape<_2,_2>,_4>>,
                         Stride<Stride< _2,_64>, Stride<Stride<_1,_32>,_256>>>;
};


template <>
struct MMA_Traits<GFX928_32x32x16_F32F16F16F32_NT_ALT>
{
  using ElementDVal = float;
  using ElementAVal = half_t;
  using ElementBVal = half_t;
  using ElementCVal = float;


  using Shape_MNK = Shape<_32,_32,_16>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,Shape<_2,_4>>,
                          Stride<Stride<_2,_128>,Stride<_1,_32>>>;
  
  using BLayout = Layout<Shape<Shape<_16, _4>,Shape<_2,_4>>,
                          Stride<Stride<_2,_128>,Stride<_1,_32>>>;
                          
  using CLayout = Layout<Shape <Shape <_16, _4>, Shape<_4,_2,_2>>,
                         Stride<Stride< _2,_64>, Stride<_256,_32,_1>>>;
};

/////////////////////////////////////v_mmac_f32_16x16x16_bf16//////////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x16_F32BF16BF16F32_NT>
      : MMA_Traits<GFX928_16x16x16_F32F16F16F32_NT>
{
  using ElementDVal = float;
  using ElementAVal = bfloat16_t;
  using ElementBVal = bfloat16_t;
  using ElementCVal = float;
};
////////////////////////////v_mmac_i32_16x16x32_i8///////////////////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x32_I32I8I8I32_NT>
{
  using ElementDVal = int32_t;
  using ElementAVal = int8_t;
  using ElementBVal = int8_t;
  using ElementCVal = int32_t;
  using Shape_MNK = Shape<_16,_16,_32>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape<Shape<_16, _4>,_8>,
                          Stride<Stride<_1,_128>,_16>>;
  
  using BLayout = Layout<Shape <Shape <_16, _4>,_8>,
                         Stride<Stride< _1,_128>,_16>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};
///////////////////////////////////v_mmac_i32_16x16x32_u8////////////////////////////////////////////
template <>
struct MMA_Traits<GFX928_16x16x32_I32U8U8I32_NT>
      : MMA_Traits<GFX928_16x16x32_I32I8I8I32_NT> 
{
  using ElementDVal = int32_t;
  using ElementAVal = uint8_t;
  using ElementBVal = uint8_t;
  using ElementCVal = int32_t;
};
} // namespace cute
