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

#include <cute/arch/mma_gfx926.hpp>

#include <cute/atom/mma_traits.hpp>
#include <cute/layout.hpp>

namespace cute
{

template <>
struct MMA_Traits<GFX926_16x16x4_F32F32F32F32_NT>
{
  using ElementDVal = float;
  using ElementAVal = float;
  using ElementBVal = float;
  using ElementCVal = float;

  using Shape_MNK = Shape<_16,_16,_4>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape <Shape <_16, _4>,_1>,
                         Stride<Stride< _1,_16>,_1>>;
  using BLayout = Layout<Shape <Shape <_16, _4>,_1>,
                         Stride<Stride< _1,_16>,_1>>;
  using CLayout = Layout<Shape <Shape <_16, _4>, _4>,
                         Stride<Stride< _1,_16>,_64>>;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX926_32x32x4_F32F32F32F32_NT>
{
  static constexpr int vn = 2; 

  using ElementDVal = float;
  using ElementAVal = float;
  using ElementBVal = float;
  using ElementCVal = float;

  using Shape_MNK = Shape<Int<16*vn>,Int<16*vn>,_4>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape <Shape <    _16,        _4 >, Int<vn>>,
                         Stride<Stride<Int<vn>, Int<16*vn>>,      _1>>;
  using BLayout = Layout<Shape <Shape <    _16,        _4 >, Int<vn>>,
                         Stride<Stride<Int<vn>, Int<16*vn>>,      _1>>;
  using CLayout = Layout<Shape <Shape <    _16,            _4>, 
                                Shape <Shape <Int<vn>,    Int<vn>>,  _4>>,
                         Stride<Stride<Int<vn>, Int<16*vn*vn>>, 
                                Stride<Stride<     _1, Int<16*vn>>,Int<64*vn*vn>>>>;
};

///////////////////////////////////////////////////////////////////////////////

template <>
struct MMA_Traits<GFX926_64x64x4_F32F32F32F32_NT>
{
  static constexpr int vn = 4; 

  using ElementDVal = float;
  using ElementAVal = float;
  using ElementBVal = float;
  using ElementCVal = float;

  using Shape_MNK = Shape<Int<16*vn>,Int<16*vn>,_4>;
  using ThrID   = Layout<_64>;
  using ALayout = Layout<Shape <Shape <    _16,        _4 >, Int<vn>>,
                         Stride<Stride<Int<vn>, Int<16*vn>>,      _1>>;
  using BLayout = Layout<Shape <Shape <    _16,        _4 >, Int<vn>>,
                         Stride<Stride<Int<vn>, Int<16*vn>>,      _1>>;
  using CLayout = Layout<Shape <Shape <    _16,            _4>, 
                                Shape <Shape <Int<vn>,    Int<vn>>,            _4>>,
                         Stride<Stride<Int<vn>, Int<16*vn*vn>>, 
                                Stride<Stride<     _1, Int<16*vn>>,Int<64*vn*vn>>>>;
};

///////////////////////////////////////////////////////////////////////////////


} // namespace cute
