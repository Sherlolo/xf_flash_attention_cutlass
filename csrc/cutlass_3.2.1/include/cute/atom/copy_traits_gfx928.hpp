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

#include <cute/arch/copy_gfx928.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/layout.hpp>

namespace cute
{

/*****************************************************************************************/
/*****************************************************************************************/
template <>
struct Copy_Traits<GFX928_DS_READ_DS_M32x8_B32>
{
  // Logical thread id to thread idx (warp)
  using ThrID = Layout<_64>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_64, _128>,
                           Stride<_128, _1>>;

  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<Shape<_16,_4>,Shape<_32,_2,_2>>,
                            Stride<Stride<_32,_2048>,Stride<_1,_1024,_512>>>;
                        
                          
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  // bool using_lds_offset = true;
};
/*****************************************************************************************/
/*****************************************************************************************/
template <>
struct Copy_Traits<GFX928_DS_READ_DS_M32x16_B16>
{
  // Logical thread id to thread idx (warp)

  using ThrID = Layout<_64>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<Shape <_4,_16>,Shape<_16,_4,_2>>,
                           Stride<Stride<_64,_512>,Stride<_1,_16,_256>>>;
  // using SrcLayout = Layout<Shape<Shape <_4,_16>,Shape<Shape<_64,_2>,_1>>,
  //                          Stride<Stride<_64,_512>,Stride<Stride<_1,_256>,_0>>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout =  Layout<Shape <Shape <_16, _4>,Shape<_16,_2,_4>>,
                              Stride<Stride<_16,_2048>,Stride<_1,_256,_512>>>;
                        

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  // bool using_lds_offset = true;
};

template <>
struct Copy_Traits<GFX928_DS_READ_DS_M32x16_B16_ALT>
{
  using ThrID = Layout<_64>;

  // // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape< Shape<_4,_16>,_128>,
                           Stride< Stride<_128,_512>,_1>>;
  // // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<Shape <_16,_4>,Shape<_16,_2,_4>>,
                          Stride<Stride<_32,_2048>,Stride<_1,_16,_512>>>;
  // // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  // bool using_lds_offset = true;
};
/*****************************************************************************************/
/* not complete,to do... */ 
/*****************************************************************************************/
template <>
struct Copy_Traits<GFX928_DS_READ_DS_M32x32_B8>
{
  // Logical thread id to thread idx (warp)

  using ThrID = Layout<_64>;

  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<Shape <_8,_8>,_4>,
                           Stride<Stride<_128,_1024>,_32>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<Shape<_16,_4>,Shape<_2,_2>>,
                           Stride<Stride<_32,_2048>,Stride<_512,_1024>>>;

  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;

  // bool using_lds_offset = true;
};
/*****************************************************************************************/
/*****************************************************************************************/
}
