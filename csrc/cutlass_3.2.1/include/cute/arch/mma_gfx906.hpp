/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cute/config.hpp>

#include <cute/arch/mma.hpp>

namespace cute
{

struct GFX906_DP4A
{
  using DRegisters = int32_t[1];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = int32_t[1];

  CUTE_HOST_DEVICE static void
  fma(int32_t& d, uint32_t const& a, uint32_t const& b, int32_t const& c)
  {
    #if defined(DCU_ASM)
        asm volatile("v_dot4_i32_i8 %0, %1, %2, %3;"
                    : "=v"(d)
                    : "v"(a), "v"(b), "v"(c));
    #endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// 1x1x2 half mix
//

struct GFX906_DP2A
{
  using DRegisters = float[1];
  using ARegisters = uint32_t[1];
  using BRegisters = uint32_t[1];
  using CRegisters = float[1];

  CUTE_HOST_DEVICE static void
  fma(float& d, uint32_t const& a, uint32_t const& b, float const& c)
  {
#if defined(DCU_ASM)
      asm volatile("v_dot2_f32_f16 %0,%1,%2,%3\n\t"
                :"=v"(d)
                :"v"(a),"v"(b),"v"(c));
#endif
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
