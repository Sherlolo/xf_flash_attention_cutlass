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

#include <cute/config.hpp>

#include <cute/arch/copy.hpp>

// Config

namespace cute
{
/*************************************************/ 
/*not completed,to do ...*/
/*************************************************/ 
template <class TS, class TD = TS>
struct GFX928_CP_GLOBAL_DIRECT_TO_LDS
{
  using SRegisters = TS[1];
  using DRegisters = TD[1];

  static_assert(sizeof(TS) == sizeof(TD), "cp.async requires sizeof(src_value_type) == sizeof(dst_value_type)");
  static_assert(sizeof(TS) == 4 || sizeof(TS) == 8 || sizeof(TS) == 16, "cp.async sizeof(TS) is not supported");

  CUTE_HOST_DEVICE static void
  copy(TS const& gmem_src,
       TD      & smem_dst)
  {
    	// buffer_load_dwordx4	v[0:3], v[vgprGlobalOffsetA], s[sgprSrcA:sgprSrcA+3], 0, offen, offset:0x00, lds	
  }
};


struct GFX928_DS_READ_DS_M32x8_B32
{
  using SRegisters = uint128_t[1];
  using DRegisters = float[4];

  
  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       float& dst0, 
       float& dst1, 
       float& dst2, 
       float& dst3)
  {
    #if defined(__gfx928__) && defined(DCU_ASM)
    v4f d;

    int lds_offset = reinterpret_cast<const char *>(&smem_src) - (char *)(0x1000000000000);
    // printf("threadId[%d]:%p\n",threadIdx.x,&(smem_src));
    asm volatile("DS_READ_M32x8_B32 %0, %1, offset:0x00 \n\t"
                  "s_waitcnt lgkmcnt(0)\n\t"
                  : "+v"(d)
                  : "v"(lds_offset));
    float * dst = reinterpret_cast<float *>(&d);
    dst0 = dst[0];
    dst1 = dst[1];
    dst2 = dst[2];
    dst3 = dst[3];
    #endif
  }
};


struct GFX928_DS_READ_DS_M32x16_B16
{
  //源共享内存数据是两个64位共享内存
  using SRegisters = uint64_t[2];
  using DRegisters = uint16_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint64_t const& smem_src1,
       uint64_t const& smem_src2,
       uint16_t& dst0, uint16_t& dst1, uint16_t& dst2, uint16_t& dst3,
       uint16_t& dst4, uint16_t& dst5, uint16_t& dst6, uint16_t& dst7)
  {

    #if defined(__gfx928__) && defined(DCU_ASM)
      uint32x4_t d;
      int lds_offset = reinterpret_cast<const char *>(&smem_src1) - (char *)(0x1000000000000);
      // __builtin_amdgcn_ds_read_m32x16f16_alt(lds,offset)
      asm volatile("DS_READ_M32x16_B16 %0, %1, offset:0x00 \n\t"
                    "s_waitcnt lgkmcnt(0) \n\t"
                    : "+v"(d)
                    : "v"(lds_offset));

      uint16_t * dst = reinterpret_cast<uint16_t *>(&d);
      dst0 = dst[0];
      dst1 = dst[1];
      dst2 = dst[2];
      dst3 = dst[3];
      dst4 = dst[4];
      dst5 = dst[5];
      dst6 = dst[6];
      dst7 = dst[7];

    #endif
  
  }

};

struct GFX928_DS_READ_DS_M32x16_B16_ALT
{
  using SRegisters = uint128_t[1];
  using DRegisters = half_t[8];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       half_t& dst0, half_t& dst1, half_t& dst2, half_t& dst3,
       half_t& dst4, half_t& dst5, half_t& dst6, half_t& dst7)
  {
    #if defined(__gfx928__) && defined(DCU_ASM)
    uint32x4_t d;
    int lds_offset = reinterpret_cast<const char *>(&smem_src) - (char *)(0x1000000000000);
    asm volatile("DS_READ_M32x16_B16_alt %0, %1, offset:0x00 \n\t"
                  "s_waitcnt lgkmcnt(0) \n\t"
                  : "+v"(d)
                  : "v"(lds_offset));
    half_t * dst = reinterpret_cast<half_t *>(&d);
    dst0 = dst[0];
    dst1 = dst[1];
    dst2 = dst[2];
    dst3 = dst[3];
    dst4 = dst[4];
    dst5 = dst[5];
    dst6 = dst[6];
    dst7 = dst[7];
    #endif
  }
};
/*************************************************/ 
struct GFX928_DS_READ_DS_M32x32_B8
{
  using SRegisters = uint128_t[1];
  using DRegisters = uint32_t[4];

  CUTE_HOST_DEVICE static void
  copy(uint128_t const& smem_src,
       uint32_t& dst0, uint32_t& dst1, uint32_t& dst2, uint32_t& dst3)
  {
    #if defined(__gfx928__) && defined(DCU_ASM)

    

    #endif
  }
};
/////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
