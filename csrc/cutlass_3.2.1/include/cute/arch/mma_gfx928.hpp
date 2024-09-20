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
   struct GFX928_16x16x8_F32F32F32F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = float[2];
    using BRegisters = float[2];
    using CRegisters = float[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        float const& a0, float const& a1,
        float const& b0, float const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      // printf("a:%f %f b:%f %f\n",a0,a1,b0,b1);
      #if defined(__gfx928__) && defined(DCU_ASM)
          v4f c;
          v4f d;
          c.x = c0;
          c.y = c1;
          c.z = c2;
          c.w = c3;
          v2f a;
          v2f b;
          a.x = a0;
          a.y = a1;
          b.x = b0;
          b.y = b1;
          asm volatile("v_mmac_16x16x8_f32 %0, %1, %2, %3\n\t"
                            : "+v"(d)
                            : "v"(a), "v"(b), "v"(c));
          d0 = d.x;
          d1 = d.y;
          d2 = d.z;
          d3 = d.w;
      #endif
    }
  };
  
  struct GFX928_16x16x8_F32TF32TF32F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = tfloat32_t[2];
    using BRegisters = tfloat32_t[2];
    using CRegisters = float[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        tfloat32_t const& a0, tfloat32_t const& a1,
        tfloat32_t const& b0, tfloat32_t const& b1,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if defined(__gfx928__) && defined(DCU_ASM)
          v4f c;
          v4f d;
          c.x = c0;
          c.y = c1;
          c.z = c2;
          c.w = c3;

          cutlass::Array<tfloat32_t,2> a;
          a[0] = a0;
          a[1] = a1;
          cutlass::Array<tfloat32_t,2> b;
          b[0] = b0;
          b[1] = b1;
          asm volatile("v_mmac_f32_16x16x8_tf32 %0, %1, %2, %3\n\t"
                            : "+v"(d)
                            : "v"(a), "v"(b), "v"(c));
          d0 = d.x;
          d1 = d.y;
          d2 = d.z;
          d3 = d.w;
      #endif
    }
  };
  struct GFX928_16x16x16_F32F16F16F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = half_t[4];
    using BRegisters = half_t[4];
    using CRegisters = float[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if defined(__gfx928__) && defined(DCU_ASM)
        v4f c;
        v4f d;
        c.x = c0;
        c.y = c1;
        c.z = c2;
        c.w = c3;



        __fp16x4_t A,B;
        A.x = a0; A.y = a1; A.z = a2; A.w = a3;
        B.x = b0; B.y = b1; B.z = b2; B.w = b3;
        d = __builtin_amdgcn_mmac_f32_16x16x16f16(A,B,c);

        d0 = d.x;
        d1 = d.y;
        d2 = d.z;
        d3 = d.w;
      #endif
    }
  
  };

  struct GFX928_16x16x16_F32F16F16F32_NT_FOR_GEMM1
  {
    using DRegisters = float[4];
    using ARegisters = half_t[4];
    using BRegisters = half_t[4];
    using CRegisters = float[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if defined(__gfx928__) && defined(DCU_ASM)
        v4f c;
        v4f d;
        c.x = c0;
        c.y = c1;
        c.z = c2;
        c.w = c3;

        __fp16x4_t A,B;
        A.x = a0; A.y = a1; A.z = a2; A.w = a3;
        B.x = b0; B.y = b1; B.z = b2; B.w = b3;
        d = __builtin_amdgcn_mmac_f32_16x16x16f16(A,B,c);

        d0 = d.x;
        d1 = d.y;
        d2 = d.z;
        d3 = d.w;
      #endif
    }
  
  };

  struct GFX928_16x16x16_F32BF16BF16F32_NT_FOR_GEMM1
  {
    using DRegisters = float[4];
    using ARegisters = bfloat16_t[4];
    using BRegisters = bfloat16_t[4];
    using CRegisters = float[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if defined(__gfx928__) && defined(DCU_ASM)
          v4f c;
          v4f d;
          c.x = c0;
          c.y = c1;
          c.z = c2;
          c.w = c3;
          cutlass::Array<bfloat16_t,4> a;
          a[0] = a0;
          a[1] = a1;
          a[2] = a2;
          a[3] = a3;
          cutlass::Array<bfloat16_t,4> b;
          b[0] = b0;
          b[1] = b1;
          b[2] = b2;
          b[3] = b3;

          // asm volatile("v_mmac_f32_16x16x16_bf16 %0, %1, %2, %3\n\t"
          //                   : "+v"(d)
          //                   : "v"(a), "v"(b), "v"(c));
          __bf16x4_t A,B;
          A = *(reinterpret_cast<__bf16x4_t *>(&a));
          B = *(reinterpret_cast<__bf16x4_t *>(&b));
          
          d = __builtin_amdgcn_mmac_f32_16x16x16bf16(A, B, c);

          d0 = d.x;
          d1 = d.y;
          d2 = d.z;
          d3 = d.w;

          // d0 = d.x;
          // d1 = d.y;
          // d2 = d.z;
          // d3 = d.w;
      #endif
    }
  
  };

  struct GFX928_16x16x16_F32BF16BF16F32_NT
  {
    using DRegisters = float[4];
    using ARegisters = bfloat16_t[4];
    using BRegisters = bfloat16_t[4];
    using CRegisters = float[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        bfloat16_t const& a0, bfloat16_t const& a1,bfloat16_t const& a2, bfloat16_t const& a3,
        bfloat16_t const& b0, bfloat16_t const& b1,bfloat16_t const& b2, bfloat16_t const& b3,
        float const& c0, float const& c1, float const& c2, float const& c3)
    {
      #if defined(__gfx928__) && defined(DCU_ASM)
          v4f c;
          v4f d;
          c.x = c0;
          c.y = c1;
          c.z = c2;
          c.w = c3;
          cutlass::Array<bfloat16_t,4> a;
          a[0] = a0;
          a[1] = a1;
          a[2] = a2;
          a[3] = a3;
          cutlass::Array<bfloat16_t,4> b;
          b[0] = b0;
          b[1] = b1;
          b[2] = b2;
          b[3] = b3;

          // asm volatile("v_mmac_f32_16x16x16_bf16 %0, %1, %2, %3\n\t"
          //                   : "+v"(d)
          //                   : "v"(a), "v"(b), "v"(c));
          __bf16x4_t A,B;
          A = *(reinterpret_cast<__bf16x4_t *>(&a));
          B = *(reinterpret_cast<__bf16x4_t *>(&b));
          
          d = __builtin_amdgcn_mmac_f32_16x16x16bf16(A, B, c);

          d0 = d.x;
          d1 = d.y;
          d2 = d.z;
          d3 = d.w;

          // d0 = d.x;
          // d1 = d.y;
          // d2 = d.z;
          // d3 = d.w;
      #endif
    }
  };
  struct GFX928_16x16x32_I32I8I8I32_NT
  {
    using DRegisters = int[4];
    using ARegisters = int8_t[8];
    using BRegisters = int8_t[8];
    using CRegisters = int[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3, 
        int8_t const& a0, int8_t const& a1,int8_t const& a2, int8_t const& a3,
        int8_t const& a4, int8_t const& a5,int8_t const& a6, int8_t const& a7,
        int8_t const& b0, int8_t const& b1,int8_t const& b2, int8_t const& b3,
        int8_t const& b4, int8_t const& b5,int8_t const& b6, int8_t const& b7,
        int const& c0, int const& c1, int const& c2, int const& c3)
    {
      #if defined(__gfx928__) && defined(DCU_ASM)
        intx4_t c;
        intx4_t d;
        c.x = c0;
        c.y = c1;
        c.z = c2;
        c.w = c3;
        cutlass::Array<int8_t,8> a;
        a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
        a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;
    
        cutlass::Array<int8_t,8> b;
        b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
        b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
        long A, B;
        A = *(reinterpret_cast<long *>(&a));
        B = *(reinterpret_cast<long *>(&b));
        d = __builtin_amdgcn_mmac_i32_16x16x32i8(A,B,c);

        d0 = d.x;
        d1 = d.y;
        d2 = d.z;
        d3 = d.w;
      #endif
    }
 
  };
  struct GFX928_16x16x32_I32U8U8I32_NT
  {
    using DRegisters = int[4];
    using ARegisters = uint8_t[8];
    using BRegisters = uint8_t[8];
    using CRegisters = int[4];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(int      & d0, int      & d1, int      & d2, int      & d3, 
        uint8_t const& a0, uint8_t const& a1,uint8_t const& a2, uint8_t const& a3,
        uint8_t const& a4, uint8_t const& a5,uint8_t const& a6, uint8_t const& a7,
        uint8_t const& b0, uint8_t const& b1,uint8_t const& b2, uint8_t const& b3,
        uint8_t const& b4, uint8_t const& b5,uint8_t const& b6, uint8_t const& b7,
        int const& c0, int const& c1, int const& c2, int const& c3)
    {
      #if defined(__gfx928__) && defined(DCU_ASM)
        intx4_t c;
        intx4_t d;
        c.x = c0;
        c.y = c1;
        c.z = c2;
        c.w = c3;
        cutlass::Array<uint8_t,8> a;
        a[0] = a0; a[1] = a1; a[2] = a2; a[3] = a3;
        a[4] = a4; a[5] = a5; a[6] = a6; a[7] = a7;
    
        cutlass::Array<uint8_t,8> b;
        b[0] = b0;b[1] = b1;b[2] = b2;b[3] = b3;
        b[4] = b4;b[5] = b5;b[6] = b6;b[7] = b7;
        long A, B;
        A = *(reinterpret_cast<long *>(&a));
        B = *(reinterpret_cast<long *>(&b));
        d = __builtin_amdgcn_mmac_i32_16x16x32u8(A,B,c);

        d0 = d.x;
        d1 = d.y;
        d2 = d.z;
        d3 = d.w;
      #endif
    }
  };
  
  struct GFX928_32x32x16_F32F16F16F32_NT
  {
    using DRegisters = float[16];
    using ARegisters = half_t[8];
    using BRegisters = half_t[8];
    using CRegisters = float[16];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        float      & d4, float      & d5, float      & d6, float      & d7, 
        float      & d8, float      & d9, float      & d10, float      & d11, 
        float      & d12, float      & d13, float      & d14, float      & d15, 
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {

      #if defined(__gfx928__) && defined(DCU_ASM)

        v4f C0,C1,C2,C3;
        v4f D0,D1,D2,D3;

        C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
        C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
        C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        __fp16x4_t A,B,A0,B0;

        A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
        A0.x = a4; A0.y = a5; A0.z = a6; A0.w = a7;
        B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;
        B0.x = b4; B0.y = b5; B0.z = b6; B0.w = b7;


        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
                  : "+v"(D0)
                  : "v"(A), "v"(B), "v"(C0));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
                  : "+v"(D1)
                  : "v"(A), "v"(B0), "v"(C1));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
                  : "+v"(D2)
                  : "v"(A0), "v"(B), "v"(C2));
        asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
                  : "+v"(D3)
                  : "v"(A0), "v"(B0), "v"(C3));

        d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
        d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
        d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;

        #endif
    }
  };



  struct GFX928_32x32x16_F32F16F16F32_NT_ALT
  {
    using DRegisters = float[16];
    using ARegisters = half_t[8];
    using BRegisters = half_t[8];
    using CRegisters = float[16];

    // Register asm fma
    CUTE_HOST_DEVICE static void
    fma(float      & d0, float      & d1, float      & d2, float      & d3, 
        float      & d4, float      & d5, float      & d6, float      & d7, 
        float      & d8, float      & d9, float      & d10, float      & d11, 
        float      & d12, float      & d13, float      & d14, float      & d15, 
        half_t const& a0, half_t const& a1,half_t const& a2, half_t const& a3,
        half_t const& a4, half_t const& a5,half_t const& a6, half_t const& a7,
        half_t const& b0, half_t const& b1,half_t const& b2, half_t const& b3,
        half_t const& b4, half_t const& b5,half_t const& b6, half_t const& b7,
        float const& c0, float const& c1, float const& c2, float const& c3,
        float const& c4, float const& c5, float const& c6, float const& c7,
        float const& c8, float const& c9, float const& c10, float const& c11,
        float const& c12, float const& c13, float const& c14, float const& c15)
    {

      #if defined(__gfx928__) && defined(DCU_ASM)

        v4f C0,C1,C2,C3;
        v4f D0,D1,D2,D3;

        C0.x = c0;  C0.y = c1;  C0.z = c2;  C0.w = c3;
        C1.x = c4;  C1.y = c5;  C1.z = c6;  C1.w = c7;
        C2.x = c8;  C2.y = c9;  C2.z = c10; C2.w = c11;
        C3.x = c12; C3.y = c13; C3.z = c14; C3.w = c15;

        __fp16x4_t A,B,A0,B0;

        A.x  = a0; A.y  = a1; A.z  = a2; A.w  = a3;
        A0.x = a4; A0.y = a5; A0.z = a6; A0.w = a7;
        B.x  = b0; B.y  = b1; B.z  = b2; B.w  = b3;
        B0.x = b4; B0.y = b5; B0.z = b6; B0.w = b7;
        
        D0 = __builtin_amdgcn_mmac_f32_16x16x16f16(A,B,C0);
        D1 = __builtin_amdgcn_mmac_f32_16x16x16f16(A,B0,C1);
        D2 = __builtin_amdgcn_mmac_f32_16x16x16f16(A0,B,C2);
        D3 = __builtin_amdgcn_mmac_f32_16x16x16f16(A0,B0,C3);
       
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D0)
        //           : "v"(A), "v"(B), "v"(C0));
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D1)
        //           : "v"(A), "v"(B0), "v"(C1));
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D2)
        //           : "v"(A0), "v"(B), "v"(C2));
        // asm volatile("v_mmac_f32_16x16x16_f16 %0, %1, %2, %3\n\t"
        //           : "+v"(D3)
        //           : "v"(A0), "v"(B0), "v"(C3));

        d0 = D0.x;  d1 = D0.y;    d2 = D0.z;  d3 = D0.w;
        d4 = D1.x;  d5 = D1.y;    d6 = D1.z;  d7 = D1.w;
        d8 = D2.x;  d9 = D2.y;    d10 = D2.z; d11 = D2.w;
        d12 = D3.x; d13 = D3.y;   d14 = D3.z; d15 = D3.w;


        #endif
    
    }
  
  };
////////////////////////////////////////////////////////////////////////////////////////////////////

} // end namespace cute
