/* ============================================================================
* Copyright (c) 2009-2016 BlueQuartz Software, LLC
*
* Redistribution and use in source and binary forms, with or without modification,
* are permitted provided that the following conditions are met:
*
* Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
*
* Redistributions in binary form must reproduce the above copyright notice, this
* list of conditions and the following disclaimer in the documentation and/or
* other materials provided with the distribution.
*
* Neither the name of BlueQuartz Software, the US Air Force, nor the names of its
* contributors may be used to endorse or promote products derived from this software
* without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
* USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
* The code contained herein was partially funded by the followig contracts:
*    United States Air Force Prime Contract FA8650-07-D-5800
*    United States Air Force Prime Contract FA8650-10-D-5210
*    United States Prime Contract Navy N00173-07-C-2068
*
* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

#pragma once
/** @file EbsdLibMath.h
 * @brief This file performs the necessary including of <math.h> with certain
 * define constants (like M_PI) defined on all platforms.
 */

#if defined(_MSC_VER)
/* [i_a] MSVC again: see the comment from Microsoft's math.h below (MSVC2005).

     Other compilers may also lack M_PI / M_PI_2 which both are used in the
     OpenEXR (test) code.

     Microsoft says:

         Define _USE_MATH_DEFINES before including math.h to expose these macro
         definitions for common math constants.  These are placed under an #ifdef
         since these commonly-defined names are not part of the C/C standards.

     End of quote.

     The other defines have been added for completeness sake (they exist on
     BSD and Linux at least and other code may expect these).

     Microsoft doesn't define M_2PI ever, other compilers may lack some of these
     too, hence the sequence as it is: load math.h, then see what's lacking still.
  */
#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES 1
#endif

/*
  "It's a known, long-standing bug in the compiler system's headers.  For
  some reason the manufacturer, in its infinite wisdom, chose to #define
  macros min() and max() in violation of the upper-case convention and so
  break any legitimate functions with those names, including those in the
  standard C++ library."
  */
#ifndef NOMINMAX
#define NOMINMAX
#endif

#endif

#include <cstddef>
#include <cmath>
#include <vector>
#include <limits>

#include "EbsdLib/EbsdLib.h"

#ifndef M_E
#define M_E            2.7182818284590452354   /* e */
#endif

#ifndef M_LOG2E
#define M_LOG2E        1.4426950408889634074   /* log_2 e */
#endif

#ifndef M_LOG10E
#define M_LOG10E       0.43429448190325182765  /* log_10 e */
#endif

#ifndef M_LN2
#define M_LN2          0.69314718055994530942  /* log_e 2 */
#endif

#ifndef M_LN10
#define M_LN10         2.30258509299404568402  /* log_e 10 */
#endif

#ifndef M_PI_2
#define M_PI_2         1.57079632679489661923132169163975144   /* pi/2 */
#endif

#ifndef M_PI_4
#define M_PI_4         0.785398163397448309615660845819875721  /* pi/4 */
#endif

#ifndef M_1_PI
#define M_1_PI         0.31830988618379067154  /* 1/pi */
#endif

#ifndef M_2_PI
#define M_2_PI         0.63661977236758134308  /* 2/pi */
#endif

#ifndef M_2_SQRTPI
#define M_2_SQRTPI     1.12837916709551257390  /* 2/sqrt(pi) */
#endif

#ifndef M_SQRT2
#define M_SQRT2        1.41421356237309504880  /* sqrt(2) */
#endif

#ifndef M_SQRT1_2
#define M_SQRT1_2      0.70710678118654752440  /* 1/sqrt(2) */
#endif

#ifndef M_PI
#define M_PI        3.14159265358979323846  /* pi */
#endif

#ifndef M_2PI
#define M_2PI           6.283185307179586232    /* 2*pi  */
#endif

namespace EbsdLib
{
  namespace Constants
  {
    static const float k_Pif = static_cast<float>(M_PI);
    static const double k_Pi = M_PI;
    static const double k_SqrtPi = sqrt(M_PI);
    static const double k_2OverSqrtPi = 2.0 / sqrt(M_PI);
    static const double k_HalfOfSqrtPi = sqrt(M_PI) / 2.0;
    static const double k_SqrtHalfPi = sqrt(M_PI_2);
    static const double k_2Pi = 2.0 * M_PI;
    static const double k_1OverPi = 1.0 / M_PI;
    static const double k_PiOver180 = M_PI / 180.0;
    static const double k_360OverPi = 360.0 / M_PI;
    static const double k_180OverPi = 180.0 / M_PI;
    static const double k_PiOver2 = M_PI / 2.0;
    static const double k_PiOver3 = M_PI / 3.0;
    static const double k_PiOver4 = M_PI / 4.0;
    static const double k_PiOver8 = M_PI / 8.0;
    static const double k_PiOver12 = M_PI / 12.0;
    static const double k_Sqrt2 = sqrt(2.0);
    static const double k_Sqrt3 = sqrt(3.0);
    static const double k_HalfSqrt2 = 0.5 * sqrt(2.0);
    static const double k_1OverRoot2 = 1.0 / sqrt(2.0);
    static const double k_1OverRoot3 = 1.0 / sqrt(3.0);
    static const double k_Root3Over2 = sqrt(3.0) / 2.0;
    static const double k_DegToRad = M_PI / 180.0;
    static const double k_RadToDeg = 180.0 / M_PI;
    static const double k_1Point3 = 1.0 + 1.0 / 3.0;
    static const double k_1Over3 = 1.0 / 3.0;

    static const double k_ACosNeg1 = acos(-1.0);
    static const double k_ACos1 = acos(1.0);

    static const double k_Tan_OneEigthPi = tan(k_PiOver8);
    static const double k_Cos_OneEigthPi = cos(k_PiOver8);
    static const double k_Cos_ThreeEightPi = cos(3.0 * k_PiOver8);
    static const double k_Sin_ThreeEightPi = sin(3.0 * k_PiOver8);



  }
}

class EbsdLibMath
{
  public:
    virtual ~EbsdLibMath();

    static EbsdLib_EXPORT float Gamma(float);
    
    template <typename T>
    static void bound(T& val, T min, T max)
    {
      if(val < min)
      {
        val = min;
      }
      else if(val > max)
      {
        val = max;
      }
    }

    static EbsdLib_EXPORT float erf(float);
    static EbsdLib_EXPORT float erfc(float);
    static EbsdLib_EXPORT float gammastirf(float);
    static EbsdLib_EXPORT float LnGamma(float, float&);
    static EbsdLib_EXPORT float incompletebeta(float, float, float);
    static EbsdLib_EXPORT float incompletebetafe(float, float, float, float, float);
    static EbsdLib_EXPORT float incompletebetafe2(float, float, float, float, float);
    static EbsdLib_EXPORT float incompletebetaps(float, float, float, float);

    /**
     * @brief generates a linearly space array between 2 numbers (inclusive, assumes first number <= second number) [as matlabs linspace]
     * @param first number
     * @param second number
     * @param length of return array
     * @return
     */
    static EbsdLib_EXPORT std::vector<double> linspace(double first, double second, int length);

    /**
     * @brief closeEnough
     * @param a
     * @param b
     * @param epsilon
     * @return
     */
    template<typename K>
    static bool closeEnough(const K& a, const K& b,
                            const K& epsilon = std::numeric_limits<K>::epsilon())
    {
      return (epsilon > fabs(a - b));
    }


    /**
     * @brief transfer_sign
     * @param a
     * @param b
     * @return
     */
    template<typename K>
    static K transfer_sign(K a, K b)
    {
      if( a > 0.0 && b > 0.0) { return a; }
      if( a < 0.0 && b > 0.0) { return -1 * a; }

      if( a < 0.0 && b < 0.0) { return a; }

      return -1 * a;

    }


  protected:
    EbsdLibMath();

  public:
    EbsdLibMath(const EbsdLibMath&) = delete;            // Copy Constructor Not Implemented
    EbsdLibMath(EbsdLibMath&&) = delete;                 // Move Constructor Not Implemented
    EbsdLibMath& operator=(const EbsdLibMath&) = delete; // Copy Assignment Not Implemented
    EbsdLibMath& operator=(EbsdLibMath&&) = delete;      // Move Assignment Not Implemented
};


