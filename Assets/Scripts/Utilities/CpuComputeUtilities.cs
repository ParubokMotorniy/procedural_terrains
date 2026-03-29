using UnityEngine;

using Unity.Mathematics;
using Unity.Collections;
using static Unity.Mathematics.math;
using System;

public static class CpuComputeUtilities
{
    public const float PI = 3.141592654f;
    public const float TWO_PI = 6.283185307f;
    public const float EPS = 1.0e-7f;

    public static int2 terrainWrap(int2 a, int2 b)
    {
        return (a % b + b) % b;
    }

    // only makes sense if height is normalized.
    public static float computeSoftnessCoefficient(float2 localGradient, float localHeight, float baseHardness)
    {
        const float heightWeight = 0.25f;
        const float gradientWeight = 1.0f - heightWeight;

        return clamp(
            1.0f - (localHeight * heightWeight) * (exp(-length(localGradient)) * gradientWeight),
            baseHardness,
            1.0f
        );
    }

    public static float2 computeGradientAtPoint(
        NativeArray<float> tex,
        uint2 dimensions,
        int2 texelOfInterest)
    {
        int2 dim = (int2)dimensions;

        int2 xp = terrainWrap(texelOfInterest + new int2(1, 0), dim);
        int2 xm = terrainWrap(texelOfInterest - new int2(1, 0), dim);

        int2 zp = terrainWrap(texelOfInterest + new int2(0, 1), dim);
        int2 zm = terrainWrap(texelOfInterest - new int2(0, 1), dim);

        float dX =
            (tex[(int)index2dTo1d(dimensions, (uint2)xp)] -
             tex[(int)index2dTo1d(dimensions, (uint2)xm)]) / 2.0f;

        float dZ =
            (tex[(int)index2dTo1d(dimensions, (uint2)zp)] -
             tex[(int)index2dTo1d(dimensions, (uint2)zm)]) / 2.0f;

        return new float2(dX, dZ);
    }

    public static uint index2dTo1d(uint2 gridDimensions, uint2 index2D)
    {
        return index2D.x * gridDimensions.y + index2D.y;
    }

    // samples from normal distribution. u must be uniformly distributed, though.
    public static float sampleGaussBoxMuller(float2 u)
    {
        float a = sqrt(-2.0f * log(1.0f - u.x));
        float b = TWO_PI * u.y;

        return cos(b) * a;
    }

    // (c) Jarzynski and Olano
    public static uint pcgHash(uint v)
    {
        v = v * 747796405u + 2891336453u;
        uint word = ((v >> ((int)(v >> 28) + 4)) ^ v) * 277803737u;
        return (word >> 22) ^ word;
    }

    public static float u01FromUint(uint x)
    {
        return (x + 1.0f) * (1.0f / 4294967296.0f);
    }

    public static uint seedFromXYPass(uint x, uint y, int passId)
    {
        uint v = x * 0x68E31DA4u ^ y * 0xB5297A4Du ^ (uint)passId * 0x1B56C4E9u;
        return v;
    }

    // https://www.shadertoy.com/view/lXjGRw
    // return value 0~1 but tends toward central values
    public static float pinkNoise2D(uint2 x)
    {
        float sum_y = 0.0f;
        float sum_w = 0.0f;
        float w = 1.0f;

        x *= 2u;

        for (int i = 0; i < 10; ++i)
        {
            uint p = (x.x / 2u) ^ ((x.y / 2u) << 16);
            sum_y += w * ((float)pcgHash(p) / 4294967296.0f);
            sum_w += w;

            x = (x + 1u) / 2u;
        }

        return max(0.0f, sum_y / sum_w);
    }

    public static float accurateSum(float a, float b, out float error)
    {
        float sum = a + b;

        float bs = sum - a;
        float @as = sum - bs;

        error = (b - bs) + (sum - @as);

        return sum;
    }
}