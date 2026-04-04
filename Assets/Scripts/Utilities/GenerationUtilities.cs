using System;
using System.Collections.Generic;
using NUnit.Framework.Internal;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.InputSystem.Controls;

public static class GenerationUtilities
{
    public static int ComputeCoprime(int valueToMatch, int startPairValue)
    {
        if (RuntimeAssert.IsTrue(startPairValue % 2 == 1, "Failed to start the search for coprimes. You should pride yourself on getting such a rare error."))
            return -1;

        int permuteA = startPairValue;
        while (true)
        {
            permuteA += 2; //only odd numbers have a chance
            int gcd = 0;
            for (gcd = permuteA; gcd > 0; --gcd)
            {
                if ((valueToMatch % gcd) == 0 && (permuteA % gcd) == 0)
                {
                    //largest so far, no need to seek further
                    return permuteA;
                }
            }
            if (gcd == 1)
            {
                return permuteA;
            }
        }
    }

    //returns: group_size, n_groups
    public static (int, int) GetOptimalNumberOfGroups(int linearWorkItems, int[] availableGroupSizes, int maxWorkPerThread, int minWorkPerThread = 1)
    {
        for (int i = availableGroupSizes.Length - 1; i >= 0; --i)
        {
            int testedGroupSizeAlongDimension = availableGroupSizes[i];
            RuntimeAssert.IsTrue(testedGroupSizeAlongDimension < 1024, "The group size exceeds hardware limitations (on my machine).");

            int preferredLocalGroups = 100 * (int)math.ceil(320 / testedGroupSizeAlongDimension); //roughly 8 * 40 = 320 threads (5 waves) per CU

            int maxGroups = (int)math.min(preferredLocalGroups, math.ceil((float)linearWorkItems / (testedGroupSizeAlongDimension * minWorkPerThread)));
            int minGroups = (int)math.max(1, math.floor((float)linearWorkItems / (testedGroupSizeAlongDimension * maxWorkPerThread)));
            for (int g = maxGroups; g >= minGroups; --g)
            {
                int totalThreads = g * testedGroupSizeAlongDimension;
                if (linearWorkItems % totalThreads == 0)
                {
#if UNITY_EDITOR
                    if (testedGroupSizeAlongDimension < 64)
                        Debug.LogWarning("The chosen group size along a dimension (" + testedGroupSizeAlongDimension + ") is not a multiple of 64 (wave size).");
#endif
                    return (testedGroupSizeAlongDimension, g);
                }
            }
        }
        return (-1, -1);
    }
}
