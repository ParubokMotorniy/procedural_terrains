using UnityEngine;
using UnityEngine.Assertions;

public static class GenerationUtilities
{
    public static int ComputeCoprime(int valueToMatch, int startPairValue)
    {
        Assert.IsTrue(startPairValue % 2 == 1);

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
        return permuteA;
    }
}
