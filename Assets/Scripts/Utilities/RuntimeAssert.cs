using UnityEngine;

public static class RuntimeAssert
{
    public static bool IsTrue(bool condition, string message)
    {
        if (!condition)
        {
            Debug.LogError("[ASSERT] " + message);
        }
        return !condition;
    }
}