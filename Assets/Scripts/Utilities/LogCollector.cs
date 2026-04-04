using System.Collections.Generic;
using UnityEngine;

public static class LogCollector
{
    public static readonly List<string> Logs = new List<string>();
    private static readonly List<LogType> allowedLogTypes = new List<LogType> { LogType.Error, LogType.Assert, LogType.Exception };

    public static void Init()
    {
        Application.logMessageReceived += HandleLog;
    }

    private static void HandleLog(string condition, string stackTrace, LogType type)
    {
        if (allowedLogTypes.Contains(type))
            Logs.Add($"[{type}] {condition}");
    }

    public static void Clear()
    {
        Logs.Clear();
    }
}