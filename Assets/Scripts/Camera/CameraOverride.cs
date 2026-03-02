using UnityEngine;
using Unity.Cinemachine;

public class CameraOverride : MonoBehaviour
{
    public CinemachineInputAxisController controller;

    void Update()
    {
        controller.enabled = Input.GetMouseButton(1);
    }
}