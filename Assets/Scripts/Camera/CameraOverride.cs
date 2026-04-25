using UnityEngine;
using Unity.Cinemachine;

public class CameraOverride : MonoBehaviour
{
    [SerializeField]
    public CinemachineInputAxisController controller;

    [SerializeField]
    public CinemachineCamera cineCamera;

    public void swapTarget(Transform newCameraTarget)
    {
        cineCamera.LookAt = newCameraTarget;
    }

    void Update()
    {
        controller.enabled = Input.GetMouseButton(1);
    }
}