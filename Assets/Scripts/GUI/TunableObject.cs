using UnityEngine;

public abstract class TunableObject : MonoBehaviour
{
    public abstract void RenderParametersTuningGUI();
    public abstract string GUIStepTitle();
}
