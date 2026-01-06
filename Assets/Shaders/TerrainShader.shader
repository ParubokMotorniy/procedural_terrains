Shader "Custom/TerrainShader"
{
    Properties
    {
        [MainColor] _BaseColor("Base Color", Color) = (1, 1, 1, 1)
        [MainTexture] _BaseMap("Base Map", 2D) = "white"

        _HeightMap("Height Map", 2D) = "black"
        _HeightScale("Height Scale", Float) = 1.0
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" }

        Pass
        {
            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv          : TEXCOORD0;
                float4 vertexColor : COLOR0;
            };

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            TEXTURE2D(_HeightMap);
            SAMPLER(sampler_HeightMap);

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                float4 _BaseMap_ST;
                float  _HeightScale;
            CBUFFER_END

            Varyings vert(Attributes IN)
            {
                Varyings OUT;

                float height =
                    SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv,
                        0
                    ).r;

                float3 positionOS = IN.positionOS.xyz;
                positionOS.y = height * _HeightScale;

                OUT.positionHCS = TransformObjectToHClip(positionOS);
                OUT.uv = TRANSFORM_TEX(IN.uv, _BaseMap);
                OUT.vertexColor = float4(height,height,height,height);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                return IN.vertexColor;
                // half4 color =
                    // SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv)
                    // * _BaseColor;
                // return color;
            }

            ENDHLSL
        }
    }
}
