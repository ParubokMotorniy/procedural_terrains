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
            Cull Off

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
                float4 vertexColor : TEXCOORD1;
            };

            static const float3 levelColoring[6] = {
                float3(0.6, 0.271, 0),
                float3(0.98, 0.894, 0.329),
                float3(0.173, 0.729, 0.016),
                float3(0.737, 0.749, 0.733),
                float3(0.9, 0.9, 0.95),
                float3(1.0, 1.0, 1.0)
            };
            static const float isoColorStep = 0.2;

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);

            TEXTURE2D(_HeightMap);
            SAMPLER(sampler_HeightMap);

            CBUFFER_START(UnityPerMaterial)
                half4 _BaseColor;
                float4 _BaseMap_ST;
                float  _HeightScale;
            CBUFFER_END

            float colorInterpolationRemap(float linearCoeff)
            {
                return 1.0 / (1.0 + exp(-40.0*(linearCoeff - 0.5)));
            }

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                ZERO_INITIALIZE(Varyings, OUT);

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

                float isoColorLevel = height / isoColorStep;
                float3 bottomColor = levelColoring[(int)floor(isoColorLevel)]; 
                float3 topColor = levelColoring[(int)ceil(isoColorLevel)];

                float3 vertexColorAtLevel = lerp(bottomColor, topColor, colorInterpolationRemap(frac(isoColorLevel)));

                OUT.vertexColor = float4(vertexColorAtLevel, 1.0);
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
