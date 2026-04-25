Shader "Custom/TerrainShader"
{
    Properties
    {
        _HeightMap("Height Map", 2D) = "black"
        _HeightScale("Height Scale", Float) = 1.0
        _Specular("Specular strength", Float) = 0.05
        _Smoothness("Smoothness", Float) = 0.05
    }
    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }
        //TODO: add runtime wiremesh rendering. Probably with geometry shaders for I don't want to waste bandwidht passing extra vertex attributes around.

        Pass
        {
            Name "Terrain shading"
            Tags {"LightMode" = "UniversalForward" "PassFlags" = "OnlyDirectional" "UniversalMaterialType" = "SimpleLit"}
            
            Cull Off
            
            HLSLPROGRAM
            
            #pragma vertex vert
            #pragma fragment frag
            #define _SPECULAR_COLOR
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"


            struct Attributes
            {
                float4 positionOS : POSITION;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv          : TEXCOORD0;
                float vertexHeight : TEXCOORD1;
                float3 positionWS  : TEXCOORD2; 
                float3 normal      : TEXCOORD3;
            };

            static const float3 levelColoring[7] = {
                float3(0.81, 0.8, 0.31),

                float3(0.529, 0.655, 0.449),
                float3(0.373, 0.537, 0.3333),
                float3(0.929, 0.867, 0.635),
                float3(0.82, 0.624, 0.424),
                float3(0.498, 0.525, 0.549),

                float3(1.0, 1.0, 1.0)
            };

            static const float levelSmoothness[7] = {
                0.0,

                0.8,
                0.65,
                0.9,
                0.9,
                0.45,

                0.15
            };

            static const float levelSpecularity[7] = {
                0.0,

                0.15,
                0.15,
                0.4,
                0.4,
                0.8,

                1.0
            };

            static const float isoColorStep = 0.2;
            static const half3 ambientLight = half3(0.208, 0.251, 0.349);
            
            TEXTURE2D(_HeightMap);
            SAMPLER(sampler_HeightMap);
            
            CBUFFER_START(UnityPerMaterial)
            float4 _HeightMap_ST;
            float  _HeightScale;
            float4 _HeightMap_TexelSize; 
            float _Specular;
            float _Smoothness;
            CBUFFER_END

            float colorInterpolationRemap(float linearCoeff)
            {
                return 1.0 / (1.0 + exp(-30.0*(linearCoeff - 0.5)));
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

                float du = SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv + float2(_HeightMap_TexelSize.x, 0.0),
                        0
                    ).r - SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv - float2(_HeightMap_TexelSize.x, 0.0),
                        0
                    ).r;
                du /= 2.0 * _HeightMap_TexelSize.x;

                float dv = SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv + float2(0.0, _HeightMap_TexelSize.y),
                        0
                    ).r - SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv - float2(0.0, _HeightMap_TexelSize.y),
                        0
                    ).r;
                dv /= 2.0 * _HeightMap_TexelSize.y;

                OUT.normal = TransformObjectToWorldNormal(normalize(float3(du, 1.0, dv)));
                OUT.positionHCS = TransformObjectToHClip(positionOS);
                OUT.positionWS = TransformObjectToWorld(positionOS);
                OUT.uv = TRANSFORM_TEX(IN.uv, _HeightMap);
                OUT.vertexHeight = height;
                return OUT;
            }

            half4 frag(Varyings IN, bool isFront : SV_IsFrontFace) : SV_Target
            {
                if(!isFront)
                {
                    return float4(0.8,0.8,0.8,1.0);
                }

                float3 actualNormal = normalize(IN.normal);

                float normalDot = max(0.0, dot(actualNormal, float3(0.0, 1.0, 0.0)));
                float isoColorLevel = (1.0 - normalDot) < 1.0e-4 ? 0.0 : (IN.vertexHeight / isoColorStep) + 1.0;
                float scaledFracPart = frac(isoColorLevel) * max(exp(-normalDot * 2.0), 0.1);

                int bottomLevel = (int)floor(isoColorLevel);
                int topLevel = (int)ceil(isoColorLevel);

                float3 bottomColor = levelColoring[bottomLevel]; 
                float3 topColor = levelColoring[topLevel];

                float3 vertexColorAtLevel = lerp(bottomColor, topColor, colorInterpolationRemap(scaledFracPart));
                float vertexSmoothnessAtLevel = levelSmoothness[bottomLevel] * _Smoothness;
                float vertexSpecularityAtLevel = levelSpecularity[bottomLevel] * _Specular;
                
                float4 shadowCoord = TransformWorldToShadowCoord(IN.positionWS);
                half shadowValue = MainLightRealtimeShadow(shadowCoord);
                
                float3 lightDirection = GetMainLight().direction;
                half3 lightColor = GetMainLight().color;

                half3 diffuseComponent = LightingLambert(lightColor, lightDirection, actualNormal);
                half3 specularComponent = LightingSpecular(lightColor, lightDirection, actualNormal, GetWorldSpaceNormalizeViewDir(IN.positionWS), vertexSpecularityAtLevel.xxxx, vertexSmoothnessAtLevel.xxxx); 
                
                half3 ambientComponent = ambientLight * vertexColorAtLevel; 

                return half4( ambientComponent + (diffuseComponent + specularComponent) * vertexColorAtLevel * shadowValue, 1.0);
            }

            ENDHLSL
        }

        Pass 
        {
            Name "Terrain shadow casting"
            Tags {"LightMode" = "ShadowCaster" "PassFlags" = "OnlyDirectional"}

            Cull Off

            HLSLPROGRAM

            #pragma vertex vert
            #pragma fragment frag

            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS _MAIN_LIGHT_SHADOWS_CASCADE _MAIN_LIGHT_SHADOWS_SCREEN

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Shadows.hlsl"

            struct Attributes
            {
                float4 positionOS   : POSITION;
                float2 uv           : TEXCOORD0;
                float3 normalOS     : NORMAL;
            };

            struct Varyings
            {
                float2 uv           : TEXCOORD0;
                float4 positionCS   : SV_POSITION;
                UNITY_VERTEX_INPUT_INSTANCE_ID
            };

            float4 GetShadowPositionHClip(Attributes input)
            {
                float3 positionWS = TransformObjectToWorld(input.positionOS.xyz);
                
                float3 normalWS = TransformObjectToWorldNormal(input.normalOS);
                float3 lightDirectionWS = GetMainLight().direction;
                float3 biasedPosition = ApplyShadowBias(positionWS, normalWS, lightDirectionWS);
            
                float4 positionCS = TransformWorldToHClip(biasedPosition);
                positionCS = ApplyShadowClamping(positionCS);
                return positionCS;
            }

            TEXTURE2D(_HeightMap);
            SAMPLER(sampler_HeightMap);
            
            CBUFFER_START(UnityPerMaterial)
            float4 _HeightMap_ST;
            float  _HeightScale;
            float4 _HeightMap_TexelSize;
            CBUFFER_END

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                ZERO_INITIALIZE(Varyings, OUT);

                UNITY_SETUP_INSTANCE_ID(IN);
                UNITY_TRANSFER_INSTANCE_ID(IN, OUT);

                Attributes inCopy = IN;
                UNITY_TRANSFER_INSTANCE_ID(IN, inCopy);

                float du = SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv + float2(_HeightMap_TexelSize.x,0.0),
                        0
                    ).r - SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv - float2(_HeightMap_TexelSize.x,0.0),
                        0
                    ).r;
                du /= 2.0 * _HeightMap_TexelSize.x;

                float dv = SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv + float2(0.0, _HeightMap_TexelSize.y),
                        0
                    ).r - SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv - float2(0.0, _HeightMap_TexelSize.y),
                        0
                    ).r;
                dv /= 2.0 * _HeightMap_TexelSize.y;

                inCopy.normalOS = normalize(float3(du, 1.0, dv));

                float height =
                    SAMPLE_TEXTURE2D_LOD(
                        _HeightMap,
                        sampler_HeightMap,
                        IN.uv,
                        0
                    ).r;
                float3 positionOS = IN.positionOS.xyz;
                positionOS.y = height * _HeightScale;
                inCopy.positionOS = float4(positionOS, 1.0);

                OUT.uv = TRANSFORM_TEX(IN.uv, _HeightMap);
                OUT.positionCS = GetShadowPositionHClip(inCopy);
                return OUT;
            }

            half4 frag(Varyings input) : SV_TARGET
            {
                UNITY_SETUP_INSTANCE_ID(input);

                #if defined(_ALPHATEST_ON)
                    Alpha(SampleAlbedoAlpha(input.uv, TEXTURE2D_ARGS(_BaseMap, sampler_BaseMap)).a, _BaseColor, _Cutoff);
                #endif

                #if defined(LOD_FADE_CROSSFADE)
                    LODFadeCrossFade(input.positionCS);
                #endif

                return 0;
            }

            ENDHLSL
        }
        
    }
}
