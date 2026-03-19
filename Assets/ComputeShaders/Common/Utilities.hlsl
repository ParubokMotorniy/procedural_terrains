#define PI 3.141592654
#define TWO_PI 6.283185307
#define EPS 1.0e-7

int2 terrainWrap(int2 a, int2 b)
{
    return (a % b + b) % b;
}

// only makes sense if height is normalized. 
float computeSoftnessCoefficient(float2 localGradient, float localHeight, float baseHardness)
{
    const static float heightWeight = 0.25;
    const static float gradientWeight = 1.0 - heightWeight;
    return clamp(1.0 - (localHeight * heightWeight) * (exp(-length(localGradient)) * gradientWeight), baseHardness, 1.0);
}

float2 computeGradientAtPoint(RWTexture2D<float> tex, uint2 dimensions, int2 texelOfInterest)
{
    float dX = (tex[terrainWrap(texelOfInterest + int2(1, 0), dimensions)] - tex[terrainWrap(texelOfInterest - int2(1, 0), dimensions)]) / 2.0;
    float dZ = (tex[terrainWrap(texelOfInterest + int2(0, 1), dimensions)] - tex[terrainWrap(texelOfInterest - int2(0, 1), dimensions)]) / 2.0;
    return float2(dX, dZ);
}

uint index2dTo1d(uint2 gridDimensions, uint2 index2D)
{
    return index2D.x * gridDimensions.y + index2D.y;
}

//samples from normal distribution. u must be uniformly distributed, though.
float sampleGaussBoxMuller(float2 u)
{
    const float a = sqrt(-2.0f * log(1.0f - u.x));
    const float b = TWO_PI * u.y;

    // return float2(cos(b), sin(b)) * a;
    return cos(b) * a;
}

//(c) Jarzynski and Olano
uint pcgHash(uint v)
{
    v = v * 747796405u + 2891336453u;
    uint word = ((v >> ((v >> 28u) + 4u)) ^ v) * 277803737u;
    return (word >> 22u) ^ word;
}

float u01FromUint(uint x)
{
    return (x + 1.0) * (1.0 / 4294967296.0);
}

uint seedFromXYPass(uint x, uint y, int passId)
{
    uint v = x * 0x9E3779B9u ^ y * 0x85EBCA6Bu ^ passId * 0xC2B2AE35u;
    return v;
}

//TODO: may want to search for non-quantizing solutions
//https://www.shadertoy.com/view/lXjGRw
// return value 0~1 but tends toward central values
float pinkNoise2D(uint2 x)
{
    float sum_y = 0.0, sum_w = 0.0, w = 1.0;
    x *= 2u; // Adds a layer of white noise
    for (int i = 0; i < 10; ++i)
    {
        uint p = (x.x/2u) ^ ((x.y/2u) << 16);
        sum_y += w * (float(pcgHash(p)) / 4294967296.0);
        sum_w += w;
        x = (x + 1u) / 2u; // offset for consistent variation per step
    }
    return max(0.0, sum_y / sum_w);
}

float accurateSum(float a, float b, out float error)
{
    float sum = a + b;
    float bs = sum - a;
    float as = sum - bs;
    error = (b - bs) + (sum - as);
    return sum;
}