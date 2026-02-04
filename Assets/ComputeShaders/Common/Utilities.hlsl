#define TWO_PI 6.283185307

int2 terrainWrap(int2 a, int2 b)
{
    return (a % b + b) % b;
}

uint index2dTo1d(uint2 gridDimensions, uint2 index2D)
{
    return index2D.x * gridDimensions.y + index2D.y;
}

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
    return sum_y / sum_w;
}