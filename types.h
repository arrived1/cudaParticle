#pragma once

#include <vector>

struct float2
{
    float2(float x_ = 0, float y_ = 0)
    : x(x_), y(y_)
    {}

    float2 operator+ (const float2 &f) {
        float2 result;
        result.x = x + f.x;
        result.y = y + f.y;
        return result;
    };

    float2 operator- (const float2 &f) {
        float2 result;
        result.x = x - f.x;
        result.y = y - f.y;
        return result;
    };

    float2 operator* (const float &f) {
        float2 result;
        result.x = x * f;
        result.y = y * f;
        return result;
    };

    float2& operator= (float2 &f) {
        x = f.x;
        y = f.y;
        return *this;
    }

    float x;
    float y;
};

struct float3
{
    float3(float x_ = 0, float y_ = 0, float z_ = 0)
    : x(x_), y(y_), z(z_)
    {}

    float x;
    float y;
    float z;

    float3& operator= (float3 &f) {
        x = f.x;
        y = f.y;
        z = f.z;
        return *this;
    }
};

struct float4 : public float3
{
    float4(float x_ = 0, float y_ = 0, float z_ = 0, float w_ = 0)
    : float3(x_, y_, z_),
    w(w_)
    {}

    float4(float3 f3_, float w_)
        : float3(f3_.x, f3_.y, f3_.z),
        w(w_)
    {}

    float w;
};

std::ostream & operator<<(std::ostream &out, const float2 &f)
{
    return out << "(" << f.x << ", " << f.y << ")";
}

std::ostream & operator<<(std::ostream &out, const float3 &f)
{
    return out << "(" << f.x << ", " << f.y << ", " << f.z << ")";
}


typedef std::vector<float3> vector3;
typedef std::vector<float4> vector4;

enum ParticleState
{
    inEmiter = 0,
    inSafeState,
    inBox
};
