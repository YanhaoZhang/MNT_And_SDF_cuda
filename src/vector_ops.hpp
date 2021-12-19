// ---------------------------------------------------------
// Author: Yanhao Zhang, University of Technoligy Sydney, 2020.04.02
// vector operation using float2 float3 float4
// based on surfelwarp (file with same name)
// ---------------------------------------------------------





// #pragma once
#ifndef CU_MEX_VECTOR_OPS
#define CU_MEX_VECTOR_OPS

// #include "common/common_types.h"
#include <vector_functions.h>

namespace aortawarp {

	/* Compute the squared norm of a vector
	 */
    __host__ __device__ __forceinline__
    float squared_norm(const float4 &vec) {
        return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z + vec.w * vec.w;
    }
	__host__ __device__ __forceinline__
		float squared_norm(const float3 &vec) {
		return vec.x * vec.x + vec.y * vec.y + vec.z * vec.z;
	}
	__host__ __device__ __forceinline__
	float squared_norm(const float2& vec){
		return vec.x * vec.x + vec.y * vec.y;
    }

	// compute norm of a vector
    __host__ __device__ __forceinline__
    float norm(const float4& vec) {
        return sqrtf(squared_norm(vec));
    }
	__host__ __device__ __forceinline__
		float norm(const float3& vec) {
		return sqrtf(squared_norm(vec));
	}
	__host__ __device__ __forceinline__
		float norm(const float2& vec) {
		return sqrtf(squared_norm(vec));
	}

	//The squared distance between two vector, only care xyz component
	__host__ __device__ __forceinline__
	float squared_distance(const float3& v3, const float3& v4) {
		return (v3.x - v4.x) * (v3.x - v4.x) + (v3.y - v4.y) * (v3.y - v4.y) +(v3.z - v4.z) * (v3.z - v4.z);
	}

	__host__ __device__ __forceinline__
	float squared_distance(const float2& v3, const float2& v4) {
		return (v3.x - v4.x) * (v3.x - v4.x) + (v3.y - v4.y) * (v3.y - v4.y);
	}

	/* Compute the inversed normal of a vector
	*/
	__host__ __device__ __forceinline__ float norm_inversed(const float4& vec) {
		return 1.0f / sqrt(squared_norm(vec));
	}
	__host__ __device__  __forceinline__ float norm_inversed(const float3& vec) {
		return 1.0f / sqrt(squared_norm(vec));
	}
	__host__ __device__ __forceinline__ float norm_inversed(const float2& vec) {
		return 1.0f / sqrt(squared_norm(vec));
	}


	// normalize a vector
	// note: different from normalized
    __host__ __device__ __forceinline__ void normalize(float4 &vec) {
        const float inv_vecnorm = 1.0f / sqrtf(squared_norm(vec));
        vec.x *= inv_vecnorm;
        vec.y *= inv_vecnorm;
        vec.z *= inv_vecnorm;
        vec.w *= inv_vecnorm;
    }
	__host__ __device__ __forceinline__ void normalize(float3 &vec) {
        const float inv_vecnorm = 1.0f / sqrtf(squared_norm(vec));
        vec.x *= inv_vecnorm;
        vec.y *= inv_vecnorm;
        vec.z *= inv_vecnorm;
    }
	__host__ __device__ __forceinline__ void normalize(float2 &vec) {
        const float inv_vecnorm = 1.0f / sqrtf(squared_norm(vec));
        vec.x *= inv_vecnorm;
        vec.y *= inv_vecnorm;
    }


	/* Return the normlized vector while keeping the original copy
	*/
	__host__ __device__ __forceinline__ float4 normalized(const float4 &vec) {
		const float inv_vecnorm = norm_inversed(vec);
		const float4 normalized_vec = make_float4(
			vec.x * inv_vecnorm, vec.y * inv_vecnorm, 
			vec.z * inv_vecnorm, vec.w * inv_vecnorm
		);
		return normalized_vec;
	}

	__host__ __device__ __forceinline__ float3 normalized(const float3 &vec) {
		const float inv_vecnorm = norm_inversed(vec);
		const float3 normalized_vec = make_float3(
			vec.x * inv_vecnorm, 
			vec.y * inv_vecnorm,
			vec.z * inv_vecnorm
		);
		return normalized_vec;
	}

	__host__ __device__ __forceinline__ float2 normalized(const float2 &vec) {
		const float inv_vecnorm = norm_inversed(vec);
		const float2 normalized_vec = make_float2(
			vec.x * inv_vecnorm, 
			vec.y * inv_vecnorm
		);
		return normalized_vec;
	}


	/* Check if the vertex or normal is zero
	*/
	__host__ __device__ __forceinline__ bool
	is_zero_vertex(const float4& v) {
		return fabsf(v.x) < 1e-3 && fabsf(v.y) < 1e-3 && fabsf(v.z) < 1e-3;
    }
	__host__ __device__ __forceinline__ bool
	is_zero_vertex(const float3& v) {
		return fabsf(v.x) < 1e-3 && fabsf(v.y) < 1e-3 && fabsf(v.z) < 1e-3;
    }
	__host__ __device__ __forceinline__ bool
	is_zero_vertex(const float2& v) {
		return fabsf(v.x) < 1e-3 && fabsf(v.y) < 1e-3;
    }



	/* Operator for add float 3
	*/
	__host__ __device__ __forceinline__ float3 
		operator+(const float3& vec, const float& scalar)
	{
		return make_float3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
	}

	__host__ __device__ __forceinline__ float3
		operator+(const float& scalar, const float3& vec)
	{
		return make_float3(vec.x + scalar, vec.y + scalar, vec.z + scalar);
	}

	__host__ __device__ __forceinline__ float3
		operator+(const float3& vec_0, const float3& vec_1)
	{
		return make_float3(vec_0.x + vec_1.x, vec_0.y + vec_1.y, vec_0.z + vec_1.z);
	}

	// float 2
	__host__ __device__ __forceinline__ float2 
		operator+(const float2& vec, const float& scalar)
	{
		return make_float2(vec.x + scalar, vec.y + scalar);
	}

	__host__ __device__ __forceinline__ float2
		operator+(const float& scalar, const float2& vec)
	{
		return make_float2(vec.x + scalar, vec.y + scalar);
	}

	__host__ __device__ __forceinline__ float2
		operator+(const float2& vec_0, const float2& vec_1)
	{
		return make_float2(vec_0.x + vec_1.x, vec_0.y + vec_1.y);
	}



	/* Operations for subtraction
	*/
	__host__ __device__ __forceinline__ float3
		operator-(const float3& vec_0, const float3& vec_1) {
		return make_float3(vec_0.x - vec_1.x, vec_0.y - vec_1.y, vec_0.z - vec_1.z);
	}

	__host__ __device__ __forceinline__ float4
		operator-(const float4& vec_0, const float4& vec_1) {
		return make_float4(vec_0.x - vec_1.x, vec_0.y - vec_1.y, vec_0.z - vec_1.z, vec_0.w - vec_1.w);
	}

	__host__ __device__ __forceinline__ float2
		operator-(const float2& vec_0, const float2& vec_1) {
		return make_float2(vec_0.x - vec_1.x, vec_0.y - vec_1.y);
	}


	/* Operations for product
	*/
	__host__ __device__ __forceinline__ float3
		operator*(const float& v, const float3& v1)
	{
		return make_float3(v * v1.x, v * v1.y, v * v1.z);
	}
	__host__ __device__ __forceinline__ float3
		operator*(const float3& v1, const float& v)
	{
		return make_float3(v1.x * v, v1.y * v, v1.z * v);
	}
	__host__ __device__ __forceinline__ float2
		operator*(const float& v, const float2& v1)
	{
		return make_float2(v * v1.x, v * v1.y);
	}
	__host__ __device__ __forceinline__ float2
		operator*(const float2& v1, const float& v)
	{
		return make_float2(v1.x * v, v1.y * v);
	}


	/* Operations for +=, *=
	*/
	__host__ __device__ __forceinline__ float3&
		operator*=(float3& vec, const float& v)
	{
		vec.x *= v;
		vec.y *= v;
		vec.z *= v;
		return vec;
	}
	__host__ __device__ __forceinline__ float4&
		operator*=(float4& vec, const float& v)
	{
		vec.x *= v;
		vec.y *= v;
		vec.z *= v;
		vec.w *= v;
		return vec;
	}

	__host__ __device__ __forceinline__ float3&
		operator+=(float3& vec_0, const float3& vec_1)
	{
		vec_0.x += vec_1.x;
		vec_0.y += vec_1.y;
		vec_0.z += vec_1.z;
		return vec_0;
	}

	/* Operator for negative
	 */
	__host__ __device__ __forceinline__ float3 operator-(const float3& vec)
    {
    	float3 negative_vec;
		negative_vec.x = - vec.x;
		negative_vec.y = - vec.y;
		negative_vec.z = - vec.z;
		return negative_vec;
    }

	__host__ __device__ __forceinline__ float2 operator-(const float2& vec)
    {
    	float2 negative_vec;
		negative_vec.x = - vec.x;
		negative_vec.y = - vec.y;
		return negative_vec;
    }


	/* The dot and cross product
	*/
	__host__ __device__ __forceinline__ float 
	dot(const float3& v1, const float3& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}
	
	__host__ __device__ __forceinline__ float
	dot(const float4& v1, const float4& v2) {
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
	}

	__host__ __device__ __forceinline__ float
	dot(const float2& v1, const float2& v2) {
		return v1.x * v2.x + v1.y * v2.y;
	}

	
	//Only use the first 3 elements of vec4
	__host__ __device__ __forceinline__ float 
	dotxyz(const float4& v1, const float4& v2)
	{
		return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
	}
	__host__ __device__ __forceinline__ float
		dotxyz(const float3& v1, const float4& v2)
	{
		return v1.x * v2.x + v1.y*v2.y + v1.z*v2.z;
	}


	__host__ __device__ __forceinline__ float3
		cross(const float3& v1, const float3& v2)
	{
		return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}

	__host__ __device__ __forceinline__ float3
		cross_xyz(const float3& v1, const float4& v2)
	{
		return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
	}
	
	//The sum of the vector
	__host__ __device__ __forceinline__ float fabsf_sum(const float3& vec) {
		return fabsf(vec.x) + fabsf(vec.y) + fabsf(vec.z);
	}
	__host__ __device__ __forceinline__ float fabsf_sum(const float4& vec) {
		return fabsf(vec.x) + fabsf(vec.y) + fabsf(vec.z) + fabsf(vec.w);
	}
	__host__ __device__ __forceinline__ float fabsf_sum(const float2& vec) {
		return fabsf(vec.x) + fabsf(vec.y);
	}

	//The L1 error between two vector
	__host__ __device__ __forceinline__ float fabsf_diff_xyz(
		const float3& vec_0, 
		const float4& vec_1
	) {
	    return fabsf(vec_0.x - vec_1.x) + fabsf(vec_0.y - vec_1.y) + fabsf(vec_0.z - vec_1.z);
    }





}
#endif