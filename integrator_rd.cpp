#include "integrator_pt.h"

#include <chrono>

void Integrator::RayDiffTraceBlock(uint tid, float4* out_color, uint a_passNum) {
  auto start = std::chrono::high_resolution_clock::now();
  #ifndef _DEBUG
  #pragma omp parallel for default(shared)
  #endif
  for(uint i=0;i<tid;i++)
    RayDiffTrace(i, out_color);
  raydiffTime = float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count())/1000.f;
}

Integrator::RayData Integrator::GetRayData(float3 rayPos, float3 rayDir) {
  transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
  
  return {to_float4(rayPos, 0.0f), to_float4(rayDir, FLT_MAX), 0};
}

void Integrator::InitRayData(RayData& ray, float3 rayPos, uint x, uint y) {
  float3 rayDir = EyeRayDirNormalized((float(x))/float(m_winWidth), 
                                      (float(y))/float(m_winHeight), m_projInv);

  ray = GetRayData(rayPos, rayDir);
}

void Integrator::InitEyeRayDiff(uint tid, RayDiffData& ray, AccumData& accum) {
  accum.Color        = make_float4(0,0,0,1);
  accum.Throughput = make_float4(1,1,1,1);

  const uint XY = m_packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  float3 rayPos = float3(0,0,0);

  InitRayData(ray.prime, rayPos, x, y);
  InitRayData(ray.dx, rayPos, x + 1, y);
  InitRayData(ray.dy, rayPos, x, y + 1);
}

void Integrator::RayTrace2Wrapper(uint tid, RayData& ray, HitData& hit) {
  kernel_RayTrace2(tid, &ray.PosAndNear, &ray.DirAndFar,
      &hit.Part1, &hit.Part2, &hit.instId, &ray.Flags);
}

SurfaceHit Integrator::UnpackHitData(HitData hit) {
  return {to_float3(hit.Part1), to_float3(hit.Part2), float2(hit.Part1.w, hit.Part2.w)};
}

void Integrator::ProcessLightSource(uint32_t matId, RayData& ray, SurfaceHit hit, AccumData& accum) {
  const float2 texCoordT = mulRows2x4(m_materials[matId].row0[0], m_materials[matId].row1[0], hit.uv);
  const float3 texColor  = to_float3(m_textures[ m_materials[matId].texId[0] ]->sample(texCoordT));

  const float3 lightIntensity = to_float3(m_materials[matId].baseColor)*texColor;
  const uint lightId          = m_materials[matId].lightId;
  float lightDirectionAtten   = (lightId == 0xFFFFFFFF) ? 1.0f : dot(to_float3(ray.DirAndFar), float3(0,-1,0)) < 0.0f ? 1.0f : 0.0f; // TODO: read light info, gety light direction and e.t.c;


  float4 currAccumColor      = accum.Color;
  float4 currAccumThroughput = accum.Throughput;

  currAccumColor.x += currAccumThroughput.x * lightIntensity.x * lightDirectionAtten;
  currAccumColor.y += currAccumThroughput.y * lightIntensity.y * lightDirectionAtten;
  currAccumColor.z += currAccumThroughput.z * lightIntensity.z * lightDirectionAtten;

  accum.Color = currAccumColor;
  ray.Flags   = ray.Flags | (RAY_FLAG_IS_DEAD | RAY_FLAG_HIT_LIGHT);
}

float4 Integrator::ProcessShaderColor(float4 shadeColor, SurfaceHit hit, uint matId, float3 ray_dir) {
  for(uint lightId = 0; lightId < m_lights.size(); ++lightId)
  {
    const float3 lightPos = to_float3(m_lights[lightId].pos);
    const float hitDist   = sqrt(dot(hit.pos - lightPos, hit.pos - lightPos));

    const float3 shadowRayDir = normalize(lightPos - hit.pos);
    const float3 shadowRayPos = hit.pos + hit.norm * std::max(maxcomp(hit.pos), 1.0f) * 5e-6f; // TODO: see Ray Tracing Gems, also use flatNormal for offset

    const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist * 0.9995f));

    if(!inShadow && dot(shadowRayDir, to_float3(m_lights[lightId].norm)) < 0.0f)
    {
      const float3 matSamColor = MaterialEvalWhitted(matId, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.uv);
      const float cosThetaOut  = std::max(dot(shadowRayDir, hit.norm), 0.0f);
      shadeColor += to_float4(to_float3(m_lights[lightId].intensity) * matSamColor*cosThetaOut / (hitDist * hitDist), 0.0f);
    }
  }
  return shadeColor;
}

void Integrator::RayDiffBounce(uint tid, uint depth, RayDiffData& ray, HitDiffData& packedHit, AccumData& accum) {
  // const uint currRayFlags = *rayFlags;
  if(isDeadRay(ray.prime.Flags))
    return;

  const uint32_t matId = extractMatId(ray.prime.Flags);

  // process surface hit case
  //
  const float3 ray_dir = to_float3(ray.prime.DirAndFar);

  SurfaceHit hit = UnpackHitData(packedHit.prime);

  // process light hit case
  //
  if(m_materials[matId].mtype == MAT_TYPE_LIGHT_SOURCE)
  {
    ProcessLightSource(matId, ray.prime, hit, accum);
    return;
  }

  float4 shadeColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
  shadeColor = ProcessShaderColor(shadeColor, hit, matId, ray_dir);

  const BsdfSample matSam = MaterialSampleWhitted(matId, (-1.0f)*ray_dir, hit.norm, hit.uv);
  const float3 bxdfVal    = matSam.val;
  const float  cosTheta   = dot(matSam.dir, hit.norm);

  accum.Color.x += accum.Throughput.x * shadeColor.x;
  accum.Color.y += accum.Throughput.y * shadeColor.y;
  accum.Color.z += accum.Throughput.z * shadeColor.z;

  accum.Throughput *= cosTheta * to_float4(bxdfVal, 0.0f);

  ray.prime.PosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.dir), 0.0f);
  ray.prime.DirAndFar  = to_float4(matSam.dir, FLT_MAX);
  ray.prime.Flags      |= matSam.flags;
}

void Integrator::RayDiffTrace(uint tid, float4* out_color)
{
  AccumData accum;
  RayDiffData ray;

  InitEyeRayDiffData(tid, ray, accum);

  for(uint depth = 0; depth < m_traceDepth; depth++)
  {
    HitDiffData hit;

    RayTrace2Wrapper(tid, ray.prime, hit.prime);
    RayTrace2Wrapper(tid, ray.dx, hit.dx);
    RayTrace2Wrapper(tid, ray.dy, hit.dy);

    RayDiffBounce(tid, depth, ray, hit, accum);

    // kernel_RayBounce(tid, depth, &hitPart1, &hitPart2,
    //                  &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &rayFlags);

    if(isDeadRay(ray.prime.Flags))
      break;
  }

  kernel_ContributeToImage3(tid, &accum.Color, m_packedXY.data(),
                            out_color);
}