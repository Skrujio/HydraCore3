#include "integrator_pt.h"

#include <chrono>

void Integrator::RayTraceBlock(uint tid, float4* out_color, uint a_passNum)
{
  auto start = std::chrono::high_resolution_clock::now();
  #ifndef _DEBUG
  #pragma omp parallel for default(shared)
  #endif
  for(uint i=0;i<tid;i++)
    RayTrace(i, out_color);
  raytraceTime = float(std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count())/1000.f;
}

void Integrator::RayTrace(uint tid, float4* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  uint      rayFlags = 0;
  kernel_InitEyeRay3(tid, m_packedXY.data(), 
                     &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &rayFlags);

  for(uint depth = 0; depth < m_traceDepth; depth++)
  {
    float4 hitPart1, hitPart2;
    uint instId;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &instId, &rayFlags);
    if(isDeadRay(rayFlags))
      break;

    kernel_RayBounce(tid, depth, &hitPart1, &hitPart2,
                     &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &rayFlags);

    if(isDeadRay(rayFlags))
      break;
  }

//  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
//                        &accumColor);

  kernel_ContributeToImage3(tid, &accumColor, m_packedXY.data(),
                            out_color);
}

void Integrator::kernel_InitEyeRay3(uint tid, const uint* packedXY, 
                                   float4* rayPosAndNear, float4* rayDirAndFar,
                                   float4* accumColor,    float4* accumuThoroughput,
                                   uint* rayFlags) // 
{
  *accumColor        = make_float4(0,0,0,1);
  *accumuThoroughput = make_float4(1,1,1,1);
  //RandomGen genLocal = m_randomGens[tid];
  *rayFlags          = 0;

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  float3 rayDir = EyeRayDirNormalized((float(x))/float(m_winWidth), 
                                      (float(y))/float(m_winHeight), m_projInv);
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, FLT_MAX);
}

void Integrator::kernel_RayTrace2(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar,
                                 float4* out_hit1, float4* out_hit2, uint* out_instId, uint* rayFlags)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  const CRT_Hit hit   = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);

  if(hit.geomId != uint32_t(-1))
  {
    const float2 uv       = float2(hit.coords[0], hit.coords[1]);
    const float3 hitPos   = to_float3(rayPos) + (hit.t*0.999999f)*to_float3(rayDir); // set hit slightlyt closer to old ray origin to prevent self-interseaction and e.t.c bugs

    const uint triOffset  = m_matIdOffsets[hit.geomId];
    const uint vertOffset = m_vertOffset  [hit.geomId];
  
    const uint A = m_triIndices[(triOffset + hit.primId)*3 + 0];
    const uint B = m_triIndices[(triOffset + hit.primId)*3 + 1];
    const uint C = m_triIndices[(triOffset + hit.primId)*3 + 2];
  
    const float3 A_norm = to_float3(m_vNorm4f[A + vertOffset]);
    const float3 B_norm = to_float3(m_vNorm4f[B + vertOffset]);
    const float3 C_norm = to_float3(m_vNorm4f[C + vertOffset]);

    const float2 A_texc = m_vTexc2f[A + vertOffset];
    const float2 B_texc = m_vTexc2f[B + vertOffset];
    const float2 C_texc = m_vTexc2f[C + vertOffset];
      
    float3 hitNorm     = (1.0f - uv.x - uv.y)*A_norm + uv.y*B_norm + uv.x*C_norm;
    float2 hitTexCoord = (1.0f - uv.x - uv.y)*A_texc + uv.y*B_texc + uv.x*C_texc;
  
    // transform surface point with matrix and flip normal if needed
    //
    hitNorm = normalize(mul3x3(m_normMatrices[hit.instId], hitNorm));
    const float flipNorm = dot(to_float3(rayDir), hitNorm) > 0.001f ? -1.0f : 1.0f; // beware of transparent materials which use normal sign to identity "inside/outside" glass for example
    hitNorm = flipNorm * hitNorm;
  
    const uint midOriginal = m_matIdByPrimId[m_matIdOffsets[hit.geomId] + hit.primId];
    const uint midRemaped  = RemapMaterialId(midOriginal, hit.instId);

    *rayFlags  = packMatId(currRayFlags, midRemaped);
    *out_hit1  = to_float4(hitPos,  hitTexCoord.x); 
    *out_hit2  = to_float4(hitNorm, hitTexCoord.y);
    *out_instId = hit.instId;
  }
  else
    *rayFlags = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_OUT_OF_SCENE) ;
}

void Integrator::kernel_RayBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2,
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput,
                                  uint* rayFlags)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;

  const uint32_t matId = extractMatId(currRayFlags);

  // process surface hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  //const float3 ray_pos = to_float3(*rayPosAndNear);

  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);

  // process light hit case
  //
  if(m_materials[matId].mtype == MAT_TYPE_LIGHT_SOURCE)
  {
    const float2 texCoordT = mulRows2x4(m_materials[matId].row0[0], m_materials[matId].row1[0], hit.uv);
    const float3 texColor  = to_float3(m_textures[ m_materials[matId].texId[0] ]->sample(texCoordT));

    const float3 lightIntensity = to_float3(m_materials[matId].baseColor)*texColor;
    const uint lightId          = m_materials[matId].lightId;
    float lightDirectionAtten   = (lightId == 0xFFFFFFFF) ? 1.0f : dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f ? 1.0f : 0.0f; // TODO: read light info, gety light direction and e.t.c;


    float4 currAccumColor      = *accumColor;
    float4 currAccumThroughput = *accumThoroughput;

    currAccumColor.x += currAccumThroughput.x * lightIntensity.x * lightDirectionAtten;
    currAccumColor.y += currAccumThroughput.y * lightIntensity.y * lightDirectionAtten;
    currAccumColor.z += currAccumThroughput.z * lightIntensity.z * lightDirectionAtten;

    *accumColor = currAccumColor;
    *rayFlags   = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_HIT_LIGHT);
    return;
  }

  float4 shadeColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
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

  const BsdfSample matSam = MaterialSampleWhitted(matId, (-1.0f)*ray_dir, hit.norm, hit.uv);
  const float3 bxdfVal    = matSam.val;
  const float  cosTheta   = dot(matSam.dir, hit.norm);

  const float4 currThoroughput = *accumThoroughput;
  float4 currAccumColor        = *accumColor;

  currAccumColor.x += currThoroughput.x * shadeColor.x;
  currAccumColor.y += currThoroughput.y * shadeColor.y;
  currAccumColor.z += currThoroughput.z * shadeColor.z;

  *accumColor       = currAccumColor;
  *accumThoroughput = currThoroughput * cosTheta * to_float4(bxdfVal, 0.0f);

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.dir), 0.0f);
  *rayDirAndFar  = to_float4(matSam.dir, FLT_MAX);
  *rayFlags      = currRayFlags | matSam.flags;
}