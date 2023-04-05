#include "integrator_pt.h"
#include "include/crandom.h"

#include <chrono>
#include <string>

#include "Image2d.h"
using LiteImage::Image2D;
using LiteImage::Sampler;
using LiteImage::ICombinedImageSampler;
using namespace LiteMath;

void Integrator::InitRandomGens(int a_maxThreads)
{
  m_randomGens.resize(a_maxThreads);
  #pragma omp parallel for default(shared)
  for(int i=0;i<a_maxThreads;i++)
    m_randomGens[i] = RandomGenInit(i);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::kernel_InitEyeRay(uint tid, const uint* packedXY, float4* rayPosAndNear, float4* rayDirAndFar) // (tid,tidX,tidY,tidZ) are SPECIAL PREDEFINED NAMES!!!
{
  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;

  float3 rayDir = EyeRayDirNormalized((float(x)+0.5f)/float(m_winWidth), (float(y)+0.5f)/float(m_winHeight), m_projInv);
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, 
                  &rayPos, &rayDir);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
}

void Integrator::kernel_InitEyeRay2(uint tid, const uint* packedXY, 
                                   float4* rayPosAndNear, float4* rayDirAndFar,
                                   float4* accumColor,    float4* accumuThoroughput,
                                   RandomGen* gen, uint* rayFlags) // 
{
  *accumColor        = make_float4(0,0,0,1);
  *accumuThoroughput = make_float4(1,1,1,1);
  RandomGen genLocal = m_randomGens[tid];
  *rayFlags          = 0;

  const uint XY = packedXY[tid];

  const uint x = (XY & 0x0000FFFF);
  const uint y = (XY & 0xFFFF0000) >> 16;
  const float2 pixelOffsets = rndFloat2_Pseudo(&genLocal);

  float3 rayDir = EyeRayDirNormalized((float(x) + pixelOffsets.x)/float(m_winWidth), 
                                      (float(y) + pixelOffsets.y)/float(m_winHeight), m_projInv);
  float3 rayPos = float3(0,0,0);

  transform_ray3f(m_worldViewInv, &rayPos, &rayDir);
  
  *rayPosAndNear = to_float4(rayPos, 0.0f);
  *rayDirAndFar  = to_float4(rayDir, MAXFLOAT);
  *gen           = genLocal;
}


bool Integrator::kernel_RayTrace(uint tid, const float4* rayPosAndNear, float4* rayDirAndFar,
                                 Lite_Hit* out_hit, float2* out_bars)
{
  const float4 rayPos = *rayPosAndNear;
  const float4 rayDir = *rayDirAndFar ;

  CRT_Hit hit = m_pAccelStruct->RayQuery_NearestHit(rayPos, rayDir);
  
  Lite_Hit res;
  res.primId = hit.primId;
  res.instId = hit.instId;
  res.geomId = hit.geomId;
  res.t      = hit.t;

  float2 baricentrics = float2(hit.coords[0], hit.coords[1]);
 
  *out_hit  = res;
  *out_bars = baricentrics;
  return (res.primId != -1);
}

void Integrator::kernel_RayTrace2(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar,
                                 float4* out_hit1, float4* out_hit2, uint* rayFlags)
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
    const float3 hitPos   = to_float3(rayPos) + hit.t*to_float3(rayDir);

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
    const float flipNorm = dot(to_float3(rayDir), hitNorm) > 0.001f ? -1.0f : 1.0f; // be ware of transparent materials which use normal sign to identity "inside/outside" glass for example
    hitNorm = flipNorm * hitNorm;
  
    const uint midOriginal = m_matIdByPrimId[m_matIdOffsets[hit.geomId] + hit.primId];
    const uint midRemaped  = RemapMaterialId(midOriginal, hit.instId);

    *rayFlags  = packMatId(currRayFlags, midRemaped);
    *out_hit1  = to_float4(hitPos,  hitTexCoord.x); 
    *out_hit2  = to_float4(hitNorm, hitTexCoord.y); 
  }
  else
    *rayFlags = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_OUT_OF_SCENE) ;
}

void Integrator::kernel_PackXY(uint tidX, uint tidY, uint* out_pakedXY)
{
  const uint inBlockIdX = tidX % 8; // 8x8 blocks
  const uint inBlockIdY = tidY % 8; // 8x8 blocks
 
  const uint localIndex = inBlockIdY*8 + inBlockIdX;
  const uint wBlocks    = m_winWidth/8;

  const uint blockX     = tidX/8;
  const uint blockY     = tidY/8;
  const uint offset     = (blockX + blockY*wBlocks)*8*8 + localIndex;

  out_pakedXY[offset] = ((tidY << 16) & 0xFFFF0000) | (tidX & 0x0000FFFF);
}

void Integrator::kernel_RealColorToUint32(uint tid, float4* a_accumColor, uint* out_color)
{
  out_color[tid] = RealColorToUint32(*a_accumColor);
}

void Integrator::kernel_GetRayColor(uint tid, const Lite_Hit* in_hit, const uint* in_pakedXY, uint* out_color)
{ 
  const Lite_Hit lhit = *in_hit;
  if(lhit.geomId == -1)
  {
    out_color[tid] = 0;
    return;
  }

  const uint32_t matId = m_matIdByPrimId[m_matIdOffsets[lhit.geomId] + lhit.primId];
  const float4 mdata   = m_materials[matId].baseColor;
  const float3 color   = mdata.w > 0.0f ? clamp(float3(mdata.w,mdata.w,mdata.w), 0.0f, 1.0f) : to_float3(mdata);

  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;

  out_color[y*m_winWidth+x] = RealColorToUint32_f3(color); 
}


void Integrator::kernel_SampleLightSource(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar, 
                                         const float4* in_hitPart1, const float4* in_hitPart2, 
                                         const uint* rayFlags,  
                                         RandomGen* a_gen, float4* out_shadeColor)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const uint32_t matId = extractMatId(currRayFlags);
  const float3 ray_dir = to_float3(*rayDirAndFar);
  
  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);

  const float2 uv = rndFloat2_Pseudo(a_gen); // todo: select ligt id also ... 
  
  const float2 sampleOff = 2.0f*(float2(-0.5f,-0.5f) + uv)*m_light.size;
  const float3 samplePos = to_float3(m_light.pos) + float3(sampleOff.x, -1e-5f*std::max(m_light.size.x, m_light.size.y), sampleOff.y);

  //TODO: apply transform matrix 
  //samplePos = matrix3x3f_mult_float3(pMatrix, samplePos) + lightPos(pLight); // translate to world light position

  const float  hitDist   = sqrt(dot(hit.pos - samplePos, hit.pos - samplePos));

  const float3 shadowRayDir = normalize(samplePos - hit.pos); // explicitSam.direction;
  const float3 shadowRayPos = hit.pos + hit.norm*std::max(maxcomp(hit.pos), 1.0f)*5e-6f; // TODO: see Ray Tracing Gems, also use flatNormal for offset

  const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f), to_float4(shadowRayDir, hitDist*0.9995f));
  
  if(!inShadow && dot(shadowRayDir, to_float3(m_light.norm)) < 0.0f)
  {
    const float lightPickProb = 1.0f;
    const float pdfA          = 1.0f / (4.0f*m_light.size.x*m_light.size.y);
    const float cosVal        = std::max(dot(shadowRayDir, (-1.0f)*to_float3(m_light.norm)), 0.0f);
    const float lgtPdfW       = PdfAtoW(pdfA, hitDist, cosVal);
    const BsdfEval bsdfV      = MaterialEval(matId, shadowRayDir, (-1.0f)*ray_dir, hit.norm, hit.uv);
    const float cosThetaOut   = std::max(dot(shadowRayDir, hit.norm), 0.0f);
    const float misWeight     = (m_intergatorType == INTEGRATOR_MIS_PT) ? misWeightHeuristic(lgtPdfW, bsdfV.pdf) : 1.0f;

    if(cosVal <= 0.0f)
      *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 0.0f);
    else
      *out_shadeColor = to_float4((1.0f/lightPickProb)*(to_float3(m_light.intensity)*bsdfV.color/lgtPdfW)*cosThetaOut*misWeight, 0.0f);
  }
  else
    *out_shadeColor = float4(0.0f, 0.0f, 0.0f, 1.0f);
}

void Integrator::kernel_EvalPointLightSource(uint tid, const float4* rayPosAndNear, const float4* rayDirAndFar,
                                             const float4* in_hitPart1, const float4* in_hitPart2,
                                             const uint* rayFlags, RandomGen* a_gen, float4* out_shadeColor)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;

  const uint32_t matId = extractMatId(currRayFlags);
  const float3 ray_dir = to_float3(*rayDirAndFar);

  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);


  for(int light_id = 0; light_id < m_pointLights.size(); ++light_id)
  {
    float3 lightPos = to_float3(m_pointLights[light_id].pos);
    const float  hitDist   = sqrt(dot(hit.pos - lightPos, hit.pos - lightPos));

    const float3 shadowRayDir = normalize(lightPos - hit.pos);
    const float3 shadowRayPos = hit.pos + hit.norm * std::max(maxcomp(hit.pos), 1.0f) * 5e-6f; // TODO: see Ray Tracing Gems, also use flatNormal for offset

    const bool inShadow = m_pAccelStruct->RayQuery_AnyHit(to_float4(shadowRayPos, 0.0f),
                                                          to_float4(shadowRayDir, hitDist * 0.9995f));

    if(!inShadow)
    {
      const BsdfSample matSam = MaterialEvalWhitted(matId, (-1.0f)*ray_dir, hit.norm, hit.uv);
      *out_shadeColor += to_float4(to_float3(m_pointLights[light_id].intensity) * matSam.color / (hitDist * hitDist), 0.0f);
    }
    else
    {
      *out_shadeColor += float4(0.0f, 0.0f, 0.0f, 1.0f);
    }
  }
}

void Integrator::kernel_NextBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_shadeColor, 
                                  float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput, RandomGen* a_gen, MisData* misPrev, uint* rayFlags)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;
    
  const uint32_t matId = extractMatId(currRayFlags);

  // process surface hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  const float3 ray_pos = to_float3(*rayPosAndNear);
  
  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;
  
  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);
  
  const MisData prevBounce = *misPrev;
  const float   prevPdfW   = prevBounce.matSamplePdf;
  const float   prevPdfA   = (prevPdfW >= 0.0f) ? PdfWtoA(prevPdfW, length(ray_pos - hit.norm), prevBounce.cosTheta) : 1.0f;

  // process light hit case
  //
  if(m_materials[matId].brdfType == BRDF_TYPE_LIGHT_SOURCE)
  {
    const float lightIntensity = m_materials[matId].baseColor.w;
    float misWeight = 1.0f;
    if(m_intergatorType == INTEGRATOR_MIS_PT) 
    {
      if(bounce > 0)
      {
        const int lightId   = 0; // #TODO: get light id from material info
        const float lgtPdf  = LightPdfSelectRev(lightId)*LightEvalPDF(lightId, ray_pos, ray_dir, &hit);
        misWeight           = misWeightHeuristic(prevPdfW, lgtPdf);
        if (prevPdfW <= 0.0f) // specular bounce
          misWeight = 1.0f;
      }
    }
    else if(m_intergatorType == INTEGRATOR_SHADOW_PT && hasNonSpecular(currRayFlags))
      misWeight = 0.0f;
    
    float4 currAccumColor       = *accumColor;
    float4 currAccumThroughput = *accumThoroughput;
    
    const float lightDirectionAtten = dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f ? 1.0f : 0.0f;
    
    currAccumColor.x += currAccumThroughput.x * lightIntensity * misWeight * lightDirectionAtten;
    currAccumColor.y += currAccumThroughput.y * lightIntensity * misWeight * lightDirectionAtten;
    currAccumColor.z += currAccumThroughput.z * lightIntensity * misWeight * lightDirectionAtten;
    if(bounce > 0)
      currAccumColor.w *= prevPdfA;
    
    *accumColor = currAccumColor;
    *rayFlags   = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_HIT_LIGHT);
    return;
  }
  
  const float4 uv         = rndFloat4_Pseudo(a_gen);
  const BsdfSample matSam = MaterialSampleAndEval(matId, uv, (-1.0f)*ray_dir, hit.norm, hit.uv);
  const float3 bxdfVal    = matSam.color * (1.0f / std::max(matSam.pdf, 1e-20f));
  const float  cosTheta   = dot(matSam.direction, hit.norm);

  MisData nextBounceData;                   // remember current pdfW for next bounce
  nextBounceData.matSamplePdf = (matSam.flags & RAY_EVENT_S) != 0 ? -1.0f : matSam.pdf; //
  nextBounceData.cosTheta     = cosTheta;   //
  *misPrev = nextBounceData;                //

  if(m_intergatorType == INTEGRATOR_STUPID_PT)
  {
    *accumThoroughput *= cosTheta*to_float4(bxdfVal, 0.0f); 
  }
  else if(m_intergatorType == INTEGRATOR_SHADOW_PT || m_intergatorType == INTEGRATOR_MIS_PT)
  {
    const float4 currThoroughput = *accumThoroughput;
    const float4 shadeColor      = *in_shadeColor;
    float4 currAccumColor        = *accumColor;

    currAccumColor.x += currThoroughput.x * shadeColor.x;
    currAccumColor.y += currThoroughput.y * shadeColor.y;
    currAccumColor.z += currThoroughput.z * shadeColor.z;
    if(bounce > 0)
      currAccumColor.w *= prevPdfA;

    *accumColor       = currAccumColor;
    *accumThoroughput = currThoroughput*cosTheta*to_float4(bxdfVal, 0.0f); 
  }

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.direction), 0.0f); // todo: use flatNormal for offset
  *rayDirAndFar  = to_float4(matSam.direction, MAXFLOAT);
  *rayFlags      = currRayFlags | matSam.flags;
}


void Integrator::kernel_RayBounce(uint tid, uint bounce, const float4* in_hitPart1, const float4* in_hitPart2, const float4* in_shadeColor,
                                 float4* rayPosAndNear, float4* rayDirAndFar, float4* accumColor, float4* accumThoroughput, RandomGen* a_gen,
                                 uint* rayFlags)
{
  const uint currRayFlags = *rayFlags;
  if(isDeadRay(currRayFlags))
    return;

  const uint32_t matId = extractMatId(currRayFlags);

  // process surface hit case
  //
  const float3 ray_dir = to_float3(*rayDirAndFar);
  const float3 ray_pos = to_float3(*rayPosAndNear);

  const float4 data1 = *in_hitPart1;
  const float4 data2 = *in_hitPart2;

  SurfaceHit hit;
  hit.pos  = to_float3(data1);
  hit.norm = to_float3(data2);
  hit.uv   = float2(data1.w, data2.w);

  // process light hit case
  //
  if(m_materials[matId].brdfType == BRDF_TYPE_LIGHT_SOURCE)
  {
    const float lightIntensity = m_materials[matId].baseColor.w;

    float4 currAccumColor      = *accumColor;
    float4 currAccumThroughput = *accumThoroughput;

//    const float lightDirectionAtten = dot(to_float3(*rayDirAndFar), float3(0,-1,0)) < 0.0f ? 1.0f : 0.0f;
    const float lightDirectionAtten = 1.0f;

    currAccumColor.x += currAccumThroughput.x * lightIntensity * lightDirectionAtten;
    currAccumColor.y += currAccumThroughput.y * lightIntensity * lightDirectionAtten;
    currAccumColor.z += currAccumThroughput.z * lightIntensity * lightDirectionAtten;

    *accumColor = currAccumColor;
    *rayFlags   = currRayFlags | (RAY_FLAG_IS_DEAD | RAY_FLAG_HIT_LIGHT);
    return;
  }

  const float4 uv         = rndFloat4_Pseudo(a_gen);
  const BsdfSample matSam = MaterialEvalWhitted(matId, (-1.0f)*ray_dir, hit.norm, hit.uv);
  const float3 bxdfVal    = matSam.color;
  const float  cosTheta   = dot(matSam.direction, hit.norm);

//  *accumThoroughput *= cosTheta * to_float4(bxdfVal, 0.0f);
  const float4 currThoroughput = *accumThoroughput;
  const float4 shadeColor      = *in_shadeColor;
  float4 currAccumColor        = *accumColor;

  currAccumColor.x += currThoroughput.x * shadeColor.x;
  currAccumColor.y += currThoroughput.y * shadeColor.y;
  currAccumColor.z += currThoroughput.z * shadeColor.z;

  *accumColor       = currAccumColor;
  *accumThoroughput = currThoroughput * cosTheta * to_float4(bxdfVal, 0.0f);

  *rayPosAndNear = to_float4(OffsRayPos(hit.pos, hit.norm, matSam.direction), 0.0f);
  *rayDirAndFar  = to_float4(matSam.direction, MAXFLOAT);
  *rayFlags      = currRayFlags | matSam.flags;
}

void Integrator::kernel_HitEnvironment(uint tid, const uint* rayFlags, const float4* rayDirAndFar, const MisData* a_prevMisData, const float4* accumThoroughput,
                                       float4* accumColor)
{
  const uint currRayFlags = *rayFlags;
  if(!isOutOfScene(currRayFlags))
    return;
  
  const float4 envData  = GetEnvironmentColorAndPdf(to_float3(*rayDirAndFar));
  const float3 envColor = to_float3(envData)/envData.w;                         // explicitly account for pdf; when MIS will be enabled, need to deal with MIS weight also!

  if(m_intergatorType == INTEGRATOR_STUPID_PT)                                  // todo: when explicit sampling will be added, disable contribution here for 'INTEGRATOR_SHADOW_PT'
    *accumColor = (*accumThoroughput) * to_float4(envColor,0);
  else
    *accumColor += (*accumThoroughput) * to_float4(envColor,0);
}


void Integrator::kernel_ContributeToImage(uint tid, const float4* a_accumColor, const RandomGen* gen, const uint* in_pakedXY, float4* out_color)
{
  const uint XY = in_pakedXY[tid];
  const uint x  = (XY & 0x0000FFFF);
  const uint y  = (XY & 0xFFFF0000) >> 16;

  float4 color = *a_accumColor;
  //if(x == 511 && (y == 1024-340-1))
  //  color = float4(0,0,1,0);
 
  out_color[y*m_winWidth+x] += color;
  m_randomGens[tid] = *gen;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Integrator::PackXY(uint tidX, uint tidY)
{
  kernel_PackXY(tidX, tidY, m_packedXY.data());
}

void Integrator::CastSingleRay(uint tid, uint* out_color)
{
  float4 rayPosAndNear, rayDirAndFar;
  kernel_InitEyeRay(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar);

  Lite_Hit hit; 
  float2   baricentrics; 
  if(!kernel_RayTrace(tid, &rayPosAndNear, &rayDirAndFar, &hit, &baricentrics))
    return;
  
  kernel_GetRayColor(tid, &hit, m_packedXY.data(), out_color);
}

void Integrator::NaivePathTrace(uint tid, float4* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags = 0;
  kernel_InitEyeRay2(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &gen, &rayFlags);

  for(int depth = 0; depth < m_traceDepth+1; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
                       &accumColor);

  kernel_ContributeToImage(tid, &accumColor, &gen, m_packedXY.data(), 
                           out_color);
}

void Integrator::PathTrace(uint tid, float4* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen; 
  MisData   mis;
  uint      rayFlags = 0;
  kernel_InitEyeRay2(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &gen, &rayFlags);

  for(int depth = 0; depth < m_traceDepth; depth++) 
  {
    float4   shadeColor, hitPart1, hitPart2;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
    
    kernel_SampleLightSource(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags, 
                             &gen, &shadeColor);

    kernel_NextBounce(tid, depth, &hitPart1, &hitPart2, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &gen, &mis, &rayFlags);

    if(isDeadRay(rayFlags))
      break;
  }

  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
                       &accumColor);

  kernel_ContributeToImage(tid, &accumColor, &gen, m_packedXY.data(), 
                           out_color);
                           
}


void Integrator::RayTrace(uint tid, float4* out_color)
{
  float4 accumColor, accumThroughput;
  float4 rayPosAndNear, rayDirAndFar;
  RandomGen gen;
  uint      rayFlags = 0;
  kernel_InitEyeRay2(tid, m_packedXY.data(), &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &gen, &rayFlags);

  for(int depth = 0; depth < m_traceDepth; depth++)
  {
    float4 shadeColor, hitPart1, hitPart2;
    kernel_RayTrace2(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags);
    if(isDeadRay(rayFlags))
      break;

    kernel_EvalPointLightSource(tid, &rayPosAndNear, &rayDirAndFar, &hitPart1, &hitPart2, &rayFlags,
                             &gen, &shadeColor);

    kernel_RayBounce(tid, depth, &hitPart1, &hitPart2, &shadeColor,
                      &rayPosAndNear, &rayDirAndFar, &accumColor, &accumThroughput, &gen, &rayFlags);
    if(isDeadRay(rayFlags))
      break;
  }

//  kernel_HitEnvironment(tid, &rayFlags, &rayDirAndFar, &mis, &accumThroughput,
//                        &accumColor);

  kernel_ContributeToImage(tid, &accumColor, &gen, m_packedXY.data(),
                           out_color);
}
