#pragma once
#include "cglobals.h"
#include "crandom.h"
#include "cmaterial.h"


typedef struct RefractResultT
{
  float3 rayDir;
  bool   success;
  float  eta;

}RefractResult;


static inline RefractResult myRefract(const float3 a_rayDir, float3 a_normal, 
  const float a_inputIor, const float a_outputIor)
{
  RefractResult res;
  res.eta        = a_inputIor / a_outputIor; 
  float cosTheta = dot(a_normal, a_rayDir) * (-1.0f);

  if (cosTheta < 0.0f)
  {
    cosTheta = cosTheta * (-1.0f);
    a_normal = a_normal * (-1.0f);
    res.eta = 1.0f / res.eta;
  }

  const float dotVN = cosTheta * (-1.0f);
  const float k     = 1.0f - res.eta * res.eta * (1.0f - cosTheta * cosTheta);

  if (k > 0.0f)
  {
    res.rayDir  = normalize(res.eta * a_rayDir + (res.eta * cosTheta - sqrt(k)) * a_normal); // refract
    res.success = true;
  }
  else
  {
    res.rayDir  = normalize((a_normal * dotVN * (-2.0f)) + a_rayDir);            // internal reflect
    res.success = false;
    res.eta     = 1.0f;
  }

  return res;
}


static inline float3 GgxVndf(float3 wo, float roughness, float u1, float u2)
{
  // -- Stretch the view vector so we are sampling as though
  // -- roughnessTransp==1
  const float3 v = normalize(make_float3(wo.x * roughness, wo.y * roughness, wo.z));

  // -- Build an orthonormal basis with v, t1, and t2
  const float3 XAxis = make_float3(1.0f, 0.0f, 0.0f);
  const float3 ZAxis = make_float3(0.0f, 0.0f, 1.0f);
  const float3 t1 = (v.z < 0.999f) ? normalize(cross(v, ZAxis)) : XAxis;
  const float3 t2 = cross(t1, v);

  // -- Choose a point on a disk with each half of the disk weighted
  // -- proportionally to its projection onto direction v
  const float a = 1.0f / (1.0f + v.z);
  const float r = sqrt(u1);
  const float phi = (u2 < a) ? (u2 / a) * M_PI : M_PI + (u2 - a) / (1.0f - a) * M_PI;
  const float p1 = r * cos(phi);
  const float p2 = r * sin(phi) * ((u2 < a) ? 1.0f : v.z);

  // -- Calculate the normal in this stretched tangent space
  const float3 n = p1 * t1 + p2 * t2 + sqrt(fmax(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;

  // -- unstretch and normalize the normal
  return normalize(make_float3(roughness * n.x, roughness * n.y, fmax(0.0f, n.z)));
}


static inline float SmithGGXMasking(const float dotNV, float roughSqr)
{
  const float denomC = sqrt(roughSqr + (1.0f - roughSqr) * dotNV * dotNV) + dotNV;
  return 2.0f * dotNV / fmax(denomC, 1e-6f);
}


static inline float SmithGGXMaskingShadowing(const float dotNL, const float dotNV, float roughSqr)
{
  const float denomA = dotNV * sqrt(roughSqr + (1.0f - roughSqr) * dotNL * dotNL);
  const float denomB = dotNL * sqrt(roughSqr + (1.0f - roughSqr) * dotNV * dotNV);
  return 2.0f * dotNL * dotNV / fmax(denomA + denomB, 1e-6f);
}


static inline void refractionGlassSampleAndEval(const float3 a_colorTransp, const float a_inputIor, 
  const float a_outputIor, const float a_roughTransp, const float3 a_normal, const float3 a_normal2,
  const float4 a_rands, const float3 a_rayDir, BsdfSample* a_pRes)
{
  const float  roughSqr = a_roughTransp * a_roughTransp;

  bool   spec = true;
  float  Pss  = 1.0f;                          // Pass single-scattering.
  float3 Pms  = make_float3(1.0f, 1.0f, 1.0f); // Pass multi-scattering
  a_pRes->pdf = 1.0f;

  RefractResult refrData = myRefract(a_rayDir, a_normal2, a_inputIor, a_outputIor);

  if (a_roughTransp > 1e-5f)
  {
    spec                 = false;
    const float eta      = a_inputIor / a_outputIor;

    float3 nx, ny, nz = a_normal;
    CoordinateSystem(nz, &nx, &ny);

    const float3 wo      = make_float3(-dot(a_rayDir, nx), -dot(a_rayDir, ny), -dot(a_rayDir, nz));
    const float3 wh      = GgxVndf(wo, roughSqr, a_rands.x, a_rands.y);             // New sampling Heitz 2017
    const float  dotWoWh = dot(wo, wh);
    float3       newDir;

    const float radicand = 1.0f + eta * eta * (dotWoWh * dotWoWh - 1.0f);
    if (radicand > 0.0f)
    {
      newDir           = (eta * dotWoWh - sqrt(radicand)) * wh - eta * wo;    // refract        
      refrData.success = true;
      refrData.eta     = eta;
    }
    else
    {
      newDir           = 2.0f * dotWoWh * wh - wo;                            // internal reflect 
      refrData.success = false;
      refrData.eta     = 1.0f;
    }

    refrData.rayDir   = normalize(newDir.x * nx + newDir.y * ny + newDir.z * nz);    // back to normal coordinate system

    const float3 v    = (-1.0f) * a_rayDir;
    const float3 l    = refrData.rayDir;
    const float3 n    = a_normal;
    const float dotNV = fabs(dot(n, v));
    const float dotNL = fabs(dot(n, l));

    // Fresnel is not needed here, because it is used for the blend.    
    const float G1    = SmithGGXMasking(dotNV, roughSqr);
    const float G2    = SmithGGXMaskingShadowing(dotNL, dotNV, roughSqr);

    // Abbreviated formula without PDF.
    Pss *= G2 / fmax(G1, 1e-6f);

    // Complete formulas with PDF, if we ever make an explicit strategy for glass.
    //const float3 h    = normalize(v + l); // half vector.
    //const float dotNH = fabs(dot(n, h));
    //const float dotHV = fabs(dot(h, v));
    //const float dotHL = fabs(dot(v, l));
    //const float D   = GGX_Distribution(dotNH, roughSqr);
    //float etaI = 1.0f;
    //float etaO = a_ior;
    //if (eta > 1.0f)
    //{      
    //  etaI = a_ior;
    //  etaO = 1.0f;
    //}
    //const float eq1 = (dotHV * dotHL) / fmax(dotNV * dotNL, 1e-6f);
    //const float eq2 = etaO * etaO * G2 * D / fmax(pow(etaI * dotHV + etaO * dotHL, 2), 1e-6f);
    //Pss             = eq1 * eq2 * dotNL; // dotNL is here to cancel cosMult at the end.
    //const float Dv    = D * G1 * dotHV / fmax(dotNV, 1e-6f);
    //const float jacob = etaO * etaO * dotHL / fmax(pow(etaI * dotHV + etaO * dotHL, 2), 1e-6f);
    //a_pRes->pdf        = Dv * jacob;

    // Pass multi-scattering. (not supported yet)
    //Pms = GetMultiscatteringFrom3dTable(a_globals->m_essTranspTable, a_roughTransp, dotNV, 1.0f / eta, 64, 64, 64, color);
  }

  const float cosThetaOut = dot(refrData.rayDir, a_normal);
  const float cosMult     = 1.0f / fmax(fabs(cosThetaOut), 1e-6f);

  a_pRes->direction = refrData.rayDir;

  // only camera paths are multiplied by this factor, and etas are swapped because radiance
  // flows in the opposite direction. See SmallVCM and or Veach adjoint bsdf.
  const bool a_isFwdDir       = true; // It should come from somewhere on top, but it has not yet been implemented, we are making a fake.
  const float adjointBtdfMult = a_isFwdDir ? 1.0f : (refrData.eta * refrData.eta);

  if (refrData.success) a_pRes->color = a_colorTransp * adjointBtdfMult * Pss * Pms * cosMult; //refrData.rayDir * 4.0f * cosMult;
  else                  a_pRes->color = make_float3(1.0f, 1.0f, 1.0f) * Pss * Pms * cosMult;

  if (spec)             a_pRes->flags = (RAY_EVENT_S | RAY_EVENT_T);
  else                  a_pRes->flags = (RAY_EVENT_G | RAY_EVENT_T);

  if      (refrData.success  && cosThetaOut >= -1e-6f) a_pRes->color = make_float3(0.0f, 0.0f, 0.0f); // refraction/transparency must be under surface!
  else if (!refrData.success && cosThetaOut < 1e-6f)   a_pRes->color = make_float3(0.0f, 0.0f, 0.0f); // reflection happened in wrong way
}


// implicit strategy

static inline void glassSampleAndEval(const Material* a_materials, const float4 a_rands, 
  float3 a_viewDir, float3 a_normal, const float2 a_tc, BsdfSample* a_pRes, MisData* a_misPrev)
{
  // PLEASE! use 'a_materials[0].' for a while ... , not a_materials-> and not *(a_materials).

  const float3 colorReflect    = to_float3(a_materials[0].colors[GLASS_COLOR_REFLECT]);   
  const float3 colorTransp     = to_float3(a_materials[0].colors[GLASS_COLOR_TRANSP]);
  const float  roughReflect    = clamp(1.0f - a_materials[0].data[GLASS_FLOAT_GLOSS_REFLECT], 0.0f, 1.0f);
  const float  roughTransp     = clamp(1.0f - a_materials[0].data[GLASS_FLOAT_GLOSS_TRANSP], 0.0f, 1.0f);                          
  float        ior             = a_materials[0].data[GLASS_FLOAT_IOR]; 

  const float  dotNV           = dot(a_normal, a_viewDir); ///< a_viewDir - direction to the incoming ray
  const bool   a_hitFromInside = dotNV < 0.0f;             // an easy way to understand that we are in volume
  float3 invNormal             = a_normal;
  if (a_hitFromInside)
  {
    invNormal     = (-1.0f) * a_normal;
    ior         = 1.0f; // exit from the volume
  }


  const float fresnel = (a_misPrev->ior == ior) ? 0.0f : FrDielectricPBRT(fabs(dotNV), a_misPrev->ior, ior);


  float3 dir;
  float  val;
  float  pdf;

  if (a_rands.w < fresnel) // reflection
  {
    if (roughReflect == 0.0f) 
    {
      dir                     = reflect((-1.0f) * a_viewDir, a_normal);
      const float cosThetaOut = dot(dir, a_normal);
      val                     = (cosThetaOut <= 1e-6f) ? 0.0f : (1.0f / std::max(cosThetaOut, 1e-6f));  // BSDF is multiplied (outside) by cosThetaOut. For mirrors this shouldn't be done, so we pre-divide here instead.
      pdf                     = 1.0f;
    }
    else
    {
      dir                     = ggxSample(float2(a_rands.x, a_rands.y), a_viewDir, a_normal, roughReflect);
      val                     = ggxEvalBSDF(dir, a_viewDir, a_normal, roughReflect);
      pdf                     = ggxEvalPDF(dir, a_viewDir, a_normal, roughReflect);
    }

    a_pRes->direction         = dir;
    a_pRes->color             = val * colorReflect;
    a_pRes->pdf               = pdf;
    a_pRes->flags             = (roughReflect == 0.0f) ? RAY_EVENT_S : RAY_FLAG_HAS_NON_SPEC;
  }
  else  // transparency
  {
    refractionGlassSampleAndEval(colorTransp, a_misPrev->ior, ior, roughTransp, a_normal, invNormal, a_rands, (-1.0f) * a_viewDir, a_pRes);
    a_misPrev->ior = ior;
  }
}


// explicit strategy
static void glassEval(const Material* a_materials, float3 l, float3 v, float3 n, float2 tc,
  float3 color, BsdfEval* res)
{
  // because we don't want to sample this material with shadow rays
  res->color = make_float3(0.0f, 0.0f, 0.0f);  
  res->pdf   = 0.0f;
}