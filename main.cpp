#include <iostream>
#include <fstream>

#include "integrator_pt.h"
#include "ArgParser.h"

bool SaveImage4fToEXR(const float* rgb, int width, int height, const char* outfilename, float a_normConst = 1.0f, bool a_invertY = false);
bool SaveImage4fToBMP(const float* rgb, int width, int height, const char* outfilename, float a_normConst = 1.0f, float a_gamma = 2.2f);

#include "vk_context.h"
std::shared_ptr<Integrator> CreateIntegrator_Generated(int a_maxThreads, vk_utils::VulkanContext a_ctx, size_t a_maxThreadsGenerated);

int main(int argc, const char** argv)
{
  #ifndef NDEBUG
  bool enableValidationLayers = true;
  #else
  bool enableValidationLayers = false;
  #endif

  int WIN_WIDTH  = 1024;
  int WIN_HEIGHT = 1024;
  std::vector<float4> realColor(WIN_WIDTH*WIN_HEIGHT);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  std::shared_ptr<Integrator> pImpl = nullptr;
  ArgParser args(argc, argv);
  
  std::string scenePath = "../resources/HydraCore/hydra_app/tests/test_42/statex_00001.xml";
  if(args.hasOption("-in"))
    scenePath = args.getOptionValue<std::string>("-in");

  std::string imageOut = "z_out.bmp";
  if(args.hasOption("-out"))
    imageOut = args.getOptionValue<std::string>("-out");

  const bool saveHDR = imageOut.find(".exr") != std::string::npos;

  const std::string imageOutClean = imageOut.substr(0, imageOut.find_last_of("."));

  std::string integratorType = "mispt";
  if(args.hasOption("-integrator"))
    integratorType = args.getOptionValue<std::string>("-integrator");

  int PASS_NUMBER = 1024;
  if(args.hasOption("-spp"))
    PASS_NUMBER = args.getOptionValue<int>("-spp");
  
  float gamma = 2.2f;
  if(args.hasOption("-gamma"))
    gamma = args.getOptionValue<float>("-gamma");
  else if(args.hasOption("-out_gamma"))
    gamma = args.getOptionValue<float>("-out_gamma");
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////
  
  bool onGPU = args.hasOption("--gpu");
  if(onGPU)
  {
    unsigned int a_preferredDeviceId = args.getOptionValue<int>("--gpu_id", 0);
    auto ctx = vk_utils::globalContextGet(enableValidationLayers, a_preferredDeviceId);
    pImpl = CreateIntegrator_Generated( WIN_WIDTH*WIN_HEIGHT, ctx, WIN_WIDTH*WIN_HEIGHT);
  }
  else
    pImpl = std::make_shared<Integrator>(WIN_WIDTH*WIN_HEIGHT);
  
  ///////////////////////////////////////////////////////////////////////////////////////
  ///////////////////////////////////////////////////////////////////////////////////////

  pImpl->SetViewport(0,0,WIN_WIDTH,WIN_HEIGHT);
  std::cout << "[main]: Loading scene ... " << scenePath.c_str() << std::endl;
  pImpl->LoadScene(scenePath.c_str());
  pImpl->CommitDeviceData();

  // remember (x,y) coords for each thread to make our threading 1D
  //
  std::cout << "[main]: PackXYBlock() ... " << std::endl; 
  pImpl->PackXYBlock(WIN_WIDTH, WIN_HEIGHT, 1);

  const float normConst = 1.0f/float(PASS_NUMBER);
  float timings[4] = {0,0,0,0};
  
  // now test path tracing
  //
  if(integratorType == "naivept" || integratorType == "all")
  {
    const int NAIVE_PT_REPEAT = 1;
  
    std::cout << "[main]: NaivePathTraceBlock() ... " << std::endl;
    memset(realColor.data(), 0, sizeof(float)*4*realColor.size());
    pImpl->SetIntegratorType(Integrator::INTEGRATOR_STUPID_PT);
    pImpl->UpdateMembersPlainData();
    pImpl->NaivePathTraceBlock(WIN_WIDTH*WIN_HEIGHT, 6, realColor.data(), PASS_NUMBER*NAIVE_PT_REPEAT);
    
    std::cout << std::endl;
    pImpl->GetExecutionTime("NaivePathTraceBlock", timings);
    std::cout << "NaivePathTraceBlock(exec)  = " << timings[0]              << " ms " << std::endl;
    std::cout << "NaivePathTraceBlock(copy)  = " << timings[1] + timings[2] << " ms " << std::endl;
    std::cout << "NaivePathTraceBlock(ovrh)  = " << timings[3]              << " ms " << std::endl;
    std::cout << std::endl;

    if(saveHDR)
      SaveImage4fToEXR((const float*)realColor.data(), WIN_WIDTH, WIN_HEIGHT, imageOut.c_str(), true);
    else
    {
      const std::string outName = (integratorType == "naivept") ? imageOut : imageOutClean + "_naivept.bmp"; 
      SaveImage4fToBMP((const float*)realColor.data(), WIN_WIDTH, WIN_HEIGHT, outName.c_str(), normConst, gamma);
    }
  }

  if(integratorType == "shadowpt" || integratorType == "all")
  {
    std::cout << "[main]: PathTraceBlock(Shadow-PT) ... " << std::endl;
    memset(realColor.data(), 0, sizeof(float)*4*realColor.size());
    pImpl->SetIntegratorType(Integrator::INTEGRATOR_SHADOW_PT);
    pImpl->UpdateMembersPlainData();
    pImpl->PathTraceBlock(WIN_WIDTH*WIN_HEIGHT, 6, realColor.data(), PASS_NUMBER);
    
    if(saveHDR)
      SaveImage4fToEXR((const float*)realColor.data(), WIN_WIDTH, WIN_HEIGHT, imageOut.c_str(), true);
    else
    {
      const std::string outName = (integratorType == "shadowpt") ? imageOut : imageOutClean + "_shadowpt.bmp"; 
      SaveImage4fToBMP((const float*)realColor.data(), WIN_WIDTH, WIN_HEIGHT, outName.c_str(), normConst, gamma);
    }
  }

  if(integratorType == "mispt" || integratorType == "all")
  {
    std::cout << "[main]: PathTraceBlock(MIS-PT) ... " << std::endl;
    memset(realColor.data(), 0, sizeof(float)*4*realColor.size());
    pImpl->SetIntegratorType(Integrator::INTEGRATOR_MIS_PT);
    pImpl->UpdateMembersPlainData();
    pImpl->PathTraceBlock(WIN_WIDTH*WIN_HEIGHT, 6, realColor.data(), PASS_NUMBER);
    
    pImpl->GetExecutionTime("PathTraceBlock", timings);
    std::cout << "PathTraceBlock(exec) = " << timings[0]              << " ms " << std::endl;
    std::cout << "PathTraceBlock(copy) = " << timings[1] + timings[2] << " ms " << std::endl;
    std::cout << "PathTraceBlock(ovrh) = " << timings[3]              << " ms " << std::endl;

    if(saveHDR)
      SaveImage4fToEXR((const float*)realColor.data(), WIN_WIDTH, WIN_HEIGHT, imageOut.c_str(), normConst, true);
    else
    {  
      const std::string outName = (integratorType == "mispt") ? imageOut : imageOutClean + "_mispt.bmp"; 
      SaveImage4fToBMP((const float*)realColor.data(), WIN_WIDTH, WIN_HEIGHT, outName.c_str(), normConst, gamma);
    }
  }

  return 0;
}
