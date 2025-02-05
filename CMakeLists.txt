cmake_minimum_required (VERSION 3.15)
project (HydraCore3)

option(USE_VULKAN "Enable Vulkan implementation" OFF)

# get rid of annoying MSVC warnings.
add_definitions(-D_CRT_SECURE_NO_WARNINGS)

add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

# compiler warnings
add_library(project_warnings INTERFACE)

include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# sanitizer options
include(cmake/Sanitizers.cmake)
enable_sanitizers(project_options)


set(GENERATOR_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "Root output directory for all artifacts")
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin-debug)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin-debug)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin-debug)
  add_definitions(-D_DEBUG)
else()
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin-release)
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin-release)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin-release)
endif()


find_package(OpenMP)

if(USE_VULKAN)

  if(CMAKE_SYSTEM_NAME STREQUAL Windows)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_WIN32_KHR)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Linux)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_XLIB_KHR)
  elseif(CMAKE_SYSTEM_NAME STREQUAL Darwin)
    set(VOLK_STATIC_DEFINES VK_USE_PLATFORM_MACOS_MVK)
  endif()

  add_subdirectory(external/volk)

  add_compile_definitions(USE_VOLK)
  add_compile_definitions(USE_VULKAN)
  
  include_directories(${Vulkan_INCLUDE_DIR})
endif()


# try to silence warnings for include files from these dirs
include_directories(SYSTEM ${Vulkan_INCLUDE_DIR})
include_directories(SYSTEM "external/tinyexr"
                    "external/volk"
                    "external/LiteMath")
include_directories("external"
                    "external/vkutils"
                    "external/CrossRT")

link_directories("external/CrossRT/lib")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -Wno-attributes")
  message(STATUS "Debug is ON!")
  add_definitions(-D_DEBUG)
  #uncomment this to detect broken memory problems via gcc sanitizers
  #set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fsanitize-address-use-after-scope -fno-omit-frame-pointer -fsanitize=leak -fsanitize=undefined -fsanitize=bounds-strict")
else()
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes -Wno-unused-parameter -Wno-useless-cast")
  message(STATUS "Debug is OFF!")
endif()


set(VULKAN_EXTERNAL_SRC
        external/vkutils/vk_utils.cpp
        external/vkutils/vk_copy.cpp
        external/vkutils/vk_buffers.cpp
        external/vkutils/vk_images.cpp
        external/vkutils/vk_context.cpp
        external/vkutils/vk_alloc_simple.cpp
        external/vkutils/vk_pipeline.cpp
        external/vkutils/vk_descriptor_sets.cpp
        external/vkutils/ray_tracing/vk_rt_utils.cpp
        external/vkutils/geom/vk_mesh.cpp
        external/CrossRT/VulkanRTX.cpp
        external/LiteScene/scene_mgr.cpp
        external/vkutils/geom/cmesh.cpp)

set(VULKAN_SRC
        integrator_pt1_generated.cpp
        integrator_pt1_generated_init.cpp
        integrator_pt1_generated_ds.cpp )

add_library(miniz external/tinyexr/miniz.c)

set(LOADER_EXTERNAL_SRC
        external/LiteScene/pugixml.cpp
        external/LiteScene/hydraxml.cpp
        external/LiteScene/cmesh4.cpp)

set(OTHER_EXTERNAL_SRC
        external/LiteMath/Image2d.cpp
        external/CrossRT/EmbreeRT.cpp)

set(CORE_SRC
        main.cpp
        integrator_rt.cpp
        integrator_pt1.cpp
        integrator_pt2.cpp
        integrator_pt_host.cpp
        integrator_pt_scene.cpp
        mi_materials.cpp
        imageutils.cpp
        utils.cpp)


if(USE_VULKAN)               
  add_executable(hydra ${CORE_SRC}
                       ${VULKAN_EXTERNAL_SRC}
                       ${VULKAN_SRC}
                       ${LOADER_EXTERNAL_SRC}
                       ${OTHER_EXTERNAL_SRC})

#  set(LINKED_VULKAN ${Vulkan_LIBRARY})
else()
  add_executable(hydra ${CORE_SRC}
                       ${LOADER_EXTERNAL_SRC}
                       ${OTHER_EXTERNAL_SRC})
#  set(LINKED_VULKAN "")
endif()


set_target_properties(hydra PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  link_directories("${CMAKE_SOURCE_DIR}/external/CrossRT/lib_win64")
  set(EMBREE_LIBS ${CMAKE_SOURCE_DIR}/external/CrossRT/lib_win64/embree3.lib)
elseif(CMAKE_SYSTEM_NAME STREQUAL Linux)
  link_directories(${CMAKE_SOURCE_DIR}/external/CrossRT/lib)
  set(EMBREE_LIBS embree3 embree_sse42 embree_avx embree_avx2 lexers simd sys tasking)
endif()



target_link_libraries(hydra LINK_PUBLIC project_options project_warnings miniz)

if(USE_VULKAN)
  target_link_libraries(hydra LINK_PUBLIC volk ${EMBREE_LIBS})
else()
  target_link_libraries(hydra LINK_PUBLIC ${EMBREE_LIBS})
endif ()

if(CMAKE_SYSTEM_NAME STREQUAL Windows)
  # add windows specific options here
  add_custom_command(TARGET hydra POST_BUILD COMMAND ${CMAKE_COMMAND}
          -E copy_directory "${PROJECT_SOURCE_DIR}/external/CrossRT/bin_win64" $<TARGET_FILE_DIR:hydra>)
else()
  target_link_libraries(hydra LINK_PUBLIC dl)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  # add debug specific options here
else()
  target_link_libraries(hydra LINK_PUBLIC OpenMP::OpenMP_CXX)
endif()