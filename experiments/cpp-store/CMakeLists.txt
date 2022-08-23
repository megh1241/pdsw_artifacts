cmake_minimum_required (VERSION 3.12)
project (dummy CXX)
#set(CMAKE_CXX_STANDARD 17)
set (CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set (CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

if (NOT CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Release
         CACHE STRING "Choose the type of build." FORCE)
    set_property (CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
                  "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif ()
set (CMAKE_PREFIX_PATH "" CACHE STRING "External dependencies path")
set (BUILD_SHARED_LIBS "ON" CACHE BOOL "Build a shared library")
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set (TMCI_INCLUDE_DIR "/home/mmadhya1/.local/lib/python3.8" CACHE STRING "Path to TMCI headers")
#set (TMCI_INCLUDE_DIR "/gpfs/mira-home/mmadhya1/dh1/lib/python3.8/site-packages/" CACHE STRING "Path to TMCI headers")
set (TENSORFLOW_INCLUDE_DIR "/home/mmadhya1/dh-cooley-clean/lib/python3.8/site-packages/tensorflow/include" CACHE STRING "Path to Tensorflow headers")
find_package (Python REQUIRED)
set(PYTHON_LIBRARY "/home/mmadhya1/dh-cooley-clean/lib/libpython3.8.so")

set(MARGO_INCLUDE_DIR "/home/mmadhya1/install/include" CACHE STRING "ptah to margo")
set(MARGO_LIB_DIR "/home/mmadhya1/install/lib")
set(THALLUM_LIB_DIR "/home/mmadhya1/install/lib/thallium")

find_package(CUDAToolkit REQUIRED)
#find_library(CUFile_LIBRARY cufile PATHS ${CUDAToolkit_LIBRARY_DIR} REQUIRED)



#set(RAPIDJSON_INCLUDE_DIR "/home/mmadhya1/install_theta2/lib/rapidjson")
#link_directories ("/lus/swift/home/mmadhya1/.local/conda/2021-09-22/lib/python3.8/site-packages/tmci-0.1-py3.8-linux-x86_64.egg")
#link_directories ("/home/mmadhya1/.local/lib/python3.8/site-packages/tmci-0.1-py3.8-linux-x86_64.egg")
#link_directories (${THALLIUM_INCLUDE_DIR})
#include_directories (${PYTHON_INCLUDE_DIR})
#set(TF_LIB $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())'))
#set(GCC_COVERAGE_LINK_FLAGS)
#add_definitions(${GCC_COVERAGE_LINK_FLAGS})

#set(Python_ROOT_DIR ${Python_ROOT_DIR})
#find_package (PythonLibs REQUIRED 3.8 EXACT REQUIRED)
# Find TMCI
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tmci; print(tmci.get_library_dir())"
	OUTPUT_VARIABLE TMCI_LIBRARY_DIR
	OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND "${Python_EXECUTABLE}" "-c" "import tmci; print(tmci.get_library())"
	OUTPUT_VARIABLE TMCI_LIBRARIES
	OUTPUT_STRIP_TRAILING_WHITESPACE)
message (STATUS "TMCI library dir: ${TMCI_LIBRARY_DIR}")
message (STATUS "TMCI library: ${TMCI_LIBRARIES}")

link_directories (${TMCI_LIBRARY_DIR})
link_directories (${MARGO_LIB_DIR})

include_directories (${TENSORFLOW_INCLUDE_DIR} ${TMCI_INCLUDE_DIR} ${MARGO_INCLUDE_DIR} ${THALLIUM_INCLUDE_DIR} ${PYTHON_INCLUDE_DIR})
set (DUMMY_SRC model_client.cpp DummyBackend.cpp)

add_library (dummy ${DUMMY_SRC})
target_link_libraries (dummy ":${TMCI_LIBRARIES}") # we need the ":" because the library doesn't start with "lib"
target_link_libraries (dummy "-lmargo") # we need the ":" because the library doesn't start with "lib"
target_link_libraries (dummy "-labt") # we need the ":" because the library doesn't start with "lib"
target_link_libraries (dummy "-lmercury") # we need the ":" because the library doesn't start with "lib"
#target_link_libraries (dummy "-lpthread") # we need the ":" because the library doesn't start with "lib"
#target_link_libraries (dummy "-lrocksdb") # we need the ":" because the library doesn't start with "lib"
target_link_libraries(dummy CUDA::cuda_driver CUDA::cudart )
#target_link_libraries (dummy "libmodel_client.so") # we need the ":" because the library doesn't start with "lib"

#add_subdirectory (src)
