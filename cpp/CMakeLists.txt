cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
#cmake_policy(SET CMP0094 NEW)

if (UNIX)
    # cuda compiler path for linux, as cmake cant find it
    set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
endif (UNIX)

project(photonc LANGUAGES CXX CUDA)

include(CheckLanguage)
check_language(CUDA)

find_package(Python COMPONENTS Interpreter Development)
#find_package(pybind11 REQUIRED)
#find_package(CUDA)

if(CMAKE_CUDA_COMPILER AND NOT CUDA_DISABLED)  # the latter is a user definable variable
   enable_language(CUDA)
   add_compile_definitions(CUDA_ENABLED)

   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --gpu-architecture=sm_50")
else()
   message(WARNING "No CUDA compiler found or disabled by user defined command line variable '-DCUDA_DISABLED=True'.
   You may need to specify its path like \t'export CUDACXX=/usr/local/cuda/bin/nvcc'")
endif()

set(SourceFiles 
	Context.cpp 
	CameraCalibration.cu 
#	ContinuousFRC.cu
	DebugImageCallback.cpp
	DriftEstimation.cu
	CudaUtils.cu
	ImageProcessor.cpp
	ImgFilterQueue.cu
	LinkLocalizations.cpp
	Rendering.cpp
	ROIQueue.cpp
	StringUtils.cpp
	FFT.cpp
	simflux/SIMFLUX.cu
	SpotDetection/PSFCorrelationSpotDetector.cu
	SpotDetection/SpotDetector.cu
	Estimators/EstimationQueue.cpp
	Estimators/EstimatorImpl.cu
	Estimators/Estimator.cpp
	Estimators/Gaussian/GaussianPSF.cu
	Estimators/CSpline/CubicSplinePSF.cu
	)

list(TRANSFORM SourceFiles PREPEND "SMLMLib/")

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../photonpy/x64)

#message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")

if(CMAKE_CUDA_COMPILER AND NOT CUDA_DISABLED)

	add_library(photonc SHARED ${SourceFiles})
	target_include_directories(photonc PUBLIC SMLMLib) #${CMAKE_BINARY_DIR})
	target_compile_definitions(photonc PUBLIC SMLM_EXPORTS)
	target_compile_features(photonc PUBLIC cxx_std_14)
	set_property(TARGET photonc PROPERTY POSITION_INDEPENDENT_CODE ON)

	#target_include_directories(photonc PUBLIC src/*.h)
	#target_compile_features(dme PUBLIC cxx_std_11)

   #add_library(dme STATIC src/DriftEstimation.cu)
   #target_include_directories(dme PUBLIC include)
   set_target_properties(photonc PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
   set_property(TARGET photonc PROPERTY CUDA_ARCHITECTURES 72 75)
   target_compile_options(photonc PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:  # seems to be unnecssary but keep the code block for the moment
                        --expt-relaxed-constexpr
			--expt-extended-lambda
                         >)
                        # -rdc=true
   set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math)                     # --gpu-architecture=sm_50

	#find_package(CUDA REQUIRED)
	target_link_libraries(photonc PUBLIC ${CUDA_LIBRARIES} ${CUDA_cufft_LIBRARY})
   
   #target_link_libraries(photonc 
   #cuda_add_cufft_to_target(photonc)

   link_directories(/usr/local/cuda/lib64)
   #include_directories("${CUDA_INCLUDE_DIRS}")

   #pybind11_add_module(spline src/pybind_spline.cpp)
   #target_link_libraries(dme PRIVATE dme_cu_impl dme_cpu_impl)

else()  # NO CUDA
	#target_include_directories(dme_cpu_impl PUBLIC include)
	message(SEND_ERROR "photonpy currently needs CUDA to compile")
   #pybind11_add_module(spline src/pybind_spline.cpp)

endif()
