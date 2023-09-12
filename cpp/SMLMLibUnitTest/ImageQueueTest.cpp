#include "PSFModels/GaussianPSF.h"
#include "ImageProcessingQueue.h"
#include "SpotDetection/SpotDetector.h"
#include "RandomDistributions.h"

#include "../SMLMLibTest/TIFFReadWrite.h"
#include "GetPreciseTime.h"

#include "catch.hpp"

void printCudaStats()
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);
	for (int i = 0; i < nDevices; i++) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		DebugPrintf("Device Number: %d\n", i);
		DebugPrintf("  Device name: %s\n", prop.name);
		DebugPrintf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
		DebugPrintf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
		DebugPrintf("  Peak Memory Bandwidth (GB/s): %.1f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth / 8) / 1.0e6);
		DebugPrintf("  Number of CUDA Processors: %d\n", prop.multiProcessorCount);
		DebugPrintf("  Concurrent Kernels: %d\n", prop.concurrentKernels);
	}
}

std::vector<float> GenerateImageWithSpots(float sigma, int nspots, int w, int h)
{
	// Generate image
	std::vector<float> image(w*h);
	std::vector<float> spotList(5 * nspots);

	for (int i = 0; i < nspots; i++) {
		float*spot = &spotList[5 * i];

		spot[0] = rand_uniform<float>()*w;
		spot[1] = rand_uniform<float>()*h;
		spot[2] = spot[3] = sigma;
		spot[4] = rand_uniform<float>() * 1000 + 500;// intensity
	}

	Gauss2D_Draw(&image[0], w, h, &spotList[0], nspots);

	return image;
}

TEST_CASE("2D Gaussian Image Queue")
{
	ImageQueueConfig cfg;
	float psfSigma = 2;

	cfg.imageWidth = 1024;
	cfg.imageHeight = 512;
	cfg.historyLength = 4;
#ifdef _DEBUG
	cfg.cudaStreamCount = 1;
#else
	cfg.cudaStreamCount = 3;
#endif

	printCudaStats();

	int nspots = 4000, nframes = 500;
#ifdef _DEBUG
	nspots = 100;
	nframes = 10;
	cfg.imageWidth = 256;
#endif
	auto image = GenerateImageWithSpots(psfSigma, nspots, cfg.imageWidth, cfg.imageHeight);

	//	WriteTIFF("spots.tif", &image[0], cfg.imageWidth, cfg.imageHeight);
		//OpenTIFF("C:\\projects\\MT_datasets\\sequence-MT1.N1.LD-AS-Exp-as-stack\\sequence-as-stack-MT1.N1.LD-AS-Exp.tif");

	int roisize = 10;
	SpotDetectorConfig sdcfg(psfSigma, roisize, 20, 1e20);
	ISpotDetectorFactory* sdfactory = SpotDetector_Configure(sdcfg);

	auto *q = Gauss2D_CreateImageQueue(cfg, psfSigma, roisize, cfg.imageWidth*cfg.imageHeight / 200, sdfactory, 0);

	SpotDetector_DestroyFactory(sdfactory);

	for (int i = 0; i < nframes; i++) {
		if (i % 50 == 0)
			DebugPrintf("Frame %d scheduled.\n", i);
		q->PushFrame(i, &image[0]);
	}
	q->Start();
	double t0 = GetPreciseTime();

	volatile bool stopProgressUpdater = false;
	std::thread progressUpdater([&]() {
		while (!stopProgressUpdater) {
			int queueLen = q->GetQueueLength();
			DebugPrintf("Queue length: %d. Active streams: %d. Number of frames in memory: %d\n", queueLen, q->ActiveStreams(), q->GetNumFrames());
			std::this_thread::sleep_for(std::chrono::milliseconds(500));
		}
	});

	q->WaitForAllDone();
	bool idle = q->IsIdle();

	stopProgressUpdater = true;
	progressUpdater.join();

	double t1 = GetPreciseTime();

	int n = 0;
	nframes = q->resultContainer.GetFrameCount();

	for (int f = 0; f < nframes; f++) {
		auto fr = q->resultContainer.GetResults(f);
		if (f == 0) {
			DebugPrintf("Frame #%d has %d results:\n", f, fr->results.size());
			for (int i = 0; i < min(100, (int)fr->results.size()); i++) {
				auto p = fr->results[i].estimate;
				Int2 roi = fr->results[i].roiPosition;
				//				p[0] += fr.results[i].roiPosition[0];
					//			p[1] += fr.results[i].roiPosition[1];

				DebugPrintf("x = %.3f,y = %.3f, I=%.1f, bg=%.2f. roi.x = %d, roi.y = %d\n", p[0], p[1], p[2], p[3], roi[0], roi[1]);
			}
		}
		n += (int)fr->results.size();
	}

	double t = t1 - t0;

	for (int i = 0; i < q->NumTasks(); i++) {
		DebugPrintf("Task %s avg runtime: %f\n", q->GetTask(i)->GetTaskName(), q->GetAverageTaskRunTime(i));
	}

	DebugPrintf("Processed %d frames (%.1f loc/s) in %.1f s.\n", nframes, n / t, t);

	delete q;
}