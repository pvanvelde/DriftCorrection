# Drift Estimation

The drift estimation method is described in the paper "Drift Correction in Localization Microscopy Using Entropy Minimization," you can find it [here](https://opg.optica.org/oe/fulltext.cfm?uri=oe-29-18-27961&id=457245).

Please note that the code for this method is available in the following repository: [Drift Estimation Repository](https://github.com/qnano/drift-estimation). However, the installation process vary slightly.

## Requirements

Before you begin, make sure you have the following prerequisites:

- **Operating System:** Windows (Tested on Windows 10)
- **CUDA Compatible GPU:** It's essential to have a CUDA 11.2 compatible GPU (tested on NVIDIA GeForce GTX 980). You can check compatibility information [here](https://docs.nvidia.com/deploy/cuda-compatibility/).

## Installation and Usage

Follow these steps to install and set up the project:

1. **Install Python:**
   - We recommend using Anaconda to manage Python environments. You can download Anaconda [here](https://www.anaconda.com/distribution/).

2. **Create a Virtual Environment:**
   - Create a virtual environment using Anaconda with the following commands:
     ```bash
     conda create -n myenv anaconda python=3.9
     conda activate myenv
     ```

3. **Install Visual Studio 2019 Community Edition:**
   - Download and install Visual Studio 2019 Community Edition from [here](https://visualstudio.microsoft.com/vs/older-downloads/).

4. **Extract External Libraries:**
   - Extract the external libraries from `cpp/external.zip` so that the `cpp` directory contains a folder called "external."

5. **Install CUDA Toolkit 11.2:**
   - Install [CUDA Toolkit 11.2](https://developer.nvidia.com/cuda-toolkit-archive).
   - Refer to the installation guide [here](https://docs.nvidia.com/cuda/archive/11.2.0/cuda-installation-guide-microsoft-windows/index.html).

6. **Build SMLMLib:**
   - In Visual Studio, open the `smlm.sln` solution.
   - Set the build mode to "**Release**" and platform to "**x64**."
   - Build the `SMLMLib` project.

7. **Install Python Dependencies:**
   - Install the required Python packages by running:
     ```bash
     pip install -r requirements.txt
     ```

8. **Run the example code:**
   - You should now be able to run the script `drift_correction_example/runfile.py`.

## Troubleshooting

If you encounter issues or the above steps do not yield the desired results, consider these fixes:

1. **Check CUDA Installation:**
   - Run the following command in the terminal to check if CUDA is installed correctly:
     ```
     nvidia-smi
     ```

2. **Change CUDA Settings:**
   - Search for "NSight Monitor" and run it as an administrator.
   - It may not open a new window but appear as a background process in the system tray (bottom right part of your screen).
   - Right-click the NSight Monitor icon and go to 'Options.'
   - Change 'WDDM TDR Delay' to '180' and 'WDDM TDR Enabled' to 'False.'
   - You may need to reboot your system.

3. **Update GPU Driver:**
   - Search for 'Device Manager.'
   - Under 'Display adapters,' right-click your video card (usually NVIDIA XXXX).
   - Click 'Properties' -> 'Driver' -> 'Update Driver...'
   - Follow the prompts to update the GPU driver.
   - You may need to reboot your system.

## License

This project is licensed under the MIT license. For more details, see the [LICENSE](LICENSE.txt) file.