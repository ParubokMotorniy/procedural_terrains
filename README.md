**Terrain generation sandbox**


This sandbox allows one to build simple terrain generation pipelines, run them and behold the results.   

![demo](./data/sandbox_demo.png)

## How to get it running

The `Releases` page of this repository contains pre-built binaries for Windows and Linux. If all you want is to play with the sandbox - download the archive that matches your platform, decompress it and run the executable `GenerationSandbox.exe/x86_64`.    

In case you want to open the project in Unity engine, please, note that it was developed in Unity of version **6000.3.2f1**. Obviously, earlier version might not work. 

## Important directories  

- *datasets* - this directory contains the datasets used in the thesis work that this sandbox has been devloped for in the first place.
- *experiments* - this directory contains the python scripts used for the processing of data. If you ever decide to run any of them, the package dependencies for `pip` are listed in `requirements.txt`. The names of the scripts are quite self-descriptive and the key ones feature a command-line interface.


## Bit of theory

There are two classes of steps that can be put into a pipeline:
- *Generators*: these steps generate the base geometry of the terrain by running one of the following algorithms: Uber noise (UN), spectral synthesis (FFT), random midpoint displacement (RMD) or signed distance function-based (SDF). There can only be one generator in a pipeline.      
  Here is an example of what results you might expect from the generators:
  |      |     |
  |------|-----|
  |UN                                  |                                  FFT|
  |   ![un_demo](./data/un_demo.png)   |   ![fft_demo](./data/fft_demo.png)  |
  |RMD                                 |                                  SDF|
  |   ![rmd_demo](./data/rmd_demo.png) |   ![sdf_demo](./data/sdf_demo.png)  |
- *Eroders*: these steps erode the base geometry. That is, they simulate the various processes of soil being moved around and redeposited. TE stands for "thermal erosion", (C)HE stands for "(cellular) hydraulic erosion" and PE stands for "particle erosion". You can stack as many eroders as you want.       
  Here is an example of what effects the eroders can produce:
  |      |     |
  |------|-----|
  |No erosion                                |                                  Thermal erosion|
  |   ![un_demo](./data/no_erosion_demo.png)   |   ![fft_demo](./data/te_erosion_demo.png)  |
  |Hydraulic erosion                                 |                                  Particle erosion|
  |   ![rmd_demo](./data/he_erosion_demo.png) |   ![sdf_demo](./data/pe_erosion_demo.png)  |

## UI

The steps comprising the pipelines can be tuned on the panel to the left. This panel also allows to choose generation seeds, dimensions of the desired heightmap and the scale of the rendered terrain.       
The panel on the right is where the pipelines are assmebled. Click "+" to add a new step. Click "-" to remove the last one.      
The buttons that offer to "collect statistics" trigger a sequence of executions of the pipeline you've assembled. Steps and their parameters remain the same between runs (unless "Randomize parameters" is checked) but seeds change every time. The heightmaps generated over the course of this sequence, as well as performance measurements, are deposited in the persistent directory of the sandbox: `%userprofile%\AppData\LocalLow\CoffeeSquirrel\GenerationSandbox` under Windows or `$XDG_CONFIG_HOME/unity3d/CoffeeSquirrel/GenerationSandbox` under Linux.     

## Controls

The pipelines can be assembled separately for CPU and GPU. To switch between corresponding terrain tiles, use buttons on the 'Camera focus' panel.        
To rotate the heightmap, keep the right mouse buttow down while moving the mouse. Use the wheel to zoom in/out.


Now you are ready to go nuts with the sandbox.
     