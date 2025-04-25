# E7 Appkit Lab
This repository contains multiple projects to experiment with the Alif Ensemble E7 AppKit. The following projects are included:
- bare_metal
- check_features
- dotp_tests
- gemm_tests
- helium_instructions
- jit_test
- stream_benchmark
- tflm_test

More information about the projects can be found in the project folders

## Prerequisites
The following prerequisites need to be installed:
- SEGGER J-Link (includes the J-Link GDB Server). This is needed for the debug extension.
- ARM GNU Toolchain. This is needed for the debug extension.
- Alif SeToolkit. This is needed for programming the AppKit. Can be downloaded from [alifsemi.com](alifsemi.com).
- VS Code with the following extensions (they will be recommended when opening VS Code)
  - Arm Tools Environment Manager
  - Arm CMSIS Solution
  - Cortex Debug (used for providing support to debug)
  - Command Variable (needed for modifying the `launch.json`-file to load the .hex-file while debugging)

## Quick start
First clone the template project repository
```
git clone https://github.com/01ill/a7-appkit-lab.git
cd a7-appkit-lab
git submodule update --init
```

The CMSIS Toolbox and `tasks.json` contain all needed information for building a project. The project needs to be selected from the CMSIS Toolbox and can then be built via the included build task.

The file `launch.json` contains the debugging configuration and a debug session can be started natively in VS Code.

## Project structure
```
├── libs - contains all needed helper libraries for board support and timing
├── src - contains all projects
│   ├── helium_instructions - sample project
│   │   ├── helium_instructions.cproject.yml - CMSIS configuration of the project
│   │   ├── main.cpp
│   │   └── RTE - contains generated files from CMSIS needed for building
├── cdefault.yml - compiler and linker configuration
├── e7_tests.csolution.yml - common configuration of all projects
└── vcpkg-configuration.json - configuration of the CMSIS Toolbox
```

The projects use CMSIS intermediate layers to provide the board and timing support. All Layers are contained in the `libs` directory.