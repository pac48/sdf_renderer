# sdf_renderer
3D visualizer for parametric signed distance field (SDF) objects. 
The meshes are ray traced on the GPU using CUDA. 
A python interface is provided to facilitate experimentation.

![](res/random_radial.png)

### GUI with keyboard control
![](res/animation.gif)

### build
```bash
git clone --recursive https://github.com/pac48/sdf_renderer.git
cd sdf_renderer
pip install -r requirments.txt
mkdir build
cd build
cmake .. # use non-default Python -DCUSTOM_PYTHON_EXE=/path/to/python
make install 
```
### run
```bash
python3 main.py
```

### potential build issues
```bash
nvcc --version

nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Wed_Jun__8_16:49:14_PDT_2022
Cuda compilation tools, release 11.7, V11.7.99
Build cuda_11.7.r11.7/compiler.31442593_0
```

```bash
gcc --version

gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0
Copyright (C) 2021 Free Software Foundation, Inc.
This is free software; see the source for copying conditions.  There is NO
warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

```
