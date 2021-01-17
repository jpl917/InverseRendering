# Inverse Rendering Using Ceres
This project aims to recover the facial reflectance (diffuse albedo, specular albedo, roughness, displacement map) from single-shot multi-view images. 


### HDR environment map capture 

1. Capture a mirror sphere at several exposures to acquire an HDR light probe of the surrounding environment.
2. Estimate the position of the mirror sphere.
3. Compute the latitude-longitude environment map.


### Spherical harmonics estimation 

<p align="center">

### Median-cut to approximate the environment light
<p align="center">
<img src="imgs/sh/indoor.JPG" alt="Sample"  width="300" height="121">

From left to right: environment map, 3 order spherial harmonics approximation.


### Estimate detailed normal map from displacement map
<p align="center">
<img src="maya_render/normal_map.jpg" alt="Sample"  width="300" height="300"><img src="maya_render/normal_map_detailed.jpg" alt="Sample"  width="300" height="300">

From left to right: normal map from geometry, high-resolution normal map fusing displacement map.


### Render using cook-torrance brdf

<p align="center">
<img src="imgs/render.jpg" alt="Sample"  width="270" height="480"><img src="imgs/render_diff.jpg" alt="Sample"  width="270" height="480"><img src="imgs/render_spec.jpg" alt="Sample"  width="270" height="480">

From left to right: render, diffuse, specular.

### Inverse Rendering Optimization


### Reference
1. Gotardo et al. Practical Dynamic Facial Appearance Modeling and Acquisition. ACM Transactions on Graphics. 2018.
2. Riviere et al. Single-Shot High-Quality Facial Geometry and Skin Appearance Capture. ACM Transactions on Graphics. 2020.


