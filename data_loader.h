#ifndef UTILS_H
#define UTILS_H

#include "headers.h"

void load_mesh(const string& filename,
            std::vector<float>& vertexCoordinates,
            std::vector<float>& textureCoordinates,
            std::vector<int>&   vertexIndices,
            std::vector<int>&   textureIndices);


// load the camera info
void load_camera(const string& filename,
                std::vector<Eigen::Matrix4d>& cam_K,
                std::vector<Eigen::Matrix4d>& cam_T,
                std::vector<Eigen::Matrix4d>& cam_P);


// outside the image space -> false
bool proj_3D_to_2D(const trimesh::point& p, 
				   const Eigen::Matrix4d& projMatrix,
				   float& x, float& y, float& z,
                   int img_height, int img_width);


void calculateTBN(trimesh::TriMesh* mesh,
                  const std::vector<float>& uvCoordinates,
                  const std::vector<int>& uvIndices,
                  std::vector<Eigen::Matrix3d>& vertex_TBN,
                  std::vector<Eigen::Matrix3d>& face_TBN,
                  std::vector<Eigen::Matrix3d>& face_TBN_norm);


#endif
