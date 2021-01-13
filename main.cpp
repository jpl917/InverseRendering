#include <iostream>
#include <fstream>

#ifdef DEBUG
#include <assert.h>
#endif

#include "utils.h"
#include "render.h"

using namespace std;


int main(){
    
    // 1. load mesh
    trimesh::TriMesh* themesh = trimesh::TriMesh::read("../maya_render/model.obj");
    std::vector<trimesh::point> verts = themesh->vertices;
    std::vector<trimesh::TriMesh::Face> faces= themesh->faces;
    themesh->need_normals();
    
    // 2. load mesh topology information
    std::vector<float> vertexCoordinates, textureCoordinates;
    std::vector<int>   vertexIndices, textureIndices;
    loadMesh("../maya_render/model.obj",vertexCoordinates, textureCoordinates, vertexIndices, textureIndices);

    // 3. load camera information
    std::vector<Eigen::Matrix4d> cam_K, cam_T, cam_P;
    loadCamera("../maya_render/cam.txt", cam_K, cam_T, cam_P);
    
    // 4. load reflectance maps
    cv::Mat diff_albedo = cv::imread("../maya_render/Textures/Head_Diffuse_Unlit.exr", cv::IMREAD_UNCHANGED);
    cv::Mat spec_albedo = cv::imread("../maya_render/Textures/Head_Specular.exr", cv::IMREAD_UNCHANGED);
    cv::Mat roughness_map = cv::imread("../maya_render/Textures/Head_Roughness.exr", cv::IMREAD_UNCHANGED);
    cv::Mat displacement_map = cv::imread("../maya_render/Textures/Head_Displacement.exr", cv::IMREAD_UNCHANGED);
    
    
    // spherical harmonics test (not use)
    std::vector<std::vector<double> > sh_coeffs(3);
    cal_SH_from_envmap("../maya_render/env_map/indoor.JPG", sh_coeffs, SH_ORDER);
    
    
    // render params
    int texture_size = diff_albedo.rows; 
    // render view params
    int img_h = 1920;
    int img_w = 1080;
    int img_c = 3;
    // render view
    int view_idx = 0;
    

    // calcuate the tangent space matrix for each face 
    // to fuse displacement map
    std::vector<Eigen::Matrix3d> vertex_TBN;   // not use
    std::vector<Eigen::Matrix3d> face_TBN;     //TBN matrix for each face
    std::vector<Eigen::Matrix3d> face_TBN_norm;// T.norm, B.norm 
    calculateTBN(themesh, textureCoordinates, textureIndices, vertex_TBN, face_TBN, face_TBN_norm);
    
    
    /**************************************************
     * render detailed normal map (4096 * 4096) 
     * fusing displacement map
     * ************************************************/
//     renderNormalMap(themesh, textureCoordinates, vertexIndices, textureIndices, 
//                     face_TBN, face_TBN_norm, displacement_map, normal_map_detailed, texture_size, texture_size, 3);
    

    
    /**************************************************
     * render single view
     * ***********************************************/
    // projection on view
    std::vector<float> vertices;
    for(size_t i=0; i<verts.size(); i++){
        trimesh::point p = verts[i];
        Eigen::Vector4d pt_w(p.x, p.y, p.z, 1);
        Eigen::Vector4d pt_c = cam_P[view_idx] * pt_w;

        vertices.push_back(pt_c[0]/pt_c[2]);
        vertices.push_back(pt_c[1]/pt_c[2]);
        vertices.push_back(pt_c[2]);
    }
    
    std::vector<float> image_buffer(img_h * img_w * 3, 0.0);
    cv::Mat normal_map_detailed = cv::Mat::zeros(texture_size, texture_size, CV_32FC3);
    renderImage(themesh, vertices, textureCoordinates, vertexIndices, textureIndices, 
                face_TBN, face_TBN_norm,
                diff_albedo, spec_albedo, roughness_map, displacement_map, normal_map_detailed,
                sh_coeffs, image_buffer, img_h, img_w, img_c);
    
    
    // 1. visualize detailed normal map
//     cv::Mat normal_map_detailed_viz = cv::Mat::zeros(texture_size, texture_size, CV_8UC3);
//     for(int i=0; i<texture_size; i++){
//         for(int j=0; j<texture_size; j++){
//             cv::Vec3f val = normal_map_detailed.at<cv::Vec3f>(i,j);
//             if( val[0] == 0 && val[1] == 0 && val[2] == 0) continue;
//             
//             normal_map_detailed_viz.at<cv::Vec3b>(i,j) = cv::Vec3b(
//                                                     (val[0] + 1.0) / 2.0 * 255, 
//                                                     (val[1] + 1.0) / 2.0 * 255, 
//                                                     (val[2] + 1.0) / 2.0 * 255);
//         }
//     }
//     cv::imwrite("normal_map_detailed.jpg", normal_map_detailed_viz);
//     cv::imwrite("normal_map_detailed.exr", normal_map_detailed);
    
    
    // 2. render single view
    cv::Mat img = cv::Mat(img_h, img_w, CV_8UC3);
    for(int i=0; i<img_h; i++){
        for(int j=0; j<img_w; j++){
            cv::Vec3b val(0,0,0);
            for(int k=0; k<3; k++){
                val[k] = 255 * pow(image_buffer[3*(i*img_w+j)+k], 1/2.2);
            }
            img.at<cv::Vec3b>(i,j) = val;
        }
    }
    cv::imwrite("render.jpg", img);
    return 0;
    
}



