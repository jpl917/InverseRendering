#include <iostream>
#include <fstream>

#ifdef DEBUG
#include <assert.h>
#endif

#include "utils.h"
#include "render.h"
#include "median_cut.h"

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
    load_mesh("../maya_render/model.obj",vertexCoordinates, textureCoordinates, vertexIndices, textureIndices);

    // 3. load camera information
    std::vector<Eigen::Matrix4d> cam_K, cam_T, cam_P;
    load_camera("../maya_render/cam.txt", cam_K, cam_T, cam_P);
    
    // 4. load reflectance maps
    cv::Mat diff_albedo = cv::imread("../maya_render/Textures/Head_Diffuse_Unlit.exr", cv::IMREAD_UNCHANGED);
    cv::Mat spec_albedo = cv::imread("../maya_render/Textures/Head_Specular.exr", cv::IMREAD_UNCHANGED);
    cv::Mat roughness_map = cv::imread("../maya_render/Textures/Head_Roughness.exr", cv::IMREAD_UNCHANGED);
    cv::Mat displacement_map = cv::imread("../maya_render/Textures/Head_Displacement.exr", cv::IMREAD_UNCHANGED);
    
    // spherical harmonics test (not use)
//     std::vector<std::vector<double> > sh_coeffs(3);
//     cal_SH_from_envmap("../maya_render/env_map/indoor.JPG", sh_coeffs, SH_ORDER);
    
    // median_cut 
    std::vector<float3> lights_pos;
    std::vector<float3> lights_color;
    estimate_light_source("../maya_render/env_map/env-log.jpg", lights_pos, lights_color);  //outdoor.hdr
    //return 0;
    
    
    // render params
    int texture_size = diff_albedo.rows; 
    // render view params
    int img_h = 1920;
    int img_w = 1080;
    int img_c = 3;


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
    cv::Mat normal_map   = cv::Mat::zeros(texture_size, texture_size, CV_32FC3);
    cv::Mat position_map = cv::Mat::zeros(texture_size, texture_size, CV_32FC3);
    render_normal_position(themesh, textureCoordinates, vertexIndices, textureIndices,texture_size, texture_size, 3,  
                        normal_map, position_map);
    
    //cv::imwrite("render/normal_map.exr", normal_map);
    //cv::imwrite("render/position_map.exr", position_map);
    //visualize_normal_map(normal_map, "normal_map.jpg");
    //visualize_position_map(position_map, "position_map.jpg");
    
    
    
    /**************************************************
     * render for each view
     * ***********************************************/
    for(int view_idx = 0; view_idx < 1; view_idx++){
        //image idx
        char buff[100];
        snprintf(buff, sizeof(buff), "%04d", view_idx+1);
        std:string view_idx_str(buff);
        cout<<"use view: "<<view_idx_str<<endl;
        
        
        // depth map
        cv::Mat depth_img;
        render_depth(themesh, cam_P[view_idx], img_h, img_w, depth_img);
        //visualize_depth_map(depth_img, "render/depth_"+view_idx_str+".jpg");
        
        
        // convert image to texture space
        cv::Mat image_view_space = cv::imread("../maya_render/Image/inverse_rendering_1_"+view_idx_str+".png", cv::IMREAD_UNCHANGED);
        cv::Mat image_texture_space;
        render_texture_from_view(themesh, vertexCoordinates, textureCoordinates, vertexIndices, textureIndices,
                                cam_P[view_idx], depth_img, image_view_space, texture_size, image_texture_space);
        //cv::imwrite("render/image_texture_space_"+view_idx_str+".jpg", image_texture_space);
        
        
        /**************************************************
        * render single view
        * ***********************************************/
        cv::Mat render_result;
        //cv::Mat normal_map_detailed = cv::Mat::zeros(texture_size, texture_size, CV_32FC3);
        render_image(themesh, textureCoordinates, textureIndices, cam_P[view_idx], face_TBN, face_TBN_norm,
                    diff_albedo, spec_albedo, roughness_map, displacement_map, 
                    lights_pos, lights_color,
                    img_h, img_w, img_c, render_result);
        
        visualize_render_image(render_result, "render/render_"+view_idx_str+".jpg");
        cv::imwrite("render/render_"+view_idx_str+"_origin.jpg", image_view_space);
    }
    
    
    return 0;
    
}



