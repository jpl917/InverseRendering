#include "spherical_harmonics.h"
#include "common.h"

#include <iostream>
#include <vector>

int main(){
    
    std::vector<std::vector<double> > sh_coeffs(3);
    cal_SH_from_envmap("../maya_render/env_map/indoor.JPG", sh_coeffs, SH_ORDER);
    
    for(int i = 0; i < 3; i++){
        for(int j = 0; j < SH_ORDER * SH_ORDER; j++){
            std::cout<<sh_coeffs[i][j]<<" ";
        }
        std::cout<<std::endl;
    }
    
    return 0;
    
}
