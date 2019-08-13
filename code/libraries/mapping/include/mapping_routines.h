#ifndef __TODSYNTHMAPPINGH__
#define __TODSYNTHMAPPINGH__

void
libmapping_project_data_to_matrix_cube
(
    // input
    int nsamples , int ndets,
    double ra[], double dec[], double pa[],
    double pol_angles[],
    double data[] , int data_sample_mask[], int det_mask[],
    int map_nside,
    int map_size , int pixels_in_the_map[],
    // output
    double AtA_cube[], double AtD_cube[]
);

void                                                                                                          
libmapping_project_data_to_matrices                                                                           
(                                                                                                             
    // input                                                                                                  
    int nsamples , int ndets, 
    double phi[], double theta[], double psi[],  
    double det_pol_angles[],
    double data[] , int bad_data_samples[], int dets_to_map[], 
    int map_nside,                                                                                            
    int map_size , int pixels_in_the_map[],
    // output                                                                                              
    double AtA[], double AtD[]
);

void                                                                                                          
libmapping_get_IQU_from_matrices                                                                              
(                                                                                                             
    // input                                                                                                  
    int map_nside,                                                                                            
    int map_size ,                                                                                            
    double AtA[], double AtD[], int pixels_in_the_map[],                                                      
    // output                                                                                                 
    double I[], double Q[], double U[], double W[]                                                            
);

#endif
