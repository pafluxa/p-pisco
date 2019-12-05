#ifndef __CUDACONVOLUTION
#define __CUDACONVOLUTION


extern void                                                                                               
libconvolve_cuda_deproject_detector                                                                           
(                                                                                                             
    // Number of samples in the pointing stream                                                               
    int nsamples,                                                                                             
                                                                                                              
    // Pointing of the feedhorn                                                                               
    double ra[], double dec[], double pa[],                                                                   
                                                                                                              
    // Detector orientation angles inside the feedhorn                                                        
    double det_pol_angle,                                                                                     
                                                                                                              
    // Evaluation grid specs
    int npix_max, int *num_pixels, int *evalgrid_pixels,
    
    // Beam specs
    int input_beam_nside,
    // first row                                                                                              
    double* reM_TT , double* imM_TT, 
    double* reM_TP , double* imM_TP,
    double* reM_TPs, double* imM_TPs,
    double* reM_TV , double* imM_TV,
    // second row                                                                                             
    double* reM_PT , double* imM_PT, 
    double* reM_PP , double* imM_PP, 
    double* reM_PPs, double* imM_PPs,
    double* reM_PV , double* imM_PV,
    // third row                                                                                              
    double* reM_PsT, double* imM_PsT, 
    double* reM_PsP, double* imM_PsP,
    double* reM_PsPs,double* imM_PsPs, 
    double* reM_PsV ,double* imM_PsV,
    // fourth row                                                                                             
    double* reM_VT , double* imM_VT, 
    double* reM_VP , double* imM_VP, 
    double* reM_VPs, double* imM_VPs, 
    double* reM_VV , double* imM_VV,
    
    //
    int maxPix,

    // Maps with Stokes parameters                                                                            
    int input_map_nside, int input_map_size, float I[], float Q[], float U[], float V[],                      
                                                                                                              
    // Device to use                                                                                          
    int gpu_id,                                                                                               
                                                                                                              
    // Output: ndets x nsamples table with detector data.                                                     
    double tod[]                                                                                              
);

#endif
