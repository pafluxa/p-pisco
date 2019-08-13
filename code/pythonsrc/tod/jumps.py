import numpy
import numpy as np
from scipy.optimize import curve_fit

def find_jumps(data, thresh, step):                                                                           
    """                                                                                                       
    Find regions, of with step, where the data jumps by more than                                             
    thresh.  Return a list of such points, centered on the middle                                             
    the jump, roughly.                                                                                        
    """                                                                                                       
    dy = data[step:] - data[:-step]                                                                           
    jump_mask = (abs(dy) > thresh)                                                                            
    changes = (jump_mask[1:] != jump_mask[:-1]).nonzero()[0] + step/2                                         
    if jump_mask[0]:                                                                                          
        changes = np.hstack((0, changes))                                                                     
    if jump_mask[-1]:                                                                                         
        changes = np.hstack((changes, len(data)-1))                                                           
    jump_locs = (changes[::2] + changes[1::2]) / 2                                                            
                                                                                                              
    return jump_locs

def find_jumps_diff(self, thresh = 0.5e6):                                                                    
    '''                                                                                                       
    find jumps and return an array of idx.                                                                    
    Adjacent jump locations are grouped together                                                              
    '''                                                                                                       
    data = self.data                                                                                          
    idx = np.where(abs(np.diff(data))>thresh)[0]                                                              
    jumps_diff = []                                                                                           
    if len(idx) != 0:                                                                                         
        temp = [idx[0]]                                                                                       
        for num in idx[1:]:                                                                                   
            if temp[-1] == (num-1) or temp[-1] == (num-2):                                                    
                temp.append(num)                                                                              
            else:                                                                                             
                jumps_diff.append(temp)                                                                       
                temp = [num]                                                                                  
        jumps_diff.append(temp)                                                                               
    return(jumps_diff) 

def remove_jumps(data,jumps, max_jumps=4 ):                                                                   
    '''                                                                                                       
    remove given jumps and return the fixed data                                                              
    '''                                                                                                       
                                                                                                              
    # Ignore correction if there are too many jumps                                                           
    if len( jumps ) > max_jumps:                                                                              
        raise ValueError, 'Too many jumps. Flag detector as bad.'                                             
        return data                                                                                           
                                                                                                              
    #Fix the jumps                                                                                            
    hspan = 7 #distance away from the jump boundary                                                           
    width = 40 #length of sample to take average as baseline                                                  
                                                                                                              
    for i in range(len(jumps)):                                                                               
        #Take mean on the left                                                                                
        lstart = jumps[i]-hspan-width                                                                         
        lend = jumps[i]-hspan                                                                                 
        average_left = numpy.mean(data[lstart:lend])                                                          
        #Take mean on the right                                                                               
        rstart = jumps[i]+hspan                                                                               
        rend = jumps[i]+hspan+width                                                                           
        average_right = numpy.mean(data[rstart:rend])                                                         
                                                                                                              
        #Move the rest of the data                                                                            
        slope_left = (data[lend] - data[lstart])/(lend - lstart)                                              
        slope_right = (data[rend] - data[rstart])/(rend - rstart)                                             
        slope = (slope_left+slope_right)/2.                                                                   
        delta = slope*(rstart-lstart)                                                                         
        data[rstart:] = data[rstart:] + (average_left-average_right) + delta                                  
        data[lend:rstart] = average_right
        #Pad the junction with sinusoidal fit                                                                 
        #p0 = [1.5e5,0,0,0]#initial guess are given according to VPM modulation                                
        #popt,_ = curve_fit( vpm_sin,numpy.arange(lstart,lend),data[lstart:lend], p0=p0)                       
        #data[lend:rstart] = vpm_sin(numpy.arange(lend,rstart),*popt)                                          
                                                                                                              
    return(data)




