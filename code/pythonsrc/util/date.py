import numpy
import time                                                                                                   
import datetime                                                                                               
                                                                                                              
from mpi4py import MPI                                                                                        
                                                                                                              
def dateToCtime( date, roundUp=False ):                                                                       
                                                                                                              
    yyyy,mm,dd,HH,MM,SS = map( int, date.split( '-' ) )                                                       
                                                                                                              
    SS = 0                                                                                                    
    t = time.strptime(                                                                                        
            "%04d-%02d-%02dT%02d-%02d-%02d" % (yyyy,mm,dd,HH,MM,SS),                                          
            "%Y-%m-%dT%H-%M-%S" )                                                                             
                                                                                                              
    return time.mktime( t ) 
