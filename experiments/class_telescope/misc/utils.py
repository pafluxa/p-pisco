'''
Utility module, for things like this
'''
import os

def expand_tod_name_to_path( basepath, todname ):

    expanded_path = basepath
    # extract year, month, day, hour, minute and second to create complete path
    timestamp = todname.split( '-' )
    if len( timestamp ) != 6:
        raise ValueError, '%s is not a valid tod name' % (todname)
    year,month,day,hour,minute,seconds = timestamp
    
    # build complete path
    expanded_path = os.path.join( 
        expanded_path, 
        '%s-%s/%s-%s-%s/%s-%s-%s-%s-%s-%s' % (year,month,year,month,day,year,month,day,hour,minute,seconds) )

    return expanded_path     
