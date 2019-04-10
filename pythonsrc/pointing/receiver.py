import os
import numpy
import pandas

class Receiver( object ):
    '''
    A receiver class for TODsynth.
    '''

    def __init__( self ):
		
		pass
    
    def initialize( self, uid, daz, dalt, rot ):
        '''
        Initialize receiver object.

        input:
            uid : map between array indexes and Unique IDs for detectors. For instance, 
            a receiver might have detector 15 at the 89th place in the arrays. Then,
            uid[ 15 ] = 89

            This way one can access the different receiver attributes doing 

            r = Receiver( ... )
            r.daz[ r.uid ] 

            which will be the 'standard' indexing of the experiment.

            daz/dalt : if receiver is placed at 0 elevation, 0 azimuth and 0 boresight rotation,
            the on-sky offsets would be *exactly* daz and dalt.

            pol_angles : detector rotation angle at the focal plane. This is the detector's polarization 
            sensitive angle.
        '''


        self.uid =  numpy.asarray( uid , dtype='int32' )
        self.dx  =  numpy.asarray( daz , dtype='float64' )
        self.dy  =  numpy.asarray( dalt, dtype='float64' )

        self.pol_angles = numpy.asarray( rot, dtype='float64' )

        self.mask = numpy.zeros_like( self.uid , dtype='int32' )
        
        self.ndets = self.uid.size

    def toggle_detectors( self, detList ):
        '''
        Enables/disables a list of detectors by their UIDs. By default,
        all detectors in the receiver are enabled. This routine makes it
        possible to select which detectors will be used for pointing 
        calculations, which include projections/deprojections as well.

        input:
            detList : list of UIDs of detectors to toggle. By default, all
                      detectors in the Receiver are set to active. One can
                      access this information via the Receiver.mask attribute.
        '''
		
        for d in detList:
            self.mask[ d ] = int( not bool( self.mask[d] ) )
