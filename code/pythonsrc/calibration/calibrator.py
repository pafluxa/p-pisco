import pisco

import os
import numpy
import json
import pandas


class Calibrator( object ):
    '''
    A pisco.calibrator object is a container that stores coefficients
    that transform RAW dac units to physical units for a given TOD.
    '''
    
    # Calibrator description. 
    #000000000000000000000000000000000000000000000000000000000000000000000000
    name        = ""
    description = ""
    calType     = ""
    # Information stored in the form of a dictionary. Careful not to abuse 
    # of this in the sense of using it to process data!
    info        = {}
    #000000000000000000000000000000000000000000000000000000000000000000000000

    # Calibration coefficients
    coeffs      = numpy.empty(0)

    # Detector index to Unique Identifier array
    __uid         = numpy.empty(0)

    def __init__( self ):

        '''
        self.name = name
        self.description = descrp
        self.calType = calType
        '''

    def setCoeffs( self, c , uid=None ):
        '''
        Set calibrator coefficients to c.
        '''
        # Perform numpy.copy() to avoid cross referencing stuff
        self.__coeffs = numpy.copy( c )

        if uid is not None:
            self.__uid = numpy.copy(uid)
            self.coeffs = self.coeffs[ self.__uid ]
        else:
            self.__uid = numpy.arange( len( self.coeffs ) )

    def getCoeffs( self ):
        '''
        Get a *copy* of the coefficients array.
        '''
        return numpy.copy( self.coeffs )

    def updateInfo( self, prop, value ):
        '''
        Update calibrator info with a pair of prop : value
        '''
        self.info.update( { 'prop' : value } )


    def storeInPath( self , outPath ):
        '''
        Stores the calibrator in JSON format at the specified path.
        '''

        # Serialize this object
        data = {
                'coefficients' : self.__coeffs,
                'uid'          : self.__uid }

        # Create PANDAS DataFrame out data
        df = pandas.DataFrame( data )
        # Save DataFrame to HDF5 format
        df.to_csv( os.path.join( 
            outPath, "%s.%s.cal" % (self.name,self.calType) ), 
            index=False, 
            sep=' ', 
            header=True )

    @classmethod
    def readFromPath( cls, systemPath ):
        '''
        '''
        
        self = cls()

        name,caltype,_  = os.path.basename( systemPath ).split('.')
        self.name = name
        self.calType = caltype
        self.description = ''

        # Load file
        calDF = pandas.read_csv( 
                systemPath,
                header=0,
                names=['coefficients', 'uid'],
                delimiter=' ' ) 

        self.setCoeffs( calDF['coefficients'], uid = calDF['uid'] )
        
        return self
        


