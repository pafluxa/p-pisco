__author__ = "Pedro Fluxa R."
__email__  = "pedro@fluxanet.net"

import numpy
import pylab

class Beamsor( object ):

    def __init__( self, fields, base='v-h' ):
        '''
        '''

        self.fields = fields

    def build( self ):
        '''
        '''

        E_hh, E_hv, E_vv, E_vh = self.fields
        
        # Build the MMF in "Stokes space"
        M = numpy.empty( (4,4,E_hh.size) )

        M[0][0] = numpy.real( self.compute_MTT() )
        M[0][1] = numpy.real( self.compute_MTQ() )
        M[0][2] = numpy.real( self.compute_MTU() )
        M[0][3] = numpy.real( self.compute_MTV() )
        
        M[1][0] = numpy.real( self.compute_MQT() )
        M[1][1] = numpy.real( self.compute_MQQ() )
        M[1][2] = numpy.real( self.compute_MQU() )
        M[1][3] = numpy.real( self.compute_MQV() )
        
        M[2][0] = numpy.real( self.compute_MUT() )
        M[2][1] = numpy.real( self.compute_MUQ() )
        M[2][2] = numpy.real( self.compute_MUU() ) 
        M[2][3] = numpy.real( self.compute_MUV() )
        
        M[3][0] = numpy.real( self.compute_MVT() )
        M[3][1] = numpy.real( self.compute_MVQ() )
        M[3][2] = numpy.real( self.compute_MVU() )
        M[3][3] = numpy.real( self.compute_MVV() ) 

        self.M  = M

        return self

    def plot( self , fig=None, reshape=None, log=False, limits=None, output=None, clims_TT=None, clims_XX=None ):

        if clims_TT is None:
            clims_TT = [ 0.0, numpy.max( self.M[0][0] ) ]
        
        if clims_XX is None:
            clims_XX = clims_TT

        if fig is None:
            pylab.figure()
        
        count = 1
        for i in range(0,4):
            
            for j in range(0,4):
            
                data = self.M[i][j]
                
                pylab.subplot( 4, 4, count )

                if reshape is not None:
                    data = data.reshape( reshape )
                
                if log:
                    pylab.imshow( 10*numpy.log10(data), origin='lower', extent=limits )
                    pylab.imshow( 10*numpy.log10(data), origin='lower', extent=limits )
                else:
                    pylab.imshow( data, origin='lower', extent=limits )
                    pylab.colorbar()
                
                if count in [1,6,11,16]:
                    pylab.clim( clims_TT )
                else:
                    pylab.clim( clims_XX )

                pylab.xlim( -5,5,-5,5 )
                pylab.ylim( -5,5,-5,5 )

                count = count + 1

        if output is None:
            pylab.show()
        else:
            pylab.savefig( output )
    

    def compute_MTT( self ):
        '''
        Computes M_TT according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff
        pwr_a = numpy.abs(Ex_co)**2 + numpy.abs(Ex_cx)**2
        pwr_b = numpy.abs(Ey_co)**2 + numpy.abs(Ey_cx)**2

        M_TT = 0.5*( pwr_a + pwr_b )

        return M_TT

    def compute_MTQ( self ):
        '''
        Computes M_TQ according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff
        pwr_x_co =  Ex_co * numpy.conj( Ex_co )
        pwr_x_cx =  Ex_cx * numpy.conj( Ex_cx )

        pwr_y_co =  Ey_co * numpy.conj( Ey_co )
        pwr_y_cx =  Ey_cx * numpy.conj( Ey_cx )

        M_TQ = 0.5*( pwr_x_co - pwr_x_cx + pwr_y_cx - pwr_y_co  )

        return M_TQ


    def compute_MTU( self ):
        '''
        Computes M_TU according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields
        
        # Compute stuff
        M_TU = 0.5*( Ex_co*numpy.conj(Ex_cx) - Ey_co*numpy.conj(Ey_cx) )
        M_TU += numpy.conj(M_TU)

        return M_TU

    def compute_MTV( self ):
        '''
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff
        M_TV = 1/2. * 1.j*( Ex_co*numpy.conj(Ex_cx) + Ey_co*numpy.conj(Ey_cx) )
        M_TV += numpy.conj(M_TV)

        return M_TV

    def compute_MQT( self ):
        '''
        Computes M_TQ according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields
        # Compute stuff
        pwr_a = (Ex_co + Ex_cx )*numpy.conj(Ex_co + Ex_cx)
        pwr_b = (Ey_co + Ey_cx )*numpy.conj(Ey_co + Ey_cx)

        M_QT = 0.5*( pwr_a - pwr_b )

        return M_QT


    def compute_MQQ( self ):
        '''
        Computes M_TQ according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff
        pwr_x_co =  Ex_co * numpy.conj( Ex_co )
        pwr_x_cx =  Ex_cx * numpy.conj( Ex_cx )

        pwr_y_co =  Ey_co * numpy.conj( Ey_co )
        pwr_y_cx =  Ey_cx * numpy.conj( Ey_cx )

        M_QQ = 0.5*( pwr_x_co - pwr_x_cx + pwr_y_co - pwr_y_cx )
        
        return M_QQ

    def compute_MQU( self ):
        '''
        Computes M_QU according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff

        M_QU = 0.5*( Ex_co*numpy.conj(Ex_cx) + Ey_co*numpy.conj(Ey_cx) )
        M_QU += numpy.conj(M_QU)

        return M_QU

    def compute_MQV( self ):
        '''
        Computes M_QV according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff

        M_QV = 0.5*1j*( Ex_co*numpy.conj(Ex_cx) - Ey_co*numpy.conj(Ey_cx) )
        M_QV += numpy.conj(M_QV)

        return M_QV

    def compute_MUT( self ):
        '''
        Computes M_UT according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff
        M_UT = 0.5*( -Ex_co*numpy.conj(Ey_cx) + Ex_cx*numpy.conj(Ey_co) )
        M_UT += numpy.conj(M_UT)

        return M_UT

    def compute_MUQ( self ):
        '''
        Computes M_UQ according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff

        M_UQ = 0.5*( -Ex_co*numpy.conj(Ey_cx) - Ex_cx*numpy.conj(Ey_co) )
        M_UQ += numpy.conj(M_UQ)

        return M_UQ

    def compute_MUU( self ):
        '''
        Computes M_UU according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields
       
        # Compute stuff
        M_UU = 0.5*( Ex_co*numpy.conj(Ey_co) - Ex_cx*numpy.conj(Ey_cx) )
        M_UU += numpy.conj(M_UU)

        return M_UU


    def compute_MUV( self ):
        '''
        Computes M_UV according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff

        M_UV = 0.5j*( Ex_co*numpy.conj(Ey_co) + Ex_cx*numpy.conj(Ey_cx) )
        M_UV += numpy.conj(M_UV)

        return M_UV

    def compute_MVT( self ):
        '''
        Computes M_UV according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff

        M_VT = 0.5j*( Ex_co*numpy.conj(Ey_cx) - Ex_cx*numpy.conj(Ey_co) )
        M_VT += numpy.conj(M_VT)

        return M_VT

    def compute_MVQ( self ):
        '''
        Computes M_UV according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff
        M_VQ = 0.5j*( Ex_co*numpy.conj(Ey_cx) + Ex_cx*numpy.conj(Ey_co) )
        M_VQ += numpy.conj(M_VQ)

        return M_VQ

    def compute_MVU( self ):
        '''
        Computes M_UV according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields

        # Compute stuff

        M_VU = 0.5j*( -Ex_co*numpy.conj(Ey_co) + Ey_cx*numpy.conj(Ex_cx) )
        M_VU += numpy.conj(M_VU)

        return M_VU

    def compute_MVV( self ):
        '''
        Computes M_UV according to https://arxiv.org/pdf/astro-ph/0610361.pdf.
        See equation 22.1
        '''
        # Extract fields
        Ex_co, Ex_cx, Ey_co, Ey_cx = self.fields
        # Compute stuff
        M_VV = 0.5*( Ex_co*numpy.conj(Ey_co) + Ex_cx*numpy.conj(Ey_cx) )
        M_VV += numpy.conj(M_VV)

        return M_VV
