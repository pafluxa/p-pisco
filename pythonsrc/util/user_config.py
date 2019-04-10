# This is much more a script than a python class, but should
# work just fine.

import ConfigParser

class userConfig( object ):


    # Inherited from moby2: the warehouse
    warehousePath = ''

    # Instrument being used
    instrument = ''
    
    # Receivers in the instrument. Just a dictionary
    # mapping a receiver name to a warehouse-relative path
    # with the data file to assemble it.
    receivers  = {}

    @classmethod
    def parse( cls, path ):
        '''
        Initializes a userConfiguration object. 

        input
            
            path : system path with user configuration. It must follow
                   Python's configParser module rules, and should at least
                   contain the following directives.
        '''

        self = cls()

        uconf = ConfigParser.ConfigParser()
        uconf.read( path )

        if 'warehouse' not in uconf.sections():
            raise RuntimeError, 'invalid configuration file!'
        if 'instrument' not in uconf.sections():
            raise RuntimeError, 'invalid configuration file!'
        if 'receivers' not in uconf.sections():
            raise RuntimeError, 'invalid configuration file!'

        # Get warehouse path.
        try:
            self.warehouse = uconf.get( 'warehouse', 'path' )
        except:
            print 'invalid configuration file:'
            raise RuntimeError, '\twarehouse section must have a "path" option.'

        # Get instrument name.
        try:
            self.instrument = uconf.get( 'instrument', 'name' )
        except:
            print 'invalid configuration file:'
            raise RuntimeError, '\tinstrument section must have a "name" option.'

        # Build receivers dict.
        for rname in uconf.options( 'receivers' ):
            #self.receivers.update( { rname , uconf.get( 'receivers', rname ) } )
            path = uconf.get( 'receivers', rname )
            self.receivers.update( { rname : path} )


        return self
            
        






