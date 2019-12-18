

import os
from os.path import join as pjoin
import glob
from distutils.core import setup, Extension
import numpy.distutils.misc_util
import subprocess
from subprocess import Popen

# Locate prefix based on Python path
conda_prefix = subprocess.check_output(['which python'], shell=True )
conda_prefix = conda_prefix.split('/')[:-2]

prefix = "/"
for p in conda_prefix:
    prefix = os.path.join( prefix, p )

print 'installing in', prefix

extensions = []

# Set prefix in bash environment for extension compiling
os.environ['PREFIX'] = prefix

# Pointing and pointing correction extensions
# *****************************************************************************
# Compile pointing extension shared library
# Change directory to 'libraries/libpointing'
os.chdir( 'libraries/pointing' )
# Execute makefile
os.system( 'make' )
# Change back to main directory
os.chdir( '../../' )

# Add extension source files
ptg_source_files = glob.glob( './extensions/pointing/src/_pointing.c' )

# Add external library path
ptg_lib_dirs = []
ptg_lib_dirs.append( os.path.join( prefix, 'lib' ) )

# Add include directories
ptg_inc_dirs = []
ptg_inc_dirs.append( os.path.join( prefix, 'include' ) )
ptg_inc_dirs.append( numpy.distutils.misc_util.get_numpy_include_dirs()[0] )
ptg_inc_dirs.append( './extensions/pointing/include' )
ptg_inc_dirs.append( os.path.join( prefix, 'include/pisco/pointing' ) )

# Setup Python Extension object
_pointing =  Extension( 
        "pisco.pointing._pointing", 
        ptg_source_files ,
        language = "c",
        libraries=['pointing', 'sofa_c'],
        library_dirs = ptg_lib_dirs ,
        include_dirs = ptg_inc_dirs , 
        extra_compile_args=["-fopenmp", '-std=c99', '-lm', '-fPIC', '-Wall'] , )
extensions.append( _pointing )

# Add extension source files
ptgcorr_source_files = glob.glob( './extensions/pointing/src/_pointing_correction.c' )

# Add external library path
ptgcorr_lib_dirs = []
ptgcorr_lib_dirs.append( os.path.join( prefix, 'lib' ) )

# Add include directories
ptgcorr_inc_dirs = []
ptgcorr_inc_dirs.append( os.path.join( prefix, 'include' ) )
ptgcorr_inc_dirs.append( numpy.distutils.misc_util.get_numpy_include_dirs()[0] )
ptgcorr_inc_dirs.append( './extensions/pointing/include' )
ptgcorr_inc_dirs.append( os.path.join( prefix, 'include/pisco/pointing' ) )

# Setup Python Extension object
_pointing_correction =  Extension( 
        "pisco.pointing._pointing_correction", 
        ptgcorr_source_files ,
        language = "c",
        libraries=['sofa_c', 'pointing'],
        library_dirs = ptgcorr_lib_dirs ,
        include_dirs = ptgcorr_inc_dirs , 
        extra_compile_args=["-fopenmp", '-std=c99', '-lm', '-fPIC', '-Wall'] , )
extensions.append( _pointing_correction )
# Mapping extension
# *****************************************************************************

# Compile mapping extension shared library
# Change directory to 'libraries/libmapping'
os.chdir( 'libraries/mapping' )
# Execute makefile
os.system( 'make' )
# Change back to main directory
os.chdir( '../../' )

# Add extension source files
map_source_files = glob.glob( './extensions/mapping/src/*.c' )

# Add external library path
map_lib_dirs = []
map_lib_dirs.append( os.path.join( prefix, 'lib' ) )

# Add include directories
map_inc_dirs = []
map_inc_dirs.append( os.path.join( prefix, 'include' ) )
map_inc_dirs.append( numpy.distutils.misc_util.get_numpy_include_dirs()[0] )
map_inc_dirs.append( './extensions/mapping/include' )
map_inc_dirs.append( os.path.join( prefix, 'include/pisco/mapping' ) )

# Setup Python Extension object
_mapping =  Extension( 
        "pisco.mapping._mapping", 
        map_source_files ,
        language = "c",
        libraries=['mapping', 'chealpix', 'lapack', 'blas'],
        library_dirs = map_lib_dirs ,
        include_dirs = map_inc_dirs , 
        extra_compile_args=["-fopenmp", '-std=c99', '-lm', '-fPIC', '-Wall'] , )
# *****************************************************************************
extensions.append( _mapping )
# *****************************************************************************

# Convolution extension
# *****************************************************************************
# Compile convolution extension shared library
# Change directory to 'libraries/libmapping'
os.chdir( 'libraries/convolution' )
# Execute makefile
os.system( 'make' )
# Change back to main directory
os.chdir( '../../' )

# Add extension source files
conv_source_files = glob.glob( './extensions/convolution/src/*.c' )

# Add external library path
conv_lib_dirs = []
conv_lib_dirs.append( os.path.join( prefix, 'lib' ) )

# Add include directories
conv_inc_dirs = []
conv_inc_dirs.append( os.path.join( prefix, 'include' ) )
conv_inc_dirs.append( numpy.distutils.misc_util.get_numpy_include_dirs()[0] )
conv_inc_dirs.append( './extensions/convolution/include' )
conv_inc_dirs.append( os.path.join( prefix, 'include/pisco/convolution' ) )

# Setup Python Extension object
_convolution =  Extension( 
        "pisco.convolution._convolution", 
        conv_source_files ,
        language = "c",
        libraries=['convolution', 'chealpix', 'healpix_cxx', 'cxxsupport', 'c_utils'],
        library_dirs = conv_lib_dirs ,
        include_dirs = conv_inc_dirs , 
        extra_compile_args=["-fopenmp", '-std=c99', '-lm', '-fPIC', '-Wall'] , )
# *****************************************************************************
extensions.append( _convolution )

# *****************************************************************************
# Disutils setup specs
# *****************************************************************************
setup(
    name = 'pisco',
    package_dir = \
        { 'pisco'             : 'pythonsrc'   ,
          'pisco.experiments' : 'experiments' },

    packages = ['pisco',
                # Core modules
                #=============================================
                'pisco.beam_analysis',
                'pisco.pointing',
                'pisco.calibration',
                'pisco.mapping',
                'pisco.convolution',
                'pisco.tod',
                #=============================================
                #
                #
                # Instrument specific stuff. Add new ones here
                #=============================================
                'pisco.experiments',
                'pisco.experiments.class_telescope',
                'pisco.experiments.class_telescope.input_output',
                'pisco.experiments.class_telescope.calibration',
                'pisco.experiments.class_telescope.misc',
                #=============================================
               ],
    author  = 'Pedro A. Fluxa Rojas.',
    author_email = 'pafluxa@astro.puc.cl',
    version = '0.2',
    ext_modules = extensions,   
    )
# *****************************************************************************
