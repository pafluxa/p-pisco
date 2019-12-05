#!/bin/bash

# run ellipticall beams, mismatched pointing, matched beams, nside_out=128
#python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/unpolCMB_ellipticalBeams_mismatchedPointing_matchedBeams_128.cfg

# run elliptical beams, matched pointing, matched beams, nside_out=128
#python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/unpolCMB_ellipticalBeams_matchedPointing_matchedBeams_128.cfg

# run ellipticall beams, matched pointing, matched beams, nside_out=128
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/ellipticalBeams_matchedPointing_matchedBeams_128.cfg

# run elliptical beams, mismatched pointing, matched beams, nside_out=128
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/ellipticalBeams_mismatchedPointing_matchedBeams_128.cfg
