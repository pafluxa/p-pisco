#!/bin/bash

# run ellipticall beams, matched pointing, mismatched beams, nside_out=128
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/unpolCMB_ellipticalBeams_mismatchedPointing_mismatchedBeams_128.cfg

# run nominal beams, mismatched pointing, mismatched beams, nside_out=128
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/unpolCMB_ellipticalBeams_matchedPointing_mismatchedBeams_128.cfg
