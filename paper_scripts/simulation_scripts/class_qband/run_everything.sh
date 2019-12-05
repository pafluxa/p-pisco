#!/bin/bash

# run nominal beams, matched pointing
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/nominalBeams_matchedPointing_matchedBeams_128.cfg

# run nominal beams, mismatched pointing
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/nominalBeams_mismatchedPointing_matchedBeams_128.cfg

# run elliptical beams, matched pointing
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/ellipticalBeams_matchedPointing_matchedBeams_128.cfg

# run elliptical beams, mismatched pointing
python run_pisco_simulation_nside_128.py -config ../../configs/class_qband/ellipticalBeams_mismatchedPointing_matchedBeams_128.cfg

# NSIDE 256
# run nominal beams, matched pointing
python run_pisco_simulation_nside_256.py -config ../../configs/class_qband/nominalBeams_matchedPointing_matchedBeams_256.cfg

# run nominal beams, mismatched pointing
python run_pisco_simulation_nside_256.py -config ../../configs/class_qband/nominalBeams_mismatchedPointing_matchedBeams_256.cfg

# run elliptical beams, matched pointing
python run_pisco_simulation_nside_256.py -config ../../configs/class_qband/ellipticalBeams_matchedPointing_matchedBeams_256.cfg

# run elliptical beams, mismatched pointing
python run_pisco_simulation_nside_256.py -config ../../configs/class_qband/ellipticalBeams_mismatchedPointing_matchedBeams_256.cfg


