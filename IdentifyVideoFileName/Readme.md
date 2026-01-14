Scripts to identify labels printed on yellow sticky notes for video recordings of cylinder, grid walk, and rotating beam test. Label is identified and file name updated with the correct id. Format: studyID_time point (e.g., GV_T3_3_1_Baseline).

For the swift version run first 

swiftc -O VideoIdentFileName_v4_GW.swift -o VideoIdentFileName_v4_GW

and then

./VideoIdentFileName_v4_GW "[input path]" \              
  --mode subject_stage \
  --behavior Gridwalk \
  --step 0.5 \
  --out-dir "[output path]"
