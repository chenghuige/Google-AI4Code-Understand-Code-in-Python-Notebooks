sed -i 's/USE_GPROF=1/USE_GPROF=0/' COMAKE
sed -i 's/LEVEL=3/LEVEL=0/' COMAKE
comake2 -P
