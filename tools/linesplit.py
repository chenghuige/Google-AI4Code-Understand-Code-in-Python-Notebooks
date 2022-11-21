#!/bin/env python

# split cooc according line number.

# config section

#usage
#./linesplit.py inputfile outputdir machine_num prefix  #machine_num same as how may to split, prefix is like cooc or rcooc
#notice split use ceil //up qu zheng

import sys, os, math

input_file_name = sys.argv[1]
output_directory = sys.argv[2]
split_parts_num = int(sys.argv[3])
total_line_num = int(os.popen('wc -l %s'%sys.argv[1]).read().split()[0])

line_per_file = int(math.ceil(float(total_line_num)/split_parts_num))

output_file_prefix = "%s_"%sys.argv[4]


print "Begin split......"
fd = open(input_file_name, "r+")

current_file_no = 0 
current_fd = open("%s/%s%d" % (output_directory, output_file_prefix, current_file_no), "wb")
current_line_no = 0
for line in fd:
    current_fd.write(line)
    current_line_no += 1
    if current_line_no == line_per_file:
        current_fd.close()
        current_file_no += 1
        current_line_no = 0
        current_fd = open("%s/%s%d" % (output_directory, output_file_prefix, current_file_no), "wb")
        
current_fd.close()
fd.close()

print "Done"

