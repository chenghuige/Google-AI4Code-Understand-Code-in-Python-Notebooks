#py predicate-file feature-file
import sys,os

thre = 0.5
fp_filename = 'fp.txt'
fn_filename = 'fn.txt'
tp_filename = 'tp.txt'
tn_filename = 'tn.txt'

if (len(sys.argv) > 3):
	thre = float(sys.argv[3])
print "thre: %f"%thre

if (len(sys.argv) > 4):
	fp_filename = sys.argv[4]

if (len(sys.argv) > 5):
	fn_filename = sys.argv[5]

fp_file = open(fp_filename,'w')
fn_file = open(fn_filename,'w')
tp_file = open(tp_filename,'w')
tn_file = open(tn_filename,'w')

f = open(sys.argv[1])
f.readline()
predicate_list = f.readlines()
score_list = [item.strip().split()[2] for item in predicate_list]
predicate_list = [str(int(float(item.strip().split()[1]) > thre)) for item in predicate_list]

info_list = []
label_list = []
for line in open(sys.argv[2]):
	l = line.strip().split()
	if (l[0] == '+1'):
		l[0] = '1'
	label_list.append(l[0])
	info_list.append(line.strip()) 

tp = 0
fp = 0
tn = 0
fn = 0

for i in range(len(label_list)):
	result = "%s\t%s\n"%(score_list[i], info_list[i])
	if (predicate_list[i] == '1'):
		if (label_list[i] == '1'):
			tp += 1
			tp_file.write(result)
		else:
			fp += 1
			fp_file.write(result)
	else:
		if (label_list[i] == '1'):
			fn += 1
			fn_file.write(result)
		else:
			tn += 1
			tn_file.write(result)


print '%s %s'%(tp, fn)
print '%s %s'%(fp, tn)

print "For label 1: POS"
print "Precision: %f"%(tp * 1.0 / (tp + fp))
print "Recall: %f"%(tp * 1.0 / (tp + fn))

print "For label 0: NEG"
print "Precision: %f"%(tn * 1.0 / (tn + fn))
print "Recall: %f"%(tn * 1.0 / (tn + fp))

print "Total precision: %f"%((tp + tn) * 1.0 / (tp + tn + fp + fn))
