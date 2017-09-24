import sys
import operator

file_name1 = sys.argv[1]
file_name2 = sys.argv[2]
file_name3 = sys.argv[3]
file_name4 = sys.argv[4]
file_name5 = sys.argv[5]
file_name6 = sys.argv[6]

input_file1 = open(file_name1,'r')
input_file2 = open(file_name2,'r')
input_file3 = open(file_name3,'r')
input_file4 = open(file_name4,'r')
input_file5 = open(file_name5,'r')

out_file = open(file_name6,'w')

lines1 = input_file1.readlines()
lines2 = input_file2.readlines()
lines3 = input_file3.readlines()
lines4 = input_file4.readlines()
lines5 = input_file5.readlines()

for index in range(len(lines1)):
        args1 = lines1[index].strip().split()
        args2 = lines2[index].strip().split()
        args3 = lines3[index].strip().split()
        args4 = lines4[index].strip().split()
        args5 = lines5[index].strip().split()

        if len(args1)!=0:
                word = args1[0]
                true_tag = args1[1]
                pred1 = args1[2]
                pred2 = args2[2]
                pred3 = args3[2]
                pred4 = args4[2]
                pred5 = args5[2]
                tagnames= {}
                tagnames['O'] = 0
                tagnames['BLOC'] = 0
                tagnames['ILOC'] = 0
                tagnames['ELOC'] = 0
                tagnames['SLOC'] = 0
                tagnames['BORG'] = 0
                tagnames['IORG'] = 0
                tagnames['SORG'] = 0
                tagnames['EORG'] = 0
                tagnames['BPER'] = 0
                tagnames['IPER'] = 0
                tagnames['SPER'] = 0
                tagnames['EPER'] = 0
                tagnames['BMISC'] = 0
                tagnames['IMISC'] = 0
                tagnames['SMISC'] = 0
                tagnames['EMISC'] = 0
                tagnames[pred1] +=1
                tagnames[pred2] +=1
                tagnames[pred3] +=1
                tagnames[pred4] +=1
                tagnames[pred5] +=1
                final_tag = max(tagnames.iteritems(), key=operator.itemgetter(1))[0]
                out_file.write(word + ' ' + true_tag + ' ' + final_tag + '\n')
