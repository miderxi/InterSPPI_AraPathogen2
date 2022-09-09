"""
example:
    python ./Domain-Domain-interaction.py ppis_list.txt sequeces.fasta
    ppis_list.txt:one ppi record one line
    sequences.fasta:fasta file
"""
import numpy as np
import pandas as pd
from Bio import SeqIO
import os
import sys

ppis_file = "../data/ind_2021/p1n10_ppis_ind.txt"
seqs_db = ["../data/ppis_total.fasta","../data/ind_2021/ppis_ind.fasta"]
pfam_db_file = "../features/script/get_dmi2node2vec/pfam_db/"
DDI_db_file= "../features/script/get_dmi2node2vec/3did_flat.gz"

try:
    ppis_file = sys.argv[1]
    seqs_db = sys.argv[2]
except:
    pass

#(1) read ppi
ppis = pd.read_table(ppis_file).to_numpy()

#(2) anotate domains
seqs = {i.id : str(i.seq) for file_name in seqs_db for i in SeqIO.parse(file_name,"fasta")}
if not os.path.exists("/tmp/ddi"):
    os.makedirs("/tmp/ddi")

with open("/tmp/ddi/seqs.fasta","w") as f:
    seqs_ids = set([j for i in ppis[:,:2] for j in i])
    for seq_id in seqs_ids:
        f.write(f">{seq_id}\n")
        f.write(f"{seqs[seq_id]}\n")

cmd = f"pfam_scan.pl -fasta /tmp/ddi/seqs.fasta  -dir {pfam_db_file}  -cpu 16  > /tmp/ddi/seqs.domain"
os.system(cmd)

domains = {}
for line in pd.read_table("/tmp/ddi/seqs.domain").to_numpy()[27:]:
    line = line[0].split()
    k,v = line[0], line[6]
    if k not in domains.keys():
        domains[k]=set([v,])
    else:
        domains[k].add(v)

#(3) search DDI db
ppis_pred = []
ddis = set()
cmd="zless ../A4_DDI_and_DMI/3did_flat.gz|grep '#=ID'|cut -f 2,3|sed 's/  */\t/g' "
for (i,j) in  [line.strip("\n").split("\t") for line in os.popen(cmd).readlines()]:
    ddis.add((i,j))

if len(ppis[0]) == 2:
    for a,b in ppis[:,:2]:
        if a in domains.keys() and b in domains.keys():
            c_ppi = set(
                    [(i,j) for i in domains[a] for j in domains[b]]+
                    [(j,i) for i in domains[a] for j in domains[b]])
            if c_ppi & ddis:
                ppis_pred.append((a,b,"1"))
                print(a,b,1)
        else:
            ppis_pred.append((a,b,"0"))
else:
    for a,b,label in ppis:
        if a in domains.keys() and b in domains.keys():
            c_ppi = set(
                    [(i,j) for i in domains[a] for j in domains[b]]+
                    [(j,i) for i in domains[a] for j in domains[b]])
            if c_ppi & ddis:
                ppis_pred.append((a,b,str(label),"1"))
                #print(a,b,str(label),"1")
        else:
            ppis_pred.append((a,b,str(label),"0"))


for i in ppis_pred:
    print("\t".join(i))


