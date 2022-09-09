"""
example:
    python ./Domain-motif-interaction.py ppis_list.txt sequeces.fasta
    ppis_list.txt:ppis list file,one ppi record one line
    sequeces.fasta:fasta format sequences all need.
"""
import numpy as np
import pandas as pd
from Bio import SeqIO
import os
import sys
import re
from multiprocessing import Pool
sys.path.append("..")

ppis_file = "../data/ind_2021/p1n10_ppis_ind.txt"
seqs_db = ["../data/ppis_total.fasta","../data/ind_2021/ppis_ind.fasta"]
pfam_db_file = "../features/script/get_dmi2node2vec/pfam_db/"

dmi_db_file="../features/script/get_dmi2node2vec/3did_dmi_flat.gz"


ppis = pd.read_table(ppis_file).to_numpy()

#(1) anotate domains
seqs = {i.id : str(i.seq) for file_name in seqs_db for i in SeqIO.parse(file_name,"fasta")}
if not os.path.exists("/tmp/dmi"):
    os.makedirs("/tmp/dmi")

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


#(2) anotate motifs
cmd = f"zless {dmi_db_file}|grep -E '#=PT'|sed 's/\s\s*/\t/g'|cut -f 2|sort -u"
pattern_db = [line.strip("\n") for line in os.popen(cmd).readlines()]

def pt_search(k,v, pattern_db=pattern_db):
    re_motifs = []
    for pt in pattern_db:
        match_obj = re.search(pt,v)
        if match_obj:
            re_motifs.append(pt)
    return k,set(re_motifs)

p = Pool(16)
motifs = p.starmap(pt_search,seqs.items())
motifs = {k:v for (k,v) in motifs}
p.close()


#(3) load dmi db
dmis = set()
cmd = "zless ../features/script/get_dmi2node2vec/3did_dmi_flat.gz|\
    grep -E '#=ID|#=PT'|\
    sed -e 's/\s\s*/\t/g'|\
    cut -f 1,2|\
    awk '{if($0~/ID/){printf $2}else{print \"\t\" $2}}'"
dmis =set( [tuple(line.strip("\n").split("\t")) for line in os.popen(cmd).readlines() ])


#(4) search dmi db
ppis_pred = []
if len(ppis[0]) == 2:
    for a,b in ppis[:,:2]:
        pred_label=0
        if a in domains.keys() and b in motifs.keys():
            c_ppi = set([(i,j) for i in domains[a] for j in motifs[b]])
            if c_ppi & dmis:
                pred_label=1

        if b in domains.keys() and a in motifs.keys():
            c_ppi = set([(i,j) for i in domains[b] for j in motifs[a]])
            if c_ppi & dmis:
                pred_label=1

        ppis_pred.append((a,b,str(pred_label)))
else:
    for a,b,label in ppis:
        pred_label=0
        if a in domains.keys() and b in motifs.keys():
            c_ppi = set([(i,j) for i in domains[a] for j in motifs[b]])
            if c_ppi & dmis:
                pred_label=1

        if b in domains.keys() and a in motifs.keys():
            c_ppi = set([(i,j) for i in domains[b] for j in motifs[a]])
            if c_ppi & dmis:
                pred_label=1

        ppis_pred.append((a,b,str(label),str(pred_label)))

for i in ppis_pred:
    print("\t".join(i))






