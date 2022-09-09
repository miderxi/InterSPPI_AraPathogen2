"""
usage example:
    python ./interolog.py ppis_list.txt sequeces.fasta
    ppis_list.txt: one line one ppi record
    sequeces.fasta: fasta format sequece all need in ppi_list.txt file
"""
import numpy as np
import pandas as pd
from Bio import SeqIO
import os
import sys

ppis_file = "../data/ind_2021/p1n10_ppis_ind.txt"
seqs_db = ["../data/ppis_total.fasta","../data/ind_2021/ppis_ind.fasta"]


ppi_db_file = "./interolog_ppis_db/total_ppi.txt"
ppi_blastdb = "./interolog_ppis_db/blastdb/ppis_db"

try:
    ppis_file = sys.argv[1]
    seqs_db = sys.argv[2]
except:
    pass

#1. search homolog
ppis = pd.read_table(ppis_file).to_numpy()
seqs = {i.id : str(i.seq) for file_name in seqs_db for i in SeqIO.parse(file_name,"fasta")}

if not os.path.exists("/tmp/interolog"):
    os.makedirs("/tmp/interolog/")

with open("/tmp/interolog/ara.fasta","w") as f:
    ara_ids = set(ppis[:,0])
    for ara_id in ara_ids:
        f.write(f">{ara_id}\n")
        f.write(f"{seqs[ara_id]}\n")

with open("/tmp/interolog/eff.fasta","w") as f:
    eff_ids = set(ppis[:,1])
    for eff_id in eff_ids:
        f.write(f">{eff_id}\n")
        f.write(f"{seqs[eff_id]}\n")

cmd1 = f"blastp -query /tmp/interolog/eff.fasta -db {ppi_blastdb} -out /tmp/interolog/eff.blastp -evalue 1e-2 -outfmt 6 -num_threads 16"
cmd2 = f"blastp -query /tmp/interolog/ara.fasta -db {ppi_blastdb} -out /tmp/interolog/ara.blastp -evalue 1e-2 -outfmt 6 -num_threads 16"

os.system(cmd1)
os.system(cmd2)

homo = {}
homo_self = {}
for line in pd.read_table("/tmp/interolog/ara.blastp").to_numpy():
    if line[2] > 40 and line[2] < 100:
        if line[0] not in homo.keys():
            homo[line[0]] = set([line[1]])
        else:
            homo[line[0]].add(line[1])
    
    if line[2] == 100:
        if line[0] not in homo_self.keys():
            homo_self[line[0]] = set([line[0],line[1]])
        else:
            homo_self[line[0]].add(line[1])

for line in pd.read_table("/tmp/interolog/eff.blastp").to_numpy():
    if line[2] > 30 and line[2] < 100:
        if line[0] not in homo.keys():
            homo[line[0]] = set([line[1]])
        else:
            homo[line[0]].add(line[1])
    
    if line[2] == 100:
        if line[0] not in homo_self.keys():
            homo_self[line[0]] = set([line[0],line[1]])
        else:
            homo_self[line[0]].add(line[1])


#2. search ppi db
ppis_db =set( [ (i,j) for (i,j) in pd.read_table(ppi_db_file).to_numpy()])
ppis_pred = []

if len(ppis[0]) == 2:
    for a,b in ppis[:,:2]:
        if a in homo.keys() and b in homo.keys():
            c_ppis = set([(i,j) for i in  homo[a] for j in homo[b]] + [(j,i) for i in  homo[a] for j in homo[b]]) 
            if c_ppis & ppis_db :
                ppis_pred.append((a,b,"1"))
        else:
            ppis_pred.append((a,b,"0"))
else:
    for a,b,label in ppis:
        if a in homo.keys() and b in homo.keys():
            c_ppis = set([(i,j) for i in  homo[a] for j in homo[b]] + [(j,i) for i in  homo[a] for j in homo[b]]) 
            if c_ppis & ppis_db :
                ppis_pred.append((a,b,label,1))
        else:
            ppis_pred.append((a,b,label,0))

for i in ppis_pred:
    print("\t".join([str(j) for j in i]))



