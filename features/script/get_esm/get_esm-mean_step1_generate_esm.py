import pandas as pd
import torch
from Bio import SeqIO
import os

os.makedirs("./out_esm/",exist_ok=True)
black_list = set([i[:-3] for i in os.listdir("./out_esm/")])
#(1) input sequece for esm
seqs = {i.id:str(i.seq) for i in SeqIO.parse("../../../data/ind_2021/ppis_ind.fasta","fasta") if len(str(i.seq))}

#(2) esm not support sequence which length more than  1024.
# split long sequence to 1000bp sub sequecne with 500bp overlab
with open("./tmp_esm.fasta","w") as f:
    for k,v in seqs.items():
        if len(v) > 1000:
            for i in range(len(v) // 500):
                kid = f"{k}_{i}"
                if kid not in black_list:
                    f.write(f">{k}_{i}\n{v[i*500:(i*500+1000)]}\n")
        else:
            if k not in black_list:
                f.write(f">{k}\n{v}\n")

#(3) use uniprot50 model 
cmd = f"python  ./extract.py esm1b_t33_650M_UR50S \
        ./tmp_esm.fasta \
        ./out_esm/ \
        --include mean per_tok \
        --repr_layers 33 "

#cmd = f"python  ./extract.py esm_msa1b_t12_100M_UR50S \
#        ./seqs_esm.fasta \
#        ./out_esm/ \
#        --include mean per_tok \
#        --repr_layers 11 "
#
print(cmd)
os.system(cmd)
#os.system("python ./esm2dict.py")

