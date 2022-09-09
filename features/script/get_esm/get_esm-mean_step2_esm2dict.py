import os
import pickle
import torch
import numpy as np


#id_dict to represent long and sort sequences.
id_dict = dict()    
for f_name in os.listdir("./out_esm"):
    if f_name[-5:] in set(["_0.pt","_1.pt","_2.pt","_3.pt"]):
        k = f_name[:-5]
        if k not in id_dict.keys():
            id_dict[k]=[f_name,]
        else:
            id_dict[k].append(f_name)
    else:
        id_dict[f_name.split(".")[0]] = [f_name,]

for k,v in id_dict.items():
    id_dict[k] = sorted(id_dict[k])

#get the esm_mean feature 
esm_mean = dict()
for idx,(k, f_names) in enumerate(id_dict.items()):
    black_ids = set()
    if len(f_names) == 1:
        #pass
        temp = torch.load(f"./out_esm/{f_names[0]}")
        esm_mean[k] = temp['mean_representations'][33].numpy()
        #esm_representation = temp['representations'][33]
        #torch.save(esm_representation,f"./esm_feature/{k}.pt")
        print(f"{idx} {len(id_dict)} {k}")
    else:
        temp = [ torch.load(f"./out_esm/{f_name}")  for f_name in f_names ]
        esm_mean[k] = torch.stack([i['mean_representations'][33] for i in temp]).mean(0).numpy()
        #if len(f_names) == 2:
        #    esm_representation = torch.vstack([temp[0]['representations'][33] ,
        #                                       temp[1]['representations'][33][1000:] ])
        #    torch.save(esm_representation,f"./esm_feature/{k}.pt")
        #if len(f_names) == 3:
        #    esm_representation = torch.vstack([temp[0]['representations'][33] ,
        #                                       temp[1]['representations'][33][500:],
        #                                       temp[2]['representations'][33][500:]])
        #    torch.save(esm_representation,f"./esm_feature/{k}.pt")
        #if len(f_names) ==  4:
        #    esm_representation = torch.vstack([temp[0]['representations'][33] ,
        #                                       temp[1]['representations'][33][500:],
        #                                       temp[2]['representations'][33][500:],
        #                                       temp[3]['representations'][33][500:]])
        #    torch.save(esm_representation,f"./esm_feature/{k}.pt")


        print(f"{idx} {len(id_dict)} {k}")



with open("../../../features/esm/esm_mean_ind.pkl","wb") as f:
    pickle.dump(esm_mean, f)


        

