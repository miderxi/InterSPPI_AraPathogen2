"""
"""
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from zzd.utils.assess import multi_scores
import sys
import pickle

# parameters
print(sys.argv)
info_list = sys.argv[1:] if len(sys.argv)>1 else None
output_dir = f"../outputs/preds/RF_"+"_".join(info_list)
os.makedirs(output_dir,exist_ok=True)

# read
train_files = [f'../data/p1n10_10folds/train_{i}.txt' for i in range(10)] 
test_files = [f'../data/p1n10_10folds/test_{i}.txt' for i in range(10)] 

class Features:
    def __init__(self,info:list):
        feature_ac = np.load("../features/AC/ac.pkl",allow_pickle=True)                #shape (210,)
        feature_ct = np.load("../features/CT/ct.pkl",allow_pickle=True)                #shape (343,)
        feature_dpc = np.load("../features/DPC/dpc.pkl",allow_pickle=True)             #shape (400,)
        feature_cksaap = np.load("../features/CKSAAP/cksaap.pkl",allow_pickle=True)    #shape (1200,)
        feature_esm = np.load("../features/esm/esm.pkl",allow_pickle=True)             #shape(1280,)
        feature_esm_msa = np.load("../features/esm-msa/esm_msa.pkl",allow_pickle=True) #shape (768,)
        #feature_esm_msa.update(np.load("../features/esm-msa/esm-msa_ind.pkl",allow_pickle=True))

        feature_prottrans = np.load("../features/prottrans/prottrans_embs.pkl",allow_pickle=True)  #shape (1024,)
        feature_doc2vec = np.load("../features/doc2vec/doc2vec_100.pkl",allow_pickle=True) #shape (400,)

        class Feature_AraNetProperty:
            def __init__(self):
                self.flag_foldn=None
                self.data = [np.load(f"../features/AraNet_property/AraNet_property_train_{i}.pkl",allow_pickle=True) for i in range(10)] #shape (12,)
            def __getitem__(self,index):
                return self.data[self.flag_foldn][index] if index in self.data[self.flag_foldn].keys() else np.zeros(12,np.float32) 
        
        class Feature_AraNetNode2vec:
            def __init__(self):
                self.data = np.load("../features/AraNet_node2vec/AraNet_node2vec.pkl",allow_pickle=True)  #shape (256,)
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(256,np.float32) 
        
        class Feature_geo:
            def __init__(self):
                self.data = np.load("../features/gene_expression/geo.pkl",allow_pickle=True) #shape (117, )
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(111,np.float32) 

        class Feature_sublocation:
            def __init__(self):
                self.data = np.load("../features/sublocation/sublocaiton.pkl",allow_pickle=True) #shape (117, )
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(34,np.float32)


        class Feature_dmi:
            def __init__(self):
                self.data = np.load("../features/dmi_node2vec/prot_dmi_node2vec.pkl",allow_pickle=True)
            def __getitem__(self,index):
                return self.data[index] if index in self.data.keys() else np.zeros(128,np.float32) 
        
        class Feature_pssm_emb():
            def __init__(self):
                self.flag_foldn=None
                self.data =[np.load(f"../features/pssm_emb/pssm_emb_train_{i}.pkl",allow_pickle=True) for i in range(10)]
            def __getitem__(self,index):
                return self.data[self.flag_foldn][index] 

        class Feature_onehot_emb():
            def __init__(self):
                self.flag_foldn=None
                self.data =[np.load(f"../features/onehot_emb/onehot_emb_train_{i}.pkl",allow_pickle=True) for i in range(10)]
            def __getitem__(self,index):
                return self.data[self.flag_foldn][index] 

        self.info = info
        self.features={
            'ac':feature_ac,
            'ct':feature_ct,
            'dpc':feature_dpc,
            'cksaap':feature_cksaap,
            'esm':feature_esm,
            'esm_msa':feature_esm_msa,
            'prottrans':feature_prottrans,
            'doc2vec':feature_doc2vec,
            'AraNetProperty':Feature_AraNetProperty(),
            'AraNetNode2vec':Feature_AraNetNode2vec(),
            'geo':Feature_geo(),
            'dmi':Feature_dmi(),
            'pssm':Feature_pssm_emb(),
            'onehot':Feature_onehot_emb(),
            'sublocation':Feature_sublocation()
            }
    
    def get(self,index,foldn=None):
        self.features['AraNetProperty'].flag_foldn=foldn
        self.features['pssm'].flag_foldn=foldn
        self.features['onehot'].flag_foldn=foldn
        tmp = np.hstack([self.features[i][index] for i in self.info])
        return tmp

features = Features(info=info_list)

ten_scores = []
for foldn in range(10):
    train_table = np.genfromtxt(train_files[foldn],str)
    test_table  = np.genfromtxt(test_files[foldn],str)

    np.random.seed(0)
    np.random.shuffle(train_table)

    X_train, y_train = train_table[:,:2], train_table[:,2].astype(np.float32)
    X_test,  y_test  = test_table[:,:2],  test_table[:,2].astype(np.float32)

    x_train = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in X_train ]) 
    x_test  = np.array([np.hstack([features.get(j,foldn) for j in i]) for i in  X_test])

    model = RandomForestClassifier(n_estimators=1000,n_jobs=-1, random_state=0)
    model.fit(x_train,y_train)

    y_test_pred = model.predict_proba(x_test)[:,1]

    tmp_score = multi_scores(y_test,y_test_pred,show=True)
    ten_scores.append(tmp_score)

    with open(f"{output_dir}/pred_{foldn}.txt","w") as f:
        for line in np.hstack([X_test,y_test.reshape(-1,1),y_test_pred.reshape(-1,1)]):
            line = "\t".join(line) + "\n"
            f.write(line)

    with open(f"{output_dir}/score_{foldn}.txt","w") as f:
            f.write("TP	TN	FP	FN	PPV	TPR	TNR	Acc	mcc	f1	AUROC	AUPRC	AP\n")
            f.write("\t".join([str(i) for i in tmp_score]))

    #save model
    model_file_name = "../outputs/model_state/RF_" + "_".join(info_list)+f"_foldn_{foldn}.pkl"
    with open(model_file_name,"wb") as f:
        pickle.dump(model,f)

ten_scores = np.array(ten_scores)
with open(f"{output_dir}/average_score.txt",'w') as f:
    line1 = "TP TN  FP  FN  PPV TPR TNR Acc mcc f1  AUROC   AUPRC   AP\n"
    line2 = '\t'.join([f'{a:.3f}??{b:.3f}' for (a,b) in zip(ten_scores.mean(0),ten_scores.std(0))])
    print(line1)
    print(line2)
    f.write(line1)
    f.write(line2)



