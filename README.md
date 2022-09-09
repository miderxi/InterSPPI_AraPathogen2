## 1. download and untar the program
```
git clone git@github.com:miderxi/InterSPPI_AraPathogen2.git
cd InterSPPI_AraPathogen2
tar -xvf RF_esm_net.tar.xz
```

## 2 make prediction
### 2.1 you need prepare three file
```
ppis_list.txt   #ppis for prediction. one interaction per line, separated by tabs.
arabidopsis_thalia.fasta #arabidopsis thaliana protein sequences.
pathogen_effector.fasta #pathogen effector protein sequences.
```
### 2.2 make prediction
```
bash  RF_esm_net_predict.sh [ppi_list.txt] [arabidopsis_thalia.fasta] [pathogen_effector.fasta] [outdir]
#The prediction result will save in the file outdir/ppi_list_predicted.txt
```
