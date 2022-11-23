#Para rodar mp3 tambem: https://www.youtube.com/watch?v=xcdTIDHm4KM

import os
import shutil
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import scipy as sp
from scipy import stats
import librosa
import sys
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import scale
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

#Função de extração dos features
def extrai_feat(filepath):
    def feature_stats(values):
            return([stats.kurtosis(values, axis=1),
                    np.max(values, axis=1),
                    np.mean(values, axis=1),
                    np.median(values, axis=1),
                    np.min(values, axis=1),
                    stats.skew(values, axis=1),
                    np.std(values, axis=1),])

    #x, sr = librosa.load(filepath, sr=None, mono=True, duration=5)
    x, sr = librosa.load(filepath, sr=None, mono=True)

    stft = np.abs(librosa.stft(x, n_fft=2048, hop_length=512))

    mel = librosa.feature.melspectrogram(sr=sr, S=stft**2)

    f = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)

    a = feature_stats(f)

    statistics = ['kurtosis','max','mean','median','min','skew','std']

    #Incluir demais dados posteriormente como zcr e cia

    feat_vec = []

    for j in range(0,7):

        #Percorre cada frame
        for k in range(0,20):

            feat_vec.append(a[j][k])

    return feat_vec

#Funções de manipulação

#Não altera
def manip_x(X):
    return(X)

#Padroniza
def manip_s(X):
    return(scale(X))

#3 IC's
def manip_ica(X):
    return(FastICA(n_components = int(sys.argv[4])).fit_transform(X))

#3 PC's
def manip_pca(X):
    return(PCA(n_components = int(sys.argv[4])).fit_transform(X))

#PC's até 90% da variãncia explicada
def manip_pca90(X):
    n, p = X.shape
    pca = PCA(n_components=min(p,n)).fit(X)
    pcafit = PCA(n_components=min(p,n)).fit_transform(X)

    ind_90 = 0
    while np.sum(pca.explained_variance_ratio_[0:ind_90]) <= 0.90:
        ind_90 = ind_90 + 1

    return(pcafit[:,0:ind_90])

#Features importantes da classificação por árvores
def manip_tree(X):
    return(X[:,[20,22,23,24,26,40,43,46,47,60,63,66,67,80]])

#Funções de clusterização

#Hierarquico
def clust_h(Z):
    g = AgglomerativeClustering(n_clusters=n_clust).fit(Z)
    return(list(g.labels_))

#K medoides
def clust_kmd(Z):
    g = KMedoids(n_clust).fit(Z)
    return(list(g.labels_))

#K Medias
def clust_km(Z):
    g = KMeans(n_clust).fit(Z)
    return(list(g.labels_))

#GMM
def clust_gmm(Z):
    g = GaussianMixture(n_components=n_clust, covariance_type='full', random_state=0).fit(Z)
    labels = list(g.predict(Z))

#Spectral
def clust_spec(Z):
    g = SpectralClustering(n_clusters=n_clust).fit(Z)
    return(list(g.labels_))


#1º argumento do sistema é o caminho da pasta

#2º argumento do sistema é o algoritmo de clusterização
# -h : hierarquico (ward)
# -kmd : KMedoids
# -km : Kmedias
# -gmm : GMM
# -spec :spectral

# -rev : chama a função que reverte a clusterização (desmonta as pastas)

#3º argumento é a forma dos features
# -x : features normais
# -s : Dados padronizados
# -ica : ICs
# -pca : PC's
# -pca90 : PC's até 90% da variância explicada

#4º argumento é o numero de componentes do pca e ica

if(sys.argv[1]=='-help'):
    print('1º argumento do sistema é o caminho da pasta no sistema \n')
    print('2º argumento do sistema é o algoritmo de clusterização')
    print(' -h : hierarquico (ward) \n -kmd : KMedoids \n -km : Kmedias \n -gmm : GMM \n -spec :spectral \n')
    print('3º argumento é a forma dos features')
    print(" -x : features normais \n -s : Dados padronizados \n -ica : IC's \n -pca : PC's \n -pca90 : PC's até 90% da variância explicada")
    exit()


#Carrega o arquivo
dir_path = sys.argv[1]

#Reverte clusterização
#a implementar

#Pergunta o número de clusters
n_clust = int(input("Número de Clusters: "))

#Extrai os nomes dos arquivos e naturalmente seus respectivos indices
files = os.listdir(dir_path)

#Numero de observacoes eh o numero de arquivos
n = len(files)

#Acrescente o caminho aos nomes dos arquivos

#Versão Windows
for i in range(0,n):
    files[i] = dir_path+'\\' + files[i]

#Versão Linux
#for i in range(0,n):
#    files[i] = dir_path+ '/' + files[i]

##Gera banco de dados

#Extrai do primeiro arquivo para obter numero de features
feat0 = extrai_feat(files[0])

#Numero de features
p = len(feat0)

X = np.zeros((n,p))

X[0,:] = feat0

print("Extraindo Features \n")
for i in tqdm(range(1,n)):

    X[i,:] = extrai_feat(files[i])

#Faz possíveis manipulações

manip = {'-x':manip_x,'-s':manip_s,'-ica':manip_ica,'-pca':manip_pca,'-pca90':manip_pca90,'-tree':manip_tree}

print(sys.argv[3])

Z = manip[sys.argv[3]](X)

##Clusteriza e gera labels da clusterizacao

print("Clusterizando...")

clust = {'-h':clust_h,'-kmd':clust_kmd,'-km':clust_km,'-gmm':clust_gmm,'-spec':clust_spec}

labels = clust[sys.argv[2]](Z)

##Salva conforme a clusterização

#Cria pastas, uma para cada cluster

clust_path = []

#Versão windows
for j in range(0,n_clust):
    clust_path.append(dir_path+'\Cluster '+str(j+1))
    os.mkdir(dir_path+'\Cluster '+str(j+1))

#Versão Linux
#for j in range(0,n_clust):
#    clust_path.append(dir_path+'/Cluster '+str(j+1))
#    os.mkdir(dir_path+'/Cluster '+str(j+1))

for i in range(0,n):
    shutil.move(files[i], clust_path[labels[i]])

print('Clusterização Completa')
