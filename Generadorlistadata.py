import numpy as np 
import pandas as pd 
import os
import itertools

"""
-.  Cantidad de Chatarra por tipo

405 = 258
406 = 62
407 = 91
409 = 266
411 = 108
416 = 45
432 = 210

-.  Cantidad de pares por tipo de chatarra

formula = (N * N-1)/2

405 = 33153 pares
406 = 1891
407 = 4095
409 = 35245
411 = 5778
416 = 990
432 = 22155

"""
def parear(ancla_positivo,tipos_chatarra,data,data_test,data_val,indice,indice_test,indice_val):
    chatarra = os.listdir(os.path.join(path,tipos_chatarra))
    for r in [2]:
    # para permutar cambiar combinations por permutations
        res = itertools.permutations(chatarra, r)
        for e in res:
            ancla_positivo.append('-'.join(e))
    
    cantidad_train_antes = len(ancla_positivo)
    cantidad_test = len(ancla_positivo) * 0.2

    test = []
    for i in range(int(cantidad_test)):
        indice_r = np.random.random_integers(len(ancla_positivo)-1)
        test.append(ancla_positivo[indice_r])
        data_test.loc[indice_test] = [tipos_chatarra+"/"+ancla_positivo[indice_r].split("-")[0],tipos_chatarra+"/"+ancla_positivo[indice_r].split("-")[1]]
        ancla_positivo.remove(ancla_positivo[indice_r])
        indice_test +=1


    cantidad_train_antes = len(ancla_positivo)
    cantidad_val = len(ancla_positivo) * 0.2

    val = []
    for i in range(int(cantidad_val)):
        indice_r = np.random.random_integers(len(ancla_positivo)-1)
        val.append(ancla_positivo[indice_r])
        data_val.loc[indice_val] = [tipos_chatarra+"/"+ancla_positivo[indice_r].split("-")[0],tipos_chatarra+"/"+ancla_positivo[indice_r].split("-")[1]]
        ancla_positivo.remove(ancla_positivo[indice_r])
        indice_val +=1    

    print("Test: ",tipos_chatarra,": ",len(test))
    print("Train Antes: ",tipos_chatarra,": ",cantidad_train_antes)
    print("Train Ahora: ",tipos_chatarra,": ",len(ancla_positivo))


    for idx in ancla_positivo:
        data.loc[indice] = [tipos_chatarra+"/"+idx.split("-")[0],tipos_chatarra+"/"+idx.split("-")[1]] 
        indice+=1
    return data,data_test,data_val,indice,indice_test,indice_val


path = "Chatarra Conjunto 2"
data = pd.DataFrame(columns=('ancla','positivo'))
data_test = pd.DataFrame(columns=('ancla','positivo'))
data_val = pd.DataFrame(columns=('ancla','positivo'))
path = os.path.abspath(path)
tipos_chatarra = os.listdir(path)
print(tipos_chatarra)
chatarra = os.listdir(os.path.join(path,tipos_chatarra[0]))

indice = 0
indice_test =0 
indice_val = 0
for i in tipos_chatarra:
    ancla_positivo = []
    data,data_test,data_val,indice,indice_test,indice_val = parear(ancla_positivo,i,data,data_test,data_val,indice,indice_test,indice_val)
data.to_csv('train.csv', sep=',' , index=False)
data_test.to_csv('test.csv', sep=',' , index=False)
data_val.to_csv('val.csv', sep=',' , index=False)
print('Train:',i,":",data.shape)




