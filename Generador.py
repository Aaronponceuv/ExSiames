"""
-.  Cantidad de Chatarra por tipo

405 = 258
406 = 62
407 = 91
409 = 266
411 = 108
416 = 45
432 = 210

-.  Cantidad de pares iguales por tipo de chatarra

formula = (N * N-1)/2

 405 = 33153 pares
 406 = 1891
 407 = 4095
 409 = 35245
 411 = 5778
 416 = 990
 432 = 22155
+-------------+
Total = 103.307

-. Cantidad de pares diferentes por tipo de chatarra

 405 = 201756
 406 = 60636
 407 = 86359
 409 = 205884
 411 = 100656
 416 = 44775
 432 = 174300
+---------------+
Total = 874.366
"""
import numpy as np 
import pandas as pd 
import os
import itertools

def save_test(cantidad_test,pares,chatarra_uno,chatarra_cualquiera,indice_test):
    test = []
    for i in range(int(cantidad_test)):
        indice_r = np.random.random_integers(len(pares)-1)
        test.append(pares[indice_r])
        print(chatarra_uno+"/"+pares[indice_r].split("-")[0])
        print(chatarra_cualquiera+"/"+pares[indice_r].split("-")[1])
        data_test.loc[indice_test] = [chatarra_uno+"/"+pares[indice_r].split("-")[0],chatarra_cualquiera+"/"+pares[indice_r].split("-")[1],0]
        pares.remove(pares[indice_r])
        indice_test +=1
    return pares,indice_test,test

def save_val(cantidad_val,pares,chatarra_uno,chatarra_cualquiera,indice_val):
    val = []
    for i in range(int(cantidad_val)):
        indice_r = np.random.random_integers(len(pares)-1)
        val.append(pares[indice_r])
        data_val.loc[indice_val] = [chatarra_uno+"/"+pares[indice_r].split("-")[0],chatarra_cualquiera+"/"+pares[indice_r].split("-")[1],0]
        pares.remove(pares[indice_r])
        indice_val +=1 
    return pares,indice_val,val

def parear_similares(ancla_positivo,tipos_chatarra,data,data_test,data_val,indice,indice_test,indice_val):
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



def parear_diferentes(data, data_test, data_val,chatarra_uno, chatarra_cualquiera, indice_test,indice_val,indice):
    pares = []
    rutas_chatarra_uno = os.listdir(os.path.join(path,chatarra_uno))
    rutas_chatarra_cualquiera = os.listdir(os.path.join(path,chatarra_cualquiera))
    dif = itertools.product(rutas_chatarra_uno, rutas_chatarra_cualquiera)
    for e in dif:
            pares.append('-'.join(e))

    cantidad_train_antes = len(pares)
    cantidad_test = len(pares) * 0.2

    pares,indice_test, test = save_test(cantidad_test,pares,chatarra_uno,chatarra_cualquiera,indice_test)

    cantidad_train_antes = len(pares)
    cantidad_val = len(pares) * 0.2
    pares, indice_val, val = save_val(cantidad_val,pares,chatarra_uno,chatarra_cualquiera,indice_val)

    print("Test: ",chatarra_uno,": ",len(test))
    print("Val: ",chatarra_uno,": ",len(val))
    print("Train Ahora: ",chatarra_uno,": ",len(pares))

    for idx in pares:
        data.loc[indice] = [chatarra_uno+"/"+idx.split("-")[0],chatarra_cualquiera+"/"+idx.split("-")[1],0] 
        indice+=1
    return data,data_test,data_val,indice,indice_test,indice_val  




def generar_pares_diferentes(tipos_chatarra,data,data_test,data_val):
    indice = 0
    indice_test =0 
    indice_val = 0
    for chatarra_uno in tipos_chatarra:
        for chatarra_cualquiera in tipos_chatarra:
            if(chatarra_uno != chatarra_cualquiera):
                data,data_test,data_val,indice,indice_test,indice_val = parear_diferentes(data, data_test, data_val, chatarra_uno, chatarra_cualquiera, indice_test, indice_val,indice)
    data.to_csv('train_dif.csv', sep=',' , index=False)
    data_test.to_csv('test_dif.csv', sep=',' , index=False)
    data_val.to_csv('val_dif.csv', sep=',' , index=False)


def generar_pares_iguales(tipos_chatarra,data,data_test,data_val):
    indice = 0
    indice_test =0 
    indice_val = 0
    for i in tipos_chatarra:
        ancla_positivo = []
        data,data_test,data_val,indice,indice_test,indice_val = parear_similares(ancla_positivo,i,data,data_test,data_val,indice,indice_test,indice_val)
    data.to_csv('train_an_pos.csv', sep=',' , index=False)
    data_test.to_csv('test_an_pos.csv', sep=',' , index=False)
    data_val.to_csv('val_an_pos.csv', sep=',' , index=False)

if __name__ == "__main__":
    path = "Chatarra Conjunto 2"
    path = os.path.abspath(path)
    tipos_chatarra = os.listdir(path)

    data = pd.DataFrame(columns=('ancla','negativo','label'))
    data_test = pd.DataFrame(columns=('ancla','negativo','label'))
    data_val = pd.DataFrame(columns=('ancla','negativo','label'))

    generar_pares_diferentes(tipos_chatarra,data,data_test,data_val)

    data_ancla_positivo = pd.DataFrame(columns=('ancla','positivo'))
    data_test_ancla_positivo = pd.DataFrame(columns=('ancla','positivo'))
    data_val_ancla_positivo = pd.DataFrame(columns=('ancla','positivo'))

    generar_pares_iguales(tipos_chatarra,data_ancla_positivo,data_test_ancla_positivo,data_val_ancla_positivo)
    