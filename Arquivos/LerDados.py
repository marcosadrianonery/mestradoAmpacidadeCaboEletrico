import numpy as np
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import os
import functions
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import classification_report  # metricas de validação


listSize = ["small", "full"]
listType = ["train", "test"]
keys = ["ft_obs", "ft_obs_index", "ft_nwp",
        "ft_nwp_index", "nwp_grid", "labels", "lb_index", "ts"]

'''
database = functions.prognonetxRead(sizeData=listSize[1], typeData=listType[0])
print(database[2].shape)
database = functions.prognonetxRead(sizeData=listSize[1], typeData=listType[1])
print(database[2].shape)
'''
##########################################################################

database_train = functions.prognonetxReadKeys(
    sizeData=listSize[1], typeData=listType[0])

##########################################################################

database_test = functions.prognonetxReadKeys(
    sizeData=listSize[1], typeData=listType[1])
# print(database_x_test["ft_nwp"].shape)

##########################################################################

pos_x = 3
pos_y = 4

dados_X_PmaisP = functions.selectGrade(
    database_ft_nwp=database_train["ft_nwp"], pos_x=pos_x, pos_y=pos_y, organizaHora=False)

dados_X_PmaisP_test = functions.selectGrade(
    database_ft_nwp=database_test["ft_nwp"], pos_x=pos_x, pos_y=pos_y, organizaHora=False)


laco_Y_PmaisP = []
laco_Y_PmaisP_test = []
dados_Y_PmaisP = []
dados_Y_PmaisP_test = []

nomeEtiquetas = {"RWM_690": 0, "690_RWM": 1, "690_CIT": 2,
                 "CIT_690": 3, "CIT_ROV": 4, "ROV_CIT": 5}

for hora in database_train["lb"]:
    for etiqueta in hora:
        # print(etiqueta)
        laco_Y_PmaisP.append(etiqueta[nomeEtiquetas["690_CIT"]])
dados_Y_PmaisP = functions.normalize_2d(np.array(laco_Y_PmaisP))

for hora in database_test["lb"]:
    for etiqueta in hora:
        # print(etiqueta)
        laco_Y_PmaisP_test.append(etiqueta[nomeEtiquetas["690_CIT"]])
dados_Y_PmaisP_test = functions.normalize_2d(np.array(laco_Y_PmaisP_test))


print(dados_X_PmaisP.shape)
print(dados_Y_PmaisP.shape)


regressor_random_forest = RandomForestRegressor(n_estimators=500)
regressor_random_forest.fit(dados_X_PmaisP, dados_Y_PmaisP)
print(regressor_random_forest.score(dados_X_PmaisP, dados_Y_PmaisP))

print(regressor_random_forest.score(dados_X_PmaisP_test, dados_Y_PmaisP_test))
#   resultado_y = regressor_random_forest.predict(dados_X_PmaisP_test)
#   print(classification_report(dados_Y_PmaisP_test, resultado_y))
