import numpy as np
import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity
import os

"""
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Chave	        |   Descrição
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ft_obs	        |   Tensor de forma (16787, 48, 32) contendo as últimas 48 horas de dados observacionais
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ft_obs_index	|   Lista de forma (32,) contendo índice para o tensor ft_obs. As entradas são formatadas assim:
                    |   'stationID_variableID' exemplo: 'RWM_pressure'
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ft_nwp	        |   Tensor de forma (16787, 48, 16, 14, 3) contendo as próximas 48 horas de dados NWP
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ft_nwp_index	|   Lista de forma (3,) contendo índice para o tensor ft_nwp.
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    nwp_grid	    |   Tensor de forma (16, 14, 2) contendo coordenadas de grade para dados NWP
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Libra	        |   Tensor de forma (16787, 48, 6) contendo as próximas 48 horas de ampacidades (etiquetas)
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    lb_index	    |   Lista de forma (32,) contendo índice para o tensor lb. As entradas são formatadas assim:
                    |   'stationID_stationID' exemplo: 'RWM_690'
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    ts	            |   Tensor de forma (16787, 48, 2) contendo timestamps para todas as amostras
---------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def prognonetxRead(sizeData, typeData):

    root = np.load('Arquivos/' + str(sizeData) +
                   '/Prognonetz_INL_' + str(typeData) + '.npz')

    ft_obs = (root)['ft_obs']
    ft_obs_index = (root)['ft_obs_index']
    ft_nwp = (root)['ft_nwp']
    ft_nwp_index = (root)['ft_nwp_index']
    nwp_grid = (root)['nwp_grid']
    labels = (root)['lb']
    lb_index = (root)['lb_index']
    ts = (root)['ts']

    return [ft_obs, ft_obs_index, ft_nwp, ft_nwp_index, nwp_grid, labels, lb_index, ts]


def prognonetxReadKeys(sizeData, typeData):

    root = np.load('Arquivos/' + str(sizeData) +
                   '/Prognonetz_INL_' + str(typeData) + '.npz')

    return root


def selectGrade(database_ft_nwp, pos_x, pos_y, organizaHora=False):
    dados_ft_nwp_ponto_mais_proximo = []
    dados_hora = []
    dados_ft_nwp_ponto_mais_proximo_hora = []
    for amostra in database_ft_nwp:
        dados_hora = []

        for hora in amostra:

            for count_x, x in enumerate(hora):
                if count_x == pos_x:
                    for count_y, y in enumerate(hora[count_x]):
                        if count_y == pos_y:
                            dados_ft_nwp_ponto_mais_proximo.append(
                                np.nan_to_num(hora[count_x][count_y]))
                            dados_hora.append(
                                np.nan_to_num(hora[count_x][count_y]))
        dados_ft_nwp_ponto_mais_proximo_hora.append(dados_hora)

    if(organizaHora):
        return normalize_2d(np.array(dados_ft_nwp_ponto_mais_proximo_hora))
    else:
        return normalize_2d(np.array(dados_ft_nwp_ponto_mais_proximo))


def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix
