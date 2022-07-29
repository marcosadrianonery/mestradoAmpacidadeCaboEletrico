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

listSize = ["small", "full"]
listType = ["train", "test"]

sizeData = listSize[0]
typeData = listType[0]
chave_Features = 'ft_nwp_index'
chave_Labels = 'lb_index'

# fallback to all available features
input_shape = ['TMP', 'WSPD', 'WDIR']

# fallback to all available labels
output_shape = ['RWM_690', '690_RWM',
                '690_CIT', 'CIT_690', 'CIT_ROV', 'ROV_CIT']

# past 48 hours of observational data
# shape: (16787, 48, 32)
root = np.load('Arquivos/' + str(sizeData) +
               '/Prognonetz_INL_' + str(typeData) + '.npz')

# load features
ft_index = list((root)[chave_Features])
ft_shape = [ft_index.index(item) for item in input_shape]
# features = (root)['ft_nwp'][..., ft_shape]
features = (root)['ft_nwp']

# load labels
lb_index = list((root)[chave_Labels])
lb_shape = [lb_index.index(item) for item in output_shape]
labels = (root)['lb']
labels_index = (root)['lb_index']


print(labels_index)
