# -*- coding: utf-8 -*-
"""
"""
import sys

## -------------   captura de arquivos - Identifique as pasta em que estao os arquivos.
# sys.path.insert(0,'C:\\Users\\Stefano\\Documents\\LAMFO\\Options_Port_SPG\\Ambiente Opçoes')
# lista="C:\\Users\\Stefano\\Documents\\Options Full\\Options Full"
# numOpc=12 #     numero deativos por grupo
# Refp=1 #        numero de meses a fente, maturidade de referencia.
# numPeriodos=0 # numero de periodos, solo ou at'e expira'cao.
# callput=0 #     nao implementado.
# Capital0=50000# Captal teorico.

###---------------------   Definindo lista de variaveis de decisão ------- ###
Variaveis={} # DICIONARIO QUE COMPILA DIFERENTES SELEÇOES DE VARIAVEIS.
## --- contem dados sobre o futuro para que ele aprenda rapido. Mas irrealista.
Variaveis.update({"Ambi_teste":['STRIKE_PRC', 'DeltaP', 'EtasC', 'CLOSE_F1', 'CLOSE_F2', 'ImplDeltaC2', 'CLOSE_F3',
                             'ImplDeltaP1', 'ImplDeltaP2', 'ImplDeltaP3', 'ImplEtasC2', 'ImplGammaC3', 'CLOSE_1F_ATIVO', 'ImplThetaC1', 'ImplThetaP3',
                             'ImplValorC1', 'ImplValorC2', 'ImplValorC3', 'BLAKvolat1', 'BLAKvolat2', 'BLAKvolat3', 'Retorno_2F_ATIVO', 'RhoC',
                             'Rlog1_ATIVO', 'Rlog2_ATIVO', 'Rlog3_ATIVO', 'ThetaC','ValorC', 'VegaC', 'VegaP', 'VolatH_1F_ATIVO',
                             'VolatH1_ATIVO', 'VolatH2_ATIVO','VolatH3_ATIVO', 'BLAKvolat_F3']})


## --- contem dados sobre o tiradas informaçossobre o futuro.
Variaveis.update({"Ambi_A0":['HIGH', 'CLOSE_x', 'LOW', 'OPEN', 'VOLUME', 'HIGH_1', 'CLOSE_1', 'LOW_1', 'OPEN_1', 'VOLUME_1', 'HIGH_2', 'CLOSE_2', 'LOW_2', 'OPEN_2', 'VOLUME_2',
                             'HIGH_3', 'CLOSE_3', 'LOW_3', 'OPEN_3', 'VOLUME_3', 'HIGH_4', 'CLOSE_4', 'LOW_4', 'OPEN_4', 'VOLUME_4', 'HIGH_5', 'CLOSE_5', 'LOW_5',
                             'OPEN_5', 'VOLUME_5', 'CLOSE_ATIVO', 'Retorno_ATIVO', 'Rlog_ATIVO', 'VolatH_ATIVO', 'CLOSE1_ATIVO', 'Retorno1_ATIVO', 'Rlog1_ATIVO',
                             'VolatH1_ATIVO', 'CLOSE2_ATIVO', 'Retorno2_ATIVO', 'Rlog2_ATIVO', 'VolatH2_ATIVO', 'CLOSE3_ATIVO', 'Retorno3_ATIVO', 'Rlog3_ATIVO',
                             'VolatH3_ATIVO', 'CLOSE4_ATIVO', 'Retorno4_ATIVO', 'Rlog4_ATIVO', 'VolatH4_ATIVO', 'FreeRiskCLOSE', 'FreeRiskCLOSE_1', 'FreeRiskCLOSE_2', 'FreeRiskCLOSE_3',
                             'FreeRiskCLOSE_4', 'Retonro0', 'Retonro1', 'Retonro2', 'Retonro3', 'Retonro4', 'PriceToStrike0', 'PriceToStrike1', 'PriceToStrike2',
                             'PriceToStrike3', 'dayTOexp', 'DeltaC', 'DeltaP', 'EtasC', 'EtasP', 'GammaC', 'GammaP', 'RhoC', 'RhoP', 'ThetaC', 'ThetaP', 'ValorC', 'ValorP', 'VegaC', 'VegaP',
                             'BLAKvolat1', 'BLAKvolat2', 'BLAKvolat3', 'BLAKvolat4', 'BLAKvolat5', 'ImplDeltaC1', 'ImplDeltaP1', 'ImplEtasC1',
                             'ImplEtasP1', 'ImplGammaC1', 'ImplGammaP1', 'ImplRhoC1', 'ImplRhoP', 'ImplThetaC1', 'ImplThetaP1', 'ImplValorC1',
                             'ImplValorP1', 'ImplVegaC1', 'ImplVegaP1', 'ImplDeltaC2', 'ImplDeltaP2', 'ImplEtasC2', 'ImplEtasP2', 'ImplGammaC2',
                             'ImplGammaP2', 'ImplRhoC2', 'ImplThetaC2', 'ImplThetaP2', 'ImplValorC2', 'ImplValorP2', 'ImplVegaC2', 'ImplVegaP2',
                             'ImplDeltaC3', 'ImplDeltaP3', 'ImplEtasC3', 'ImplEtasP3', 'ImplGammaC3', 'ImplGammaP3', 'ImplRhoC3', 'ImplThetaC3',
                             'ImplThetaP3', 'ImplValorC3', 'ImplValorP3', 'ImplVegaC3', 'ImplVegaP3', 'ImplDeltaC4', 'ImplDeltaP4', 'ImplEtasC4',
                             'ImplEtasP4', 'ImplGammaC4', 'ImplGammaP4', 'ImplRhoC4', 'ImplThetaC4', 'ImplThetaP4', 'ImplValorC4', 'ImplValorP4',
                             'ImplVegaC4', 'ImplVegaP4', 'ImplDeltaC5', 'ImplDeltaP5', 'ImplEtasC5', 'ImplEtasP5', 'ImplGammaC5', 'ImplGammaP5',
                             'ImplRhoC5', 'ImplThetaC5', 'ImplThetaP5', 'ImplValorC5', 'ImplValorP5', 'ImplVegaC5', 'ImplVegaP5']})

## --- contem poiucos dados.
Variaveis.update({"Ambi_A1":['dayTOexp','STRIKE_PRC',
                            'PriceToStrike0',
                            'Retonro0', 'Retonro1', 'Retonro2',
                            'VolatH_ATIVO', 'VolatH1_ATIVO', 'VolatH2_ATIVO', 'VolatH3_ATIVO','VolatH4_ATIVO',
                            'CLOSE_x','CLOSE_1','CLOSE_2','CLOSE_3','CLOSE_4',
                            'Rlog_ATIVO','Rlog1_ATIVO', 'Rlog2_ATIVO', 'Rlog3_ATIVO',
                            'BLAKvolat1', 'BLAKvolat2', 'BLAKvolat3','BLAKvolat4','BLAKvolat5']})
