## OptionsGame ##

library(dplyr)

dados = data.table::fread('/home/cayan/Downloads/RL_optionsBasePrototipoCall.csv')

dados = dados %>% 
        group_by(Codiigo) %>% 
        mutate(Diferenca = abs(STRIKE_PRC - Value)) %>%
        top_n(-9 , Diferenca) 
