require("bizdays")
require("timeDate")

#Etapa 0 - Descricao de variaveis.
S=80 # spot price
K=85 # strik ptrice
R=0.05 # free risk
B=0 # dividends, or cost of carry. 00.00 *** O atualmente programado nao compila corretamente os dividendos.
D=0.80 # volatility
T1="2017-08-21"
T2="2017-09-18"
Market="BR" # ou USA
sty="C-EU" #"C-AM"
dcalend<-365

OpitionsPrice<-function(S,K,R,B,D,T1,T2,Market,sty){
  # Compilacaos de caractristicas:
  if(Market=="BR"){
    cal<-Calendar(start.date = as.Date("2015-01-01"), end.date = as.Date(Sys.time())+365*10,holidays=holidaysANBIMA, weekdays=c("saturday", "sunday"))
  }else{
    cal<-Calendar(start.date = as.Date("2015-01-01"), end.date = as.Date(Sys.time())+365*10,holidays=holidayNYSE(2015:2019), weekdays=c("saturday", "sunday"))
  }
  days<-(bizdays(as.Date(T1),as.Date(T2),cal))/dcalend #252
  ## "C-EU"; "C-AM";"C-EU";"P-AM"
  style<-strsplit(sty,"-")[[1]][2]
  type<-strsplit(sty,"-")[[1]][1]
  if(style=="EU"||type=="C"){
    ## 1o - Black modeling:EU
    D1=(log(S/K)+((R-B+((D^2)/2))*days))/(D*(days^0.5))
    D2=((log(S/K)+((R-B+((D^2)/2))*days))/(D*(days^0.5)))-D*(days^0.5)
    ## black greeks/ 
    if(type=="C"){
      Valor=S*pnorm(D1)-K*exp(-R*days)*pnorm(D2)
      Delta=pnorm(D1)
      Gamma=dnorm(D1)/(S*D*sqrt(days))
      Vega=S*dnorm(D1)*(sqrt(days))
      Theta=(-((S*dnorm(D1)*D)/(2*sqrt(days)))-(R*K*exp(-R*days))*pnorm(D2))/dcalend #252
      Rho=K*days*exp(-R*days)*pnorm(D2)
      Etas=Delta*S/Valor
    }else{
      Valor=K*exp(-R*days)*pnorm(-D2)-S*pnorm(-D1)
      Delta=-pnorm(-D1)
      Gamma=dnorm(D1)/(S*D*sqrt(days))
      Vega=S*dnorm(D1)*(sqrt(days))
      Theta=(-((S*dnorm(D1)*D)/(2*sqrt(days)))+(R*K*exp(-R*days))*pnorm(-D2))/dcalend #252
      Rho=-K*days*exp(-R*days)*pnorm(-D2)
      Etas=Delta*S/Valor
    }
    
  }
  ##-----  resultado experado:
  return((as.data.frame(cbind(Valor,Delta,Gamma,Vega,Theta,Rho,Etas))))
}
## ---- Resultado experado
OpitionsPrice(S,K,R,B,D,T1,T2,Market,sty)

#Etapa 0 - Descricao de variaveis.
IplVolOpitionsPrice<-function(S,K,R,B,D,Price,T1,T2,Market,sty){
  # Compilacaos de caractristicas:
  if(Market=="BR"){
    cal<-Calendar(start.date = as.Date("2015-01-01"), end.date = as.Date(Sys.time())+365*10,holidays=holidaysANBIMA, weekdays=c("saturday", "sunday"))
  }else{
    cal<-Calendar(start.date = as.Date("2015-01-01"), end.date = as.Date(Sys.time())+365*10,holidays=holidayNYSE(2015:2019), weekdays=c("saturday", "sunday"))
  }
  days<-(bizdays(as.Date(T1),as.Date(T2),cal))/dcalend #252
  ## "C-EU"; "C-AM";"C-EU";"P-AM"
  style<-strsplit(sty,"-")[[1]][2]
  type<-strsplit(sty,"-")[[1]][1]
  Valor=0
  i=0
  while(!round((Price-Valor),2)==0&&i<500){
    if(style=="EU"||type=="C"){
      ## 1o - Black modeling:EU
      D1=(log(S/K)+((R-B+((D^2)/2))*days))/(D*(days^0.5))
      D2=((log(S/K)+((R-B+((D^2)/2))*days))/(D*(days^0.5)))-D*(days^0.5)
      ## black greeks/ 
      if(type=="C"){
        Valor=S*pnorm(D1)-K*exp(-R*days)*pnorm(D2)
        Delta=pnorm(D1)
        Gamma=dnorm(D1)/(S*D*sqrt(days))
        Vega=S*dnorm(D1)*(sqrt(days))
        Theta=(-((S*dnorm(D1)*D)/(2*sqrt(days)))-(R*K*exp(-R*days))*pnorm(D2))/dcalend #252
        Rho=K*days*exp(-R*days)*pnorm(D2)
        Etas=Delta*S/Valor
      }else{
        Valor=K*exp(-R*days)*pnorm(-D2)-S*pnorm(-D1)
        Delta=-pnorm(-D1)
        Gamma=dnorm(D1)/(S*D*sqrt(days))
        Vega=S*dnorm(D1)*(sqrt(days))
        Theta=(-((S*dnorm(D1)*D)/(2*sqrt(days)))+(R*K*exp(-R*days))*pnorm(-D2))/dcalend #252
        Rho=-K*days*exp(-R*days)*pnorm(-D2)
        Etas=Delta*S/Valor
      }
    }
    i<-1+i
    D=ifelse(Price>Valor,D+0.001,D-0.001)
  }
  ##-----  resultado experado:
  return((as.data.frame(cbind(Valor,Delta,Gamma,Vega,Theta,Rho,Etas,D))))
}
## ---- Resultado experado
IplVolOpitionsPrice(S,K,R,B,D,5,T1,T2,Market,sty)
#### -------- 






