lbs <- c('car','MASS','tidyverse','ggplot2','ISLR','graphics','effects','leaps','psych',
         'lattice','lmtest','robustbase')

install.lib <- lbs[!lbs %in% installed.packages()]
for(libs in install.lib) install.packages(libs, dependences = TRUE)
sapply(lbs, require, character = TRUE)


#################################
# otázka è.1
#################################

my_data <- read.table("auto-mpg-01rean.txt",header = TRUE)

my_data = as.data.frame(my_data)
my_data

summary(my_data)

# kontrola NaNs

nan_rows = my_data[is.na(my_data$mpg)|is.na(my_data$horsepower),]
nan_rows

data_mpg = my_data[!is.na(my_data$mpg),] # dataset bez NaNu v consumption
data_mpghp = my_data[!is.na(my_data$mpg) & !is.na(my_data$horsepower),] # dataset bez NaNu  v consumption i v horsepower


#################################
# otázka è.2
#################################

summary(data_mpg)
View(data_mpg)



my_data$origin = as.factor(my_data$origin)
my_data$cylinders = as.factor(my_data$cylinders)
data_mpg$origin = as.factor(data_mpg$origin)
data_mpg$cylinders = as.factor(data_mpg$cylinders)
data_mpghp$origin = as.factor(data_mpghp$origin)
data_mpghp$cylinders = as.factor(data_mpghp$cylinders)


View(my_data)

summary(my_data)

#################################
# otázka è.3
#################################

my_data$consumption = (100*3.785411)/(my_data$mpg * 1.609344)
my_data$displacement = my_data$displacement*0.016387064
my_data$weight = my_data$weight*0.45359237
summary(my_data)

my_data <- within(my_data,rm('mpg')) # odstraníme 

data_mpg = my_data[!is.na(my_data$consumption),] # dataset bez NaNu v consumption
data_mpghp = my_data[!is.na(my_data$consumption) & !is.na(my_data$horsepower),] # dataset bez NaNu  v consumption i v horsepower

summary(data_mpghp)

#################################
# otázka è.4
#################################
?hist

# histogramy

cols = c('consumption', 'displacement', 'acceleration', 'horsepower', 'weight', 'model_year')
print(length(cols))

par(mfrow = c(2, 3))

# historamy s odhadem hustot
for (col in cols)
{
  hist(data_mpghp[,col],xlab = col,freq=FALSE,main = '')
  lines(density(data_mpghp[,col]), col="red", lwd=1)
}

par(mfrow = c(2, 3))
cols = c('displacement', 'acceleration', 'horsepower', 'weight', 'model_year')

for (col in cols)
{
  scatter.smooth(data_mpghp[,col],data_mpghp[,'consumption'],
                 xlab = col,ylab = 'consumption',main = '',lpars = list(col='blue',lwd=2))
  abline(lm(data_mpghp[,'consumption'] ~ data_mpghp[,col],data = data_mpghp),col = 'red', lwd = 2)  
  #lines(lowess(data_mpghp[,col],data_mpghp[,'consumption']),col='blue',lwd=2)
  }



#################################
# otázka è.5
#################################

install.packages("stringr")
library('stringr')

data_mpghp$producer = word(data_mpghp$car_name,1)
data_mpg$producer = word(data_mpg$car_name,1)
my_data$producer = word(my_data$car_name,1)

data_mpghp$model_year = factor(data_mpghp$model_year)
data_mpg$model_year = factor(data_mpg$model_year)
my_data$model_year = factor(my_data$model_year)




library(gplots)
library(ggplot2)

cols = c('model_year','producer','cylinders','origin')

par(mfrow = c(2,3))


p1 <- ggplot(data_mpghp,aes(x=model_year,y=consumption, fill = model_year)) + 
    geom_boxplot(notch=FALSE,outlier.color = 'black') + scale_fill_discrete(name='year')
plot(p1)


p2 <- ggplot(data_mpghp,aes(x=producer,y=consumption, fill = producer)) + 
  geom_boxplot(notch=FALSE,outlier.color = 'black') + scale_fill_discrete(name='producer') +
  theme(axis.text.x=element_text(angle=75))
plot(p2)


p3 <- ggplot(data_mpghp,aes(x=cylinders,y=consumption,fill = cylinders)) + 
  geom_boxplot(notch=FALSE,outlier.color = 'black') + scale_fill_discrete(name='cylinders')
plot(p3)

p4 <- ggplot(data_mpghp,aes(x=origin,y=consumption,fill = origin)) + 
  geom_boxplot(notch=FALSE,outlier.color = 'black') + scale_x_discrete(labels=c("1" = "USA", "2" = "Evropa","3" = "Japonsko"))+
  scale_fill_discrete(name='origin',labels=c("1" = "USA", "2" = "Evropa","3" = "Japonsko"))
  
plot(p4)


install.packages('cowplot')
library(cowplot)


plot_grid(p1, p2,p3,p4, ncol = 2,nrow = 2)



#################################
# otázka è.6
#################################

# jsou zde obì varianty

p <- ggplot(data_mpghp,aes(y=consumption))+
  geom_point(aes(x = origin,colour=cylinders))+
  scale_x_discrete(labels=c("1" = "USA", "2" = "Evropa","3" = "Japonsko"))
print(p)



p <- ggplot(data_mpghp,aes(y=consumption))+
  geom_point(aes(x = cylinders,colour=origin))+
  scale_colour_discrete(labels=c("1" = "USA", "2" = "Evropa","3" = "Japonsko"))
print(p)

#################################
# otázka è.7
#################################


data_chrysler = data_mpghp[data_mpghp[,'producer']=='chrysler',]
summary(data_chrysler)


p <- ggplot(data_chrysler,aes(x=weight,y=consumption))+
  geom_point(aes(size = displacement, colour = cylinders))

print(p)

#################################
# otázka è.8
#################################


# mám zodpovìzeno v pøedchozí otázce, staèí jen vykreslit a vysvìtlit, k èemu je to užiteèné, vhodné



#################################
# otázka è.9
#################################



lm_intercept = lm(consumption ~ weight, data_mpghp)

summary(lm_intercept)

lm_nointercept = lm (consumption ~ weight - 1, data_mpghp)

summary(lm_nointercept)
# Budeme zkoumat základní model s interceptem a mùžeme i model bez interceptu, který dává i logicky vìtší smysl než model s interceptem (kvùli bodu (0,0))
# První náznak zøejmé závislosti evidentnì vyplývá z grafu, kde vidíme, že èím vyšší váha, tím zpravidla stoupá spotøeba a naopak. Pokud se podíváme na výstupy ze summary funkce jednotlivých modelù, pak zjistíme, že
# že p hodnota pro pøíslušnou t statistiku je výraznì nízká, což napovídá zamítnutí hypotézy, že daná promìnná nemá na predikci vliv (koeficient roven 0)
# Mùžeme tedy øíci, že spotøeba opravdu závisí na váze.

# Model bez interceptu má proti modelu s interceptem vìtší hodnotu F statistiky a R^2 statistiky, což mùže poukazovat na lepší schopnost modelu vysvìtlit daná data.

# vykreslení obou regresních pøímek
p <- ggplot(data_mpghp, aes(x=weight, y=consumption)) + geom_point() +
 geom_line(aes(x = weight, y = lm_intercept$fitted.values ,colour = 'blue'),size=0.8)+
  geom_line(aes(x = weight, y = lm_nointercept$fitted.values ,colour = 'red'),size=0.8)+
scale_color_discrete(name = 'models', labels = c('with intercept','no intercept'))

plot(p)

# zmìnu spotøeby z modelu spoèítáme pohodlnì jako \delta(weight)*slope
# pokud využijeme model s interceptem, pøi zmìnì o 1000kg se spotøeba zmìní o pøibližnì 8.9kg
# pøi použití modelu bez interceptu se pøi stejné zmìnì hmotnosti spotøeba zmìní pøibližnì o 8.4kg


#################################
# otázka è.10
#################################



# první si data vykreslíme bez a s log transformací

par(mfrow = c(1,2))
plot(data_mpghp$weight,data_mpghp$consumption,xlab = 'weight',ylab='consumption')
plot(data_mpghp$weight,log(data_mpghp$consumption),xlab = 'weight',ylab='log(consumption)')

# z obrázku není patrné, že bychom si tvarem proti pøedchozímu pøípadu nìjak pomohli, podíváme se na model

lm_logcons_intercept = lm(log(consumption)~weight,data_mpghp)
lm_logcons_nointercept = lm(log(consumption)~weight-1,data_mpghp)

# log_cons_nointercept neuvažovat!!!!

summary(lm_intercept)
summary(lm_logcons_intercept)
summary(lm_logcons_nointercept)
summary(lm_nointercept)

# z výstupu summary funkcí plyne, že model s log transformací fituje transformovaná data pøibližnì stejnì kvalitnì jako pøedchozí modely bez transformace
# navíc se jedná o predikci hodnot transformovaných, které nás ve finále nezajímají a je potøeba je transformovat zpìt
# takové transformace závislé promìnné mùžou být èasto užiteèné, pokud nesplòujeme OLS podmínky na normalitu reziduí (to se potvrdí v pozdìjších otázkách)
# pokud se ptáme na zmìnu spotøeby, tak ta se nebude mìnit konstantnì jako tomu bylo v pøedchozím pøípadì, bude záviset na poèáteèní váze, od které zmìnu poèítáme
# mùžeme uvést napø. zmìnu spotøeby pøi zmìnì váhy z 1000kg na 2000 kg pro pøípad bez interceptu
# tj. 23.24799
c <- exp(predict(lm_logcons_nointercept,data.frame(weight = 2000))) - exp(predict(lm_logcons_nointercept,data.frame(weight = 1000)))
c
exp(predict(lm_logcons_nointercept,data.frame(weight = 2000)))
# z vykreslených grafù se zdá, že bychom ještì log(consumption) mohli transformovat nìjakou mocninnou funkcí a tím to lépe "vyrovnat" do pøímky, zkusíme druhou mocninu
# tj. 
exp(predict(lm_logcons_nointercept,data.frame(weight = 2000)))

lm_logcons_nointerceptsquared = lm((log(consumption))^2~weight-1,data_mpghp)
lm_logcons_interceptsquared = lm((log(consumption))^2~weight,data_mpghp)

summary(lm_logcons_interceptsquared)
summary(lm_logcons_nointerceptsquared)
summary(lm_logcons_nointercept)
summary(lm_logcons_intercept)
# vidíme, že došlo k zvýšení R^2 statistiky i F-statistiky proti ostatním modelùm

# vykreslíme scatter plot a data proložíme lin.modely bez interceptu, první klasický, pak s log transformací a nakonec log transformaci umocnìnou na druhou

p <- ggplot(data_mpghp, aes(x=weight, y=consumption)) + geom_point() +
  geom_line(aes(x = weight, y = lm_nointercept$fitted.values ,colour = 'blue'),size=0.8)+
  geom_line(aes(x = weight, y = exp(sqrt(lm_logcons_nointerceptsquared$fitted.values)) ,colour = 'red'))+
  geom_line(aes(x = weight, y = exp(lm_logcons_nointercept$fitted.values) ,colour = 'green'))+
  geom_line(aes(x = weight, y = exp(sqrt(lm_logcons_interceptsquared$fitted.values)) ,colour = 'yellow'))
  



plot(p)

# vidíme, že klasický log model bez interceptu špatnì fituje pùvodní data. Proti tomu mode s log transformací (i umocnìný na druhou) vypadá na první pohled velmi dobøe
# tomu odpovídají i F statistika s R^2 statistikou

#################################
# otázka è.11
#################################


# jako první provedeme po èástech konstantní transformaci hmotnosti, intervaly zvolíme po 100kg
# a v každém intervalu bude hodnota zvolena jako støední hodnota pùvodních hmotností spadajících do tohoto intervalu

#data_mpghp$constant_weight = 
  
?cut

breaks = seq(from = min(data_mpghp$weight), to = max(data_mpghp$weight)+100, by = 100)


data_mpghp$constant_weight2 = cut(data_mpghp$weight, breaks = breaks, include.lowest = TRUE, right = FALSE)
data_mpghp$constant_weight = replicate(length(data_mpghp$constant_weight2),0)


for (i in levels(factor(data_mpghp$constant_weight2)))
{
  temp = mean(data_mpghp[data_mpghp$constant_weight2 == i,'weight'])
  data_mpghp[data_mpghp$constant_weight2 == i,'constant_weight'] = temp
}
# ve sloupci constant_weight máme po èástech konstantní transformaci

any(is.na(data_mpghp))
?lm
# vytvoøíme pøíslušné modely
lm_const = lm(consumption ~ constant_weight,data_mpghp)
lm_quadratic = lm(consumption ~ I(weight^2),data_mpghp)
lm_cubic = lm(consumption ~ I(weight^3),data_mpghp)

summary(lm_const)
summary(lm_quadratic)
summary(lm_cubic)


# nyní si to vykreslíme i s proložením køivkami z pøíslušných lm
p_const <- ggplot(data_mpghp, aes(x=constant_weight, y=consumption)) + geom_point()+
  geom_line(aes(x = constant_weight, y = lm_const$fitted.values ,colour = 'red'),size=0.8)
plot(p_const)

p_quadratic <- ggplot(data_mpghp, aes(x=weight^2, y=consumption)) + geom_point()+
  geom_line(aes(x = weight^2, y = lm_quadratic$fitted.values ,colour = 'red'),size=0.8)
plot(p_quadratic)

p_cubic <- ggplot(data_mpghp, aes(x=weight^3, y=consumption)) + geom_point()+
  geom_line(aes(x = weight^3, y = lm_cubic$fitted.values ,colour = 'red'),size=0.8)
  
plot(p_cubic)

# u kvadratické a kubické transformace si mùžeme všimnout, že by nejspíše nebyla splnìna homoskedasticita reziduí,
# což mùžeme i vidìt na residuals vs fitted plotu

par(mfrow = c(2,2))
plot(lm_const)
bptest(lm_cubic) # test na homoskedasticitu, pokud p hodnota malá, pak se homoskedasticita zamítá



#################################
# otázka è.12
#################################

# potøebujeme ovìøit 1) nezávilost a homoskedasticitu reziduí se støední hodnotou 0, což mùžeme ovìøit napø. z residuals vs fitted plotu, homoskedasticitu mùžeme ještì ovìøit napø. pomocí breusch-paganova testu (bptest)
# 2) normalitu reziduí pomocí QQ plotu a napø. sapirova testu

# budeme testovat jednoduchý model bez interceptu a model s logaritmicky škálovanou spotøebou také bez interceptu

# Jako první si vykreslíme ploty pro model bez interceptu
plot(lm_nointercept)
# QQ plot nevypadá pøíliš dobøe, zkusíme otestovat pomocínshapirova testu normalitu
shapiro.test(residuals(lm_nointercept))

# p hodnota je kriticky nízká, normalitu tudíž musíme zamítnout a tím pádem tento model nesplòuje OLS požadavky

# nyní zkusíme model s log škálovanou spotøebou a taky s log škálovanou spotøebou umocnìný na druhou

plot(lm_logcons_intercept)

shapiro.test(residuals(lm_logcons_intercept))
bptest(lm_logcons_intercept)

# z výsledku testù je patrné, že normalitu ani homoskedasticitu nezamítáme, pøestože residuals vs fitted plot (scale-location) je podezøelý
# vidíme, že log transformace zde pomohla k naplnìní pøedpokladù pro použití OLS

# zkusíme ještì log^2

plot(lm_logcons_interceptsquared)
shapiro.test(residuals(lm_logcons_interceptsquared))
bptest(lm_logcons_interceptsquared)
 #zde také vidíme, že normalita není zamítnuta (\alpha = 0.05) ze shapirova testu a dokonce je p hodnota v tomto pøípadì vyšší než u modelu s klasickou log transformací
 # navíc bptest nezamítá homoskedasticitu, takže máme taky validní OLS model (možná ještì lépe validní než jen s log transf)

#################################
# otázka è.13
#################################

# vybral jsem model lm_logcons_interceptsquared

# první vykreslíme pro transformovanou spotøebu

pred_transf = predict(lm_logcons_interceptsquared,data_mpghp,interval = 'prediction')
pred_transfback = exp(sqrt(pred_transf))

conf_transf = predict(lm_logcons_interceptsquared,data_mpghp,interval = 'confidence')
conf_transfback = exp(sqrt(conf_transf))

data <- cbind(data_mpghp, pred_transf)

conf_transf = data.frame(conf_transf)

conf_transf$weight = data_mpghp$weight
conf_transf

# vykreslíme
p1 <- ggplot(data, aes(x=weight, y=(log(consumption))^2)) + geom_point()+
  geom_line(aes(y = lwr ,colour = 'red'),size=0.8) +
  geom_line(aes(y = upr ,colour = 'red'),size=0.8) +
  geom_line(data=conf_transf,aes(x=weight,y = lwr ,colour = 'blue'),size=0.8) +
  geom_line(data=conf_transf,aes(x=weight,y = upr ,colour = 'blue'),size=0.8)+
  scale_color_discrete(name = 'bands',labels = c('confident','prediction'))
plot(p1)


# skuteèná spotøeba
p <- ggplot(data, aes(x=weight, y=consumption)) + geom_point()+
  geom_line(aes(x=weight,y=exp(sqrt(lm_logcons_interceptsquared$fitted.values)))) +
  geom_line(aes(y = exp(sqrt(lwr)) ,colour = 'red'),size=0.8) +
  geom_line(aes(y = exp(sqrt(upr)) ,colour = 'red'),size=0.8) +
  geom_line(data=conf_transf,aes(x=weight,y = exp(sqrt(lwr)) ,colour = 'blue'),size=0.8) +
  geom_line(data=conf_transf,aes(x=weight,y = exp(sqrt(upr)) ,colour = 'blue'),size=0.8)+
  scale_color_discrete(name = 'bands',labels = c('confident','prediction'))
  

plot(p)
 # zde si nejsem jistý, jestli takto zpìtná transformace krajních bodù prediction a confident bandù platí


# nyní vykreslíme výstup z allEffects
library(effects)
plot(allEffects(lm_logcons_interceptsquared))
plot(predictorEffects(lm_logcons_interceptsquared))#, ~ type + education))

# vidíme, že confident band souhlasí s naším

#################################
# otázka è.14
#################################

# vytvoøíme lineární regresní model pøi uvážení promìnných weight a origin
# využijeme stejnou vybranou transformaci spotøeby jako v pøedchozím pøípadì

data1 = data_mpghp

lm_weight_origin<-lm(log(consumption)^2~weight+origin,data_mpghp)
summary(lm_weight_origin)


# nevím, jestli chápu správnì zadání, ale asi staèí jen nastavit tøi jednoduché regresní modely
# pro každou zemi zvláš a vykreslit regresní pøímky

data1 = data_mpghp[data_mpghp$origin == 1,]
data2 = data_mpghp[data_mpghp$origin == 2,]
data3 = data_mpghp[data_mpghp$origin == 3,]

lm1 = lm(consumption~weight , data1)
lm2 = lm(consumption~weight , data2)
lm3 = lm(consumption~weight , data3)



p <- ggplot(data_mpghp, aes(x=weight, y=consumption, color = origin))+geom_point()+
  geom_line(data = data1, aes(x=weight,y=lm1$fitted.values), size = 0.8)+
  geom_line(data = data2, aes(x=weight,y=lm2$fitted.values), size = 0.8)+
  geom_line(data = data3, aes(x=weight,y=lm3$fitted.values), size = 0.8)
  
plot(p)

#################################
# otázka è.15
#################################









