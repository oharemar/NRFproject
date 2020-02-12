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

# ALTERNATIVNÍ ODPOVÌÏ DOLE je lepší

# vytvoøíme lineární regresní model pøi uvážení promìnných weight a origin a spotøebou transformovanou jako v pøedchozím pøípadì
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


# alternativní odpovìï

# udìláme aditivní lin.model

lm_weight_origin<-lm(consumption~weight+origin,data_mpghp)
summary(lm_weight_origin)

# data splitneme podle origin
data_split_origin  <- split(data_mpghp, data_mpghp$origin)

data_split_origin[['1']]$Fit <- predict(lm_weight_origin, data_split_origin[['1']])
data_split_origin[['2']]$Fit  <- predict(lm_weight_origin, data_split_origin[['2']])
data_split_origin[['3']]$Fit  <- predict(lm_weight_origin, data_split_origin[['3']])


p <- ggplot(data_mpghp, aes(x=weight, y=consumption, color = origin))+geom_point()+
  geom_line(data = data_split_origin[['1']], aes(x=weight,y=Fit), size = 0.8)+
  geom_line(data = data_split_origin[['2']], aes(x=weight,y=Fit), size = 0.8)+
  geom_line(data = data_split_origin[['3']], aes(x=weight,y=Fit), size = 0.8)
  
plot(p)

# nakonec bych pøidal i regresní pøímky pro dva rùzné modely na jednotlivých skupinách


#################################
# otázka è.15
#################################
# jako statistický test použijeme rozšíøení t-testu, tzv. one-way anova test.
# je alternativou t-testu, pokud máme více než 2 faktory
# tento test pøedpokládá normalitu dat v každém faktoru a stejné rozptyly napøíc faktory (mìlo by se ovìøit)
# pokud bychom nesplnili tyto požadavky, lze použít jako alternativu Kruskalùv test


# vykreslíme obrázky zvláš pro origin a pro cylinders

install.packages('ggpubr')
library("ggpubr")
p <- ggline(data_mpghp, x = 'origin', y = 'consumption', 
            add = c("mean_se", "jitter"), 
            order = c("1", "2", "3"))
            
plot(p)

p <- ggline(data_mpghp, x = 'cylinders', y = 'consumption', 
            add = c("mean_se", "jitter"))#,color='origin')

plot(p)

# provedeme one-way ANOVA test zvláš pro origin a cylinders
summary(aov(consumption~origin,data_mpghp))
# díky velmi nízké p-hodnotì zamítáme nulovou hypotézu, která øíká, že støední hodnoty se napøíè jednotlivými zemìmi rovnají

kruskal.test(consumption~origin,data_mpghp)
# kruskalùv test to potvrzuje


summary(aov(consumption~cylinders,data_mpghp))
# i pro cylinders zamítáme nulovou hypotézu
kruskal.test(consumption~cylinders,data_mpghp)
# kruskalùv test to potvrzuje

# Pokud by nastala situace, že by se støední hodnoty consumption rovnali pro nìjakou faktorovou promìnnou pro každý faktor, pak si myslím, že daná faktorová promìnná nebude mít
# pøíliš velký vliv na vysvìtlovanou promìnnou(zanedbatelný èi žádný vliv), kvùli rovnostem støedních hodnot se jednotlivé hodnoty v rùzných faktorech budou 'dost podobat'

# mùžeme to demonstrovat na pøíkladu
?rep
x <- seq(-20, 20, by = .1)
x_1 <- rep(1,length(x))
y_1 <- rnorm(x,mean=0,sd=1)
plot(x_1,y_1)


x_2 <- rep(2,length(x))
y_2 <- rnorm(x,mean=0,sd=2)
plot(x_2,y_2)

x_3 <- rep(3,length(x))
y_3 <- rnorm(x,mean=0,sd=0.5)
plot(x_3,y_3)


df1=data.frame(explanatory = y_1,independent = x_1)
df2=data.frame(explanatory = y_2,independent = x_2)
df3=data.frame(explanatory = y_3,independent = x_3)

df = rbind(df1,df2,df3)
df$independent = factor(df$independent)

lm = lm(explanatory~independent,df)
summary(lm)

# vidíme, že v tomto pøípadì 'explanatory' pøíliš nezávisí na 'independent'
# pokud zopakujeme experiment s jinými støedními hodnotami

x <- seq(-20, 20, by = .1)
x_1 <- rep(1,length(x))
y_1 <- rnorm(x,mean=0,sd=1)
plot(x_1,y_1)

x_2 <- rep(2,length(x))
y_2 <- rnorm(x,mean=1,sd=1)
plot(x_2,y_2)

x_3 <- rep(3,length(x))
y_3 <- rnorm(x,mean=0.5,sd=0.5)
plot(x_3,y_3)


df1=data.frame(explanatory = y_1,independent = x_1)
df2=data.frame(explanatory = y_2,independent = x_2)
df3=data.frame(explanatory = y_3,independent = x_3)

df = rbind(df1,df2,df3)
df$independent = factor(df$independent)

lm = lm(explanatory~independent,df)
summary(lm)

# zde už to vypadá jinak a závislost 'explanatory' na 'independent' je zøejmá


#################################
# otázka è.16
#################################

# z modelu mùžeme na zaèátku rovnou neuvažovat všechny pomocné promìnné, co jsme si v prùbìhu zpracování vytvoøili
# dále nebudeme uvažovat car_name, jelikož je pro každou instanci unikátní a také
# promìnnou producer kvùli vysokému poètu faktorù a také zøejmé kolinearitì s origin
# (mohli bychom slouèit producery z jedné zemì do jednoho, viz obrázek)
p <- ggplot(data_mpghp, aes(x=producer, y=consumption, color = origin))+geom_point()
plot(p)
# promìnnou cylinders budeme uvažovat jako numerickou promìnnou, jednak kvùli snížení poètu faktorù v modelu
# (origin dohromady s model_year už dají 14 faktorù) a také kvùli pøesnìjší detekci multikolinearity

# zde možná budeme muset oddìlat více promìnných, nevím pøesnì, co budeme mít v datech
lm_all  <- lm(consumption ~ (.-car_name-producer-horsepower_discrete)*(.-car_name-producer-horsepower_discrete), data = data_mpghp)
summary(lm_all)

# otázku kolinearity v modelu vzhledem k následujícím otázkám teï øešit nebudeme
# použijeme AIC a BIC hodnoty k urèení nejvhodnìjšího modelu z lm_all, který bude výchozí


lm_temp_final1 <- stepAIC(lm_start,direction='backward') # model AIC

lm_temp_final2 <- stepAIC(lm_start,direction='backward', k=log(n)) # model BIC


# porovnáme pomocí anova()
anova(lm_temp_final2,lm_temp_final1)
# z porovnání vyšlo, že varianta získána AIC metodou je výhodnìjší, budeme tedy pokraèovat
# s lm_temp_final1

summary(lm_temp_final1)

# validujme nyní OLS pøedpoklady

plot(lm_final,which=2)
# QQ plot nevypadá moc dobøe, zkontrolujeme shapirovým testem
shapiro.test(residuals(lm_final))
# podle shapiro testu bychom mìli zamítnout nulovou hypotézu, tedy normalitu
# v dalších otázkách budeme zkoušet transformace consumption kvùli zlepšení normality reziduí
# nesplnìní OLS požadavkù mùže být taky zpùsobeno pøítomností multikolinearity v modelu, kterou budeme teprve zkoumat



#################################
# otázka è.17
#################################

form = 'log(consumption) ~ cylinders + displacement + horsepower + weight + 
acceleration + model_year + origin + cylinders:horsepower + 
cylinders:model_year + displacement:horsepower + displacement:acceleration + 
displacement:model_year + displacement:origin + horsepower:weight + 
horsepower:model_year + weight:acceleration + weight:model_year + 
weight:origin + acceleration:model_year + acceleration:origin'

# jako první udìláme logaritmickou transformaci
lm_final_log = lm(form,data_mpghp)
summary(lm_final_log)

plot(lm_final_log,which=2)
plot(lm_final_log,which=1)
shapiro.test(residuals(lm_final_log))
# vidíme, že tato tranformace pomohla a shapirùv test nezamítá normalitu
# i QQ plot vypadá mnohem lépe a residuals vs fitted taky vypadá dobøe


# vyzkoušíme box-coxovu transformaci
# zde je log-vìrohodnostní profil
library(MASS)
bc <- boxcox(lm_temp_final1)

lambda <- bc$x[which.max(bc$y)]
print(lambda)

form_lambda = '(consumption^lambda - 1)/lambda ~ cylinders + displacement + horsepower + weight + 
acceleration + model_year + origin + cylinders:horsepower + 
cylinders:model_year + displacement:horsepower + displacement:acceleration + 
displacement:model_year + displacement:origin + horsepower:weight + 
horsepower:model_year + weight:acceleration + weight:model_year + 
weight:origin + acceleration:model_year + acceleration:origin'

lm_final_bc = lm(form_lambda,data_mpghp)
summary(lm_final_bc)

plot(lm_final_bc,which=2)
plot(lm_final_bc,which=1)
bptest(lm_final_bc)
# QQ plot i residuals vs fitted plot urèující nezávislost reziduí vypadají dobøe, ovìøíme normalitu ještì shapiro testem
shapiro.test(residuals(lm_final_bc))
# test prošel bez problému

# zdá se, že model s box-cox transformací je co se týèe R^2 statistiky a splnìní OLS požadavkù
# na tom velmi podobnì jako model s log transformací
# nakonec bych se asi rozhodl pro box-cox model, který vykazuje o chloupek lepší R^2 statistiku a normalitu reziduí


#################################
# otázka è.18
#################################

# Použili jsme log transformaci a v takovém pøípadì zmìna spotøeby v závislosti na váze
# bude záviset na poèáteèní váze, od které zmìnu poèítáme (zmìna je nelineární), nelze tedy urèit obecnì, jaká bude procentuální zmìna

# model z otázky 16 obsahuje kromì samotné weight i interakce s weight, takže pak není zøejmé, jaký pøesnì má vliv urèitá zmìna váhy na spotøebu
# každopádnì pokud bychom zanedbali interakce s weight , pak má weight opaèný vliv než v modelu z otázky 9
# tj, pøi zvyšování váhy spotøeba klesá - bez interakcí a ostatních nezávislých promìnných by ale takový model pak nedával s takovou závislostí pøíliš smysl



#################################
# otázka è.19
#################################

# zde se zamìøíme na problém kolinearity mezi vysvìtlujícími promìnnými

# vyzkoušíme kompletní aditivní model bez pomocných promìnných, car_name a producer
# a necháme si vypsat VIF hodnoty kvùli detekci kolinearity
lmm <- lm(log(consumption) ~ (.) - horsepower_discrete-car_name-producer,data_mpghp)
summary(lmm)

vif(lmm)
kappa(lmm) 

# vidíme vyšší hodnoty VIF u promìnných cylinders,displacement,horsepower,weight
# které by mohly být mezi sebou provázané
# vzájemná kolinearita jde hezky vidìt z ggpairs

ggpairs(data_mpghp[c("cylinders","displacement","horsepower","weight")])
# mezi cylinders,displacement,horsepower,weight je zøejmá kolinearita

# z grafù jde vidìt, že pokud jedna z promìnných "roste", pak "roste" i druhá, což v podstatì
# sedí i s fyzikálním cítìním úlohy, kde se zdá, že se zvyšující se váhou vozidla musí rùst i spotøeba
# totéž, pokud zvyšujeme poèet válcù a zdvihový objem, také se zvyšuje spotøeba
# výkon by do tohoto výbìru nutnì patøit nemusel, ale z grafù je zøejmé, že to v našem pøípadì platí také

# z tìchto promìnných bude tedy staèit použít jenom jednu z nich

# pokud uvážíme pouze weight, pak dostaneme


lmm2 <- lm(consumption ~ (.) - horsepower_discrete-car_name-producer-horsepower-displacement-cylinders,data_mpghp)
summary(lmm2)

vif(lmm2)
kappa(lmm2)
# Tedy se podaøilo kolinearitu v modelu snížit, nyní vypadá již dostateèné nízká

# Nyní budeme uvažovat náš finální model z úlohy 16 a pokusíme se odstranit multikolinearitu


# na zaèátku máme

vif(lm_final_log)
kappa(lm_final_log)

# to znamená, že v modelu máme vysoký stupeò kolinearity

#  iterativnì budeme odebírat promìnné s nejvyšší VIF hodnotou
# nìkdy se stalo, že se VIF hodnoty pro více promìnných lišily napø. na 3 desetinném míste ->
# zkusil jsem více možností zahazování a výsledné modely porovnal pomocí anova()
# threshold jsem nastavil na 3, ale v pozdìjších fázích vyhazování se obèas stalo,
# že jedna VIF hodnota byla nápadnì vysoká proti ostatním a blízko 3 (2,8...), tak to jsem také zahodil

# postup byl dlouhý, uvedu jen nejlepší model

form = "log(consumption) ~ cylinders+ origin + acceleration+
horsepower:model_year + weight:acceleration"

lm_test = lm(form,data_mpghp)
summary(lm_test)
car::vif(lm_test)
kappa(lm_test)

plot(lm_test,which=2) # QQ plot vypadá dobøe
plot(lm_test,which=3) # scale location vypadá hùøe
shapiro.test(residuals(lm_test))
# prochází normalita
bptest(lm_test) # bptest zamítá homoskedasticitu, ale hodnì tìsnì


lm_finale = lm_test

#################################
# otázka è.20
#################################

library(car)
# tyto grafy znázoròují vliv jedné nezávislé promìnné na response variable pøi uvážení vlivu ostatních promìnných v modelu
# partial regression plot se hodí na detekci influenèních bodù
# navíc sklon regresní pøímky v tìchto plotech je stejný jako sklon dané promìnné v pùvodním vícerozmìrném lineárním modelu

# naproti tomu partial residual plot je vhodný k detekci nelinearity mezi danou promìnnou
# mùže nám odhalit vhodné transformace nezávislých promìnných k zajištìní linearity

# jako první vykreslíme partial regression ploty
avPlots(lm_finale) # partial regression plot

# vidíme, že nìkolik bodù bylo znázornìno jako podezøelé influenèní body, budeme se jimi zabývat v dalších otázkách

# residual plot pomocí podobného balíèku vykreslit nemùžeme (nechce vykreslit pro model s interakcemi)
# vykreslíme si interakce a faktorové promìnné alespoò pomocí balíèku effects, i když to není
# pøesnì to, co hledáme


plot(Effect(c("horsepower","model_year"), lm_finale, partial.residuals = TRUE), nrow = 1)
plot(Effect(c("acceleration","weight"), lm_finale, partial.residuals = TRUE), nrow = 1)
plot(Effect(c("origin"), lm_finale, partial.residuals = TRUE), nrow = 1)
# vidíme, že origin by se teoreticky jako promìnná dala vynechat, jelikož støední hodnoty log(consumption)
# se pro jednotlivé faktory témìr rovnají a jednotlivé klastry pro každý faktor mají i pøibližnì stejný rozptyl
# zkusil jsem udìlat tuto úpravu a origin z finálního modelu odstranit
# tahle úprava ale vedla k porušení podmínky normality reziduí, proto jsem se tím dál nezabýval

# pro acceleration a cylinders vykreslíme grafy ruènì

# residual plot pro acceleration
coef_acceleration <- lm_finale$coefficients['acceleration']
Y = residuals(lm_finale) + coef_acceleration*data_mpghp[,'acceleration']
X = data_mpghp[,'acceleration']
plot(X,Y)
gamLine(x=X,y=Y) # smooth køivka
abline(lm(Y~X),col='red',lwd=2)

# residual plot pro cylinders
coef_cylinders <- lm_finale$coefficients['cylinders']
Y = residuals(lm_finale) + coef_cylinders*data_mpghp[,'cylinders']
X = data_mpghp[,'cylinders']
plot(X,Y,main='Partial residual plot cylinders')
quantregLine(x=X,y=Y)
abline(lm(Y~X),col='red',lwd=2)


# pro acceleration linearita vypadá dobøe
# u cylinders to je mírnì nabouráno skupinou s nejmenším poètem válcù, kde jsou 4 reprezentanti

#################################
# otázka è.21
#################################

# Výsledný model pro predikci je
summary(lm_finale)

# Model byl v pøedchozích krocích validován na pøedpoklady OLS a kolinearitu
# kolinearita je v modelu mírná a snížená do té míry, aby ještì procházely OLS pøedpoklady
# zkoušel jsem odstranit další nezávislé promìnné z modelu za ještì vìtším snížením kolinearity,
# nicménì tyto úpravy vedly na porušení
# OLS pøedpokladù (pøedevším normalita reziduí) a proto je zde už prezentovat nebudu


# Hodnota R^2 statistiky je 0.8906 (adjusted R^2 je 0.8853), což je považováno za velmi dobrou hodnotu
# Hodnota sigma modelu je pøibližnì 0.1143
sigma(lm_finale)

# Model odhaduje nelineárnì transformovanou velièinu log(consumption)
# což výraznì stìžuje interpretabilitu "kvality" regrese v pøípadì netransformované velièiny consumption
# což je bohužel nevýhoda tohoto modelu

# mùžeme "hrubì" spoèítat "sigma" pro netransformovanou velièinu,
# tahle hodnota je ale jen orientaèní porovnání s nìjakým NE log-lineárním modelem
# a nedá se interpretovat stejnì jako "pravá" sigma, co pøesnì tahle hodnota znamená není jasné
sqrt(1/(lm_finale$df.residual)*sum((data_mpghp[,'consumption']-exp(predict(lm_finale,data_mpghp)))^2))

#################################
# otázka è.22
#################################

# podiváme se na cook distance a residuals vs leverage
plot(lm_finale, which = 4)
plot(lm_finale, which = 5)

dim(data_mpghp)
# vlivná pozorování jsou v plotech vyznaèeny 124,174 a 342, který se dá ještì zaøadit pozorování 396, které se èasto
# objevovalo v avPlotech u partial regression plotù

# vypíšeme influence.measures (leave one out regression) a podíváme se na podezøelé influenèní body
# a jejich vliv na rùzné metriky

summary(influence.measures(lm_finale))
# z tabulky mùžeme vypozorovat, že všechny podezøelé body ovlivòují pøedevším covariance ratio
# a hlavní podezøelé body z regression vs leverage plotu také kvalitu fitu (dffits kritérium)

# vzhledem k tabulce se jako nejvíce podezøelé body jeví 124,174,342 , které mají také nejvìtší cook distance
# tato 3 pozorování jsou adepty na odstranìní




#################################
# otázka è.23
#################################


# Našli jsme nìkolik influenèních bodù vzhledem k leave one out. Po prozkoumání a srovnání
# podezøelých hodnot s ostatními (napø. s podobnou vahou, výkonem,...) jsem se nakonec rozhodl
# data z finálního modelu neodstraòovat ani nemodifikovat, protože se nezdá, že by tato pozorování byla ovlivnìna
# nìjakou systematickou chybou èi by se až pøíliš nepravdìpodobnì odchylovala od ostatních záznamù.

#################################
# otázka è.24
#################################


# budeme zde porovnávat s klasickým modelem
# první nastavíme novou promìnnou s daty bez nejvýznamìjších influenèních bodù
data_noinfluential = data_mpghp[!(rownames(data_mpghp) %in% c("124","174","342")),]
data_noinfluential

# finální klasický model jsme již v druhé èásti protokolu
# vybrali model s logaritmicky škálovanou spotøebou umocnìnou na druhou
summary(lm_logcons_interceptsquared)
lm_logsquared_noinf = lm(log(consumption)^2 ~ weight,data_noinfluential) # model bez 
summary(lm_logsquared_noinf)


# nyní nachystáme MM modely, jako psi použijeme ggw a bisquare

library(robustbase) # Basic Robust Statistics
library(rrcov)      # Scalable Robust Estimators with High Breakdown Point
library(robust)

MM_bisquare_stars<- rlm(log(consumption)^2 ~ weight, method="MM",psi = psi.bisquare, data = data_mpghp) # Tukey
MM_ggw   <- rlm(log(consumption)^2 ~ weight, method="MM",psi = psi.ggw, data= data_mpghp)


# a LTS s 50% a 90% pozorování

LTS_50  <- ltsReg(log(consumption)^2 ~ weight, alpha=0.5, data = data_mpghp)
LTS_90  <- ltsReg(log(consumption)^2 ~ weight, alpha=0.9, data = data_mpghp)


# vykreslíme všechny modely

p <- ggplot(data, aes(x=weight, y=log(consumption)^2)) + geom_point()+
  geom_line(aes(x=weight,y=lm_logcons_interceptsquared$fitted.values,color = 'red'))+
  geom_line(data = data_noinfluential,aes(x=weight,y=lm_logsquared_noinf$fitted.values,color='blue'))+
  geom_line(aes(x=weight,y=MM_bisquare_stars$fitted.values,color='yellow')) +
  geom_line(aes(x=weight,y=MM_ggw$fitted.values,color='orange')) +
  geom_line(aes(x=weight,y=LTS_50$fitted.values,color='green')) +
  geom_line(aes(x=weight,y=LTS_90$fitted.values,color='brown')) +
  scale_color_discrete(name = 'methods',labels = c('classic','no influential points','MM_bisquare','MM_ggw','LTS 50%','LTS 90%'))
plot(p)

# vidíme, že regresní pøímky jsou ve všech pøípadech pøibližnì stejné


#################################
# otázka è.25
#################################

summary(lm_finale)

# když se podíváme na signifikanci jednotlivých promìnných, vidíme, že horsepower:model_year79 až horsepower:model_year 82
# jsou ménì signifikantní než ostatní, navíc jsme v pøedchozích otázkách zjistili (z grafù),
# že by se promìnná model_year dala slouèit na ménì faktorù. V tomto pøípadì by možná mohlo pomoci nechat všechny faktory až na 80-82, které bychom slouèily dohromady
# rok 79 bych tam již nepøidával, hodnoty consumption pro tento faktor (a tedy støední hodnota) se od 80-82 znaènì liší, viz obrázek

p1 <- ggplot(data_mpghp,aes(x=model_year,y=consumption, fill = model_year)) + 
  geom_boxplot(notch=FALSE,outlier.color = 'black') + scale_fill_discrete(name='year')
plot(p1)

# transformace by se mohla nabízet pro numerickou promìnnou cylinders, což plyne z partial residual plotu
# kde nám vyšla mírná nelineární závislost, ostatní promìnné bych netransformoval ani nepøevádìl na faktory


# vzhledem k partial residual plotu pro cylinders by se nabízela transformace e^x

lm_transf_cylinders = lm(log(consumption)~exp(cylinders)+origin+acceleration+horsepower:model_year+acceleration:weight,data_mpghp)

coef_cylinders <- lm_transf_cylinders$coefficients['exp(cylinders)']
Y = residuals(lm_transf_cylinders) + coef_cylinders*data_mpghp[,'cylinders']
X = exp(data_mpghp[,'cylinders'])
plot(X,Y,main='Partial residual plot cylinders')
quantregLine(x=X,y=Y)
abline(lm(Y~X),col='red',lwd=2)

# z grafu vidíme, že se nám sice vztah pro tuto promìnnou podaøilo linearizovat, nicménì
# se stává tato promìnná redundantní, sklon pøímky je v podstatì konstantní

# vzhledem k této skuteènosti a taky tomu, že vif hodnota pro cylinders je v modelu docela vysoká,
# což by mohla zpùsobovat vyšší kolinearitu, zkusíme cylinders z modelu úplnì odstranit

lm_no_cylinders = lm(log(consumption)~origin+acceleration+horsepower:model_year+acceleration:weight,data_mpghp)
summary(lm_no_cylinders)

anova(lm_no_cylinders,lm_finale) # vidíme, že anova tìsnì preferuje model bez cylinders
# zkusíme zkontrolovat OLS pøedpoklady

shapiro.test(residuals(lm_no_cylinders)) # normalita se nezamítá

plot(lm_no_cylinders, which = 3)
bptest(lm_no_cylinders) # homoskedasticita se tìsnì zamítá, diagnostické grafy ale nevypadají úplnì špatnì, tohle se dá tolerovat

# odstranìním cylinders tedy obdržíme nový model s nižší kolinearitou, který anova preferuje o nìco více než pùvodní
# nicménì, rozdíl je minimální


#################################
# otázka è.26
#################################

# Je zøejmé, že nemùžeme jen tak pøedpokládat, že pokud naložíme nìjakou zátìž na libovolný automobil
# tak se jeho spotøeba bude chovat lineárnì podle námi nalezeného modelu
# Musíme na to jít jinak.

# Mùj postup by byl takovýto:

# 1) vyrobíme jednoduchý klasický model (lm(consumption ~ weight - 1)- použijeme model bez interceptu, protože auto vážící 0kg logicky nespotøebuje nic, tedy 0l paliva), který nám dá lineární odhad závislosti spotøeby na váze
# 2) vytvoøíme podobné skupiny aut vzhledem k nìkolika nezávislým promìnným urèujícím spotøebu (napø. pomocí K-Means algoritmu, urèitì by to šlo udìlat i dùkladnìji a jinak)
# vzhledem k již vyrobenému vícerozmìrnému lin. modelu lm_finale zvolíme za tyto promìnné cylinders,horsepower,acceleration (zajímají mì pøedevším klastry na základì technických parametrù aut, proto origin ani model_year uvažovat nebudeme)
# 3) jako poslední krok se podíváme na rezidua u jednotlivých skupin a zjistíme, jak moc pøesnì tyto skupiny opisují regresní pøímku danou modelem
# a vezmeme tu skupinu, která bude mít nejmenší rezidua (suma reziduí^2 normovaná poètem èlenù ve skupinì) a vizuálnì bude opisovat nejlépe regresní pøímku
# u takových skupin lze oèekávat velmi podobné hodnoty v ostatních promìnných kromì weight a tedy mùžeme pøedpokládat, že u aut s takovými hodnotami
# se závislost spotøeby na váze automobilu chová "pøibližnì" lineárnì - hrubì si to pøedstavuji tak, že zafixujeme ostatní promìnné kromì weight a sleduju zmìnu spotøeby jen na základì pøidávání váhy
# nakonec vybereme jedno auto z této skupiny dle uvážení,
# mìli bychom vybrat nejlepšího reprezentanta dané skupiny (vektor nejblíže ke støedu klastru), pokud bude takových aut více, pak vezmeme to nejlehèí,
# protože pak máme vìtší jistotu, že pøidáním nejvìtší váhy se spotøeba zmìní lineárnì 
# (pokud v této skupinì máme nejlehèí auto vážící 1000kg a nejtìžší 1500kg, pak u 1000kg auta naložením 500kg ještì stále mùžu pøedpokládat lineární závislost spotøeby)

# rozdìlíme do 20 klastrù pomocí KMeans

data_temp = data_mpghp[,c('cylinders','horsepower','acceleration')]

# naklastrujeme

clusters <- kmeans(data_temp, centers =  20, iter.max = 100)

data_mpghp$clusters = clusters$cluster


lm_classic <- lm(consumption ~ weight - 1, data_mpghp) # model bez interceptu

# jako první si vykreslíme regresní pøímku a scatterploty všech 20 skupin, vizuálnì
# jsem zúžil výbìr na 4 adepty, které dobøe opisují regresní pøímku, grafy zde znázorním

# adepti: cluster 2,14,16,17


p1 <- ggplot(data_mpghp[data_mpghp[,'clusters']==2,], aes(weight, consumption))+geom_point()+
  geom_line(data = data_mpghp, aes(x=weight,y=lm_classic$fitted.values), size = 0.8)

p2 <- ggplot(data_mpghp[data_mpghp[,'clusters']==14,], aes(weight, consumption))+geom_point()+
  geom_line(data = data_mpghp, aes(x=weight,y=lm_classic$fitted.values), size = 0.8)

p3 <- ggplot(data_mpghp[data_mpghp[,'clusters']==16,], aes(weight, consumption))+geom_point()+
  geom_line(data = data_mpghp, aes(x=weight,y=lm_classic$fitted.values), size = 0.8)

p4 <- ggplot(data_mpghp[data_mpghp[,'clusters']==17,], aes(weight, consumption))+geom_point()+
  geom_line(data = data_mpghp, aes(x=weight,y=lm_classic$fitted.values), size = 0.8)

figure <- ggarrange(p1, p2, p3, p4,
                    labels = c("klastr 2", "klastr 14", "klastr 16","klastr 17"),
                    ncol = 2, nrow = 2)
plot(figure)

# tyto skupiny vizuálnì nejlépe "opisují" regresní pøímku, navíc jsou všechny pøibližnì stejnì poèetnì zastoupeny, což je výhoda vzhledem k vzájemnému porovnávání

# Nyní spoèítáme a porovnáme sumu normovaných reziduí^2 pro tyto skupiny

res_2 = (lm_classic$residuals[data_mpghp[,'clusters']==2])^2
sum_res_2 = (1/length(res_2))*sum(res_2)
sum_res_2

res_14 = (lm_classic$residuals[data_mpghp[,'clusters']==14])^2
sum_res_14 = (1/length(res_14))*sum(res_14)
sum_res_14

res_16 = (lm_classic$residuals[data_mpghp[,'clusters']==16])^2
sum_res_16 = (1/length(res_16))*sum(res_16)
sum_res_16

res_17 = (lm_classic$residuals[data_mpghp[,'clusters']==17])^2
sum_res_17 = (1/length(res_17))*sum(res_17)
sum_res_17

# Nejlépe dopadl klastr 17, budeme tedy uvažovat tuto skupinu
# jako nejlepšího reprezentanta vybereme bod nejblíže ke støedu klastru,
# tj. auto s nejbližší vzdáleností k 

center = clusters$centers[17,]

# tímto autem je chrysler duster z roku 74 vážící 1407.044 kg

# u tohoto auta mùžeme pøedpokládat lineární závislost zvýšení spotøeby pøi pøidání nákladu

# pøi pøevozu 800kg tedy musí kamarád doplatit

# (koeficient weight v lin. modelu)*(800kg)*(vzdálenost Brno hl.n. - Praha Trojanova 13 v jednotkách 100km)*30 =
0.0083554*800*2.05*30
# kamarád by mìl doplatit pøibližnì 411 Kè.


