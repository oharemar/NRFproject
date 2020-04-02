# diplomka

1) dodělat modely - NRF with extra layer - ReLU do mezivrstvy (1)
                  - v NRF with extra layer doimplementovat adam a nesterov (1)

2) najít další public datasety - mít alespoň 7 datasetů - vybírat tak, aby se dalo porovnat s jinými studiemi (1)
3) přichystat skript pro iteraci přes všechny datasety a přes všechny modely, použít i logistickou regresi a klasickou neuronku (1)
4) prozkoumat cross validaci v scikit, případně naimplementovat custom verzi (1-2)
5) posbírat výsledky na všech metrikách z classification report - spustit skript z bodu 3) a na vyhodnocení použít buď 5 fold crossvalidation nebo spustit 5x experiment pokaždé s jinou trénovací a testovací množinou, výsledky uložit do souboru (2-4)
6) posbírat average a std výsledků (2-4)
7) na základě výsledků zkusit vysvětlit, interpretovat, co vyšlo, pokud něco vyšlo nečekaně, zkusit vylepšit (4)
8) pro binární klasifikaci připravit ROC (AUC) a Precision Recall křivky, pro multiclass dělat nebudeme (5)
9) implementovat sparse connectivity training a "moje trénování - technika přihazování" (6-7)
10) posbírat stejně výsledky jako doposud pro modely z bodu 9) (7)
11) vyhodnotit a porovnat výsledky z bodu 10) s výsledky z bodu 5) a 8) (8-9)
12) připravit klasifikační úlohu pro Egypt (9-10)
13) spustit nejlepší model z vyhodnocení na public datasetech na Egypt a porovnat s klasickým RF a neuronkou na Egyptu (10-11)
14) načíst si články o výpočtu complexity a provést hrubší analýzu NRF vs RF vs NEURONKA
