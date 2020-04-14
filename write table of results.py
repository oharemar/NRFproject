import xlwings as xw

macroB = xw.Book('template.xlsx')
weightedB = xw.Book('template1.xlsx')
accuracyB = xw.Book('template2.xlsx')

macro_sheet = macroB.sheets['List1']
weighted_sheet = weightedB.sheets['List1']
accuracy_sheet = accuracyB.sheets['List1']



models = ['NN','logistic regression','random forest', 'random forest 30 estimators','random forest 50 estimators',
          'NRF_analyticWeights','NRF_analyticWeights_adam','NRF_analyticWeights_nesterov','NRF_basic','NRF_basic_adam',
          'NRF_basic_nesterov','NRF_extraLayer','NRF_extraLayer_adam','NRF_extraLayer_analyticWeights','NRF_extraLayer_analyticWeights_adam',
          'NRF_extraLayer_analyticWeights_nesterov','NRF_extraLayer_nesterov']

alphabet = ['B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R']

datasets = ['bank_marketing','cars','diabetes','messidor','USPS','vehicle_silhouette','wine','OBSnetwork']

j = 1
for dataset in datasets:
    j += 1
    print(dataset)
    for model in models:
        print(model)
        ind = models.index(model)
        k = alphabet[ind]
        avg1 = {}
        std1 = {}
        avg2 = {}
        std2 = {}
        with open('RESULTS_PUBLIC_DATASETS_betterSoftmax/{}/{}_mean.txt'.format(model,dataset), 'r') as file:
            avg1 = eval(file.read())
        with open('RESULTS_PUBLIC_DATASETS_betterSoftmax/{}/{}_std.txt'.format(model,dataset), 'r') as file:
            std1 = eval(file.read())
        with open('RESULTS_PUBLIC_DATASETS_betterSoftmax_ave/{}/{}_mean.txt'.format(model,dataset), 'r') as file:
            avg2 = eval(file.read())
        with open('RESULTS_PUBLIC_DATASETS_betterSoftmax_ave/{}/{}_std.txt'.format(model,dataset), 'r') as file:
            std2 = eval(file.read())
        # vytáhneme jednotlivé hodnoty, porovnáme, vybereme maximum a zapíšeme do příslušných tabulek
        val1 = None
        val2 = None
        val3 = None
        macro1 = avg1['macro avg']['f1-score']
        macro2 = avg2['macro avg']['f1-score']
        if macro1 > macro2:
            val1 = str(macro1) + '+-' + str(std1['macro avg']['f1-score'])
        else:
            val1 = str(macro2) + '+-' + str(std2['macro avg']['f1-score'])
        weighted1 = avg1['weighted avg']['f1-score']
        weighted2 = avg2['weighted avg']['f1-score']
        if weighted1 > weighted2:
            val2 = str(weighted1) + '+-' + str(std1['weighted avg']['f1-score'])
        else:
            val2 = str(weighted2) + '+-' + str(std2['weighted avg']['f1-score'])
        acc1 = avg1['accuracy']
        acc2 = avg2['accuracy']
        if acc1 > acc2:
            val3 = str(acc1) + '+-' + str(std1['accuracy'])
        else:
            val3 = str(acc2) + '+-' + str(std2['accuracy'])


        macro_sheet.range('{}{}'.format(k,str(j))).value = val1
        weighted_sheet.range('{}{}'.format(k,str(j))).value = val2
        accuracy_sheet.range('{}{}'.format(k,str(j))).value = val3

macroB.save('macro public results.xlsx')
weightedB.save('weighted public results.xlsx')
accuracyB.save('accuracy public results.xlsx')

