# Airbnb Berlin Price Prediction

## Objectif du projet :

Le marché des locations court-terme est en forte expansion dans les villes touristiques, et à Berlin y compris. L'objectif de cette étude est d'estimer notre capacité à prédire le prix d'une location en fonction des différentes caractéristiques d'une annonce sur la plateforme Airbnb. Ce problème est tiré d'un challenge disponible à [ce lien](https://dphi.tech/challenges/data-sprint-47-airbnb-berlin-price-prediction/160/data). Le git du projet est disponible [ici](https://gitlab-student.centralesupelec.fr/2019clemotl/airbnb-berlin-price-prediction-ml-2223).


## Présentation de l'approche :

La résolution de ce problème s'est faite en 3 temps :
- Analyse des données
- Preprocessing des données
- Entrainement et évaluation de différents modèles de régression

Finalement, nous calculons l'erreur RMSE et le coefficient R2 sur notre set de données test pour les deux modèles les plus performants.

## Structure des fichiers :

### Analyse des données :
Les statistiques des différentes features et une heatmap de corrélation sont crées par le script [stats.py](./Models/Preprocessing/stats.py).

### Preprocessing :
Le preprocessing modulable de la donnée se trouve dans le script [preprocessing_wrapper.py](./Models/Preprocessing/preprocessing_wrapper.py).

Les différents outils de preprocessing (PCA, scaling) ou encore la cross-validation qui sera utilisée pour évaluer les différents modèles se trouvent dans le dossier [Preprocessing](./Models/Preprocessing).

### Evaluation des différents modèles :
Les différents modèles évalués sont les suivants :
- La régression linéaire ([Reg.py](./Models/Reg.py))
- Les k plus proches voisins ([kNN.py](./Models/kNN.py))
- L'arbre de décision ([Decision_Tree.py](./Models/Decision_Tree.py))
- La forêt aléatoire ([Random_Forest.py](./Models/Random_Forest.py))
- Le AdaBoost ([AdaBoost.py](./Models/AdaBoost.py))
- Le xgBoost ([xgBoost.py](./Models/xgBoost.py))

Chaque script justifie le choix des hyperparamètres et indique les mesures de performance moyennées (en utilisant la cross-validation).

## Conclusion
Pour prédire les valeurs de prix du file de test stratifié il faut exécuter le fichier [Best_Models.py](Best_Models.py) avec la commande python3 Best_Models.py.

Ce fichier affiche le R2 et RMSE pour les deux modèles les plus performants : le XGBoost et le Random Forest.
