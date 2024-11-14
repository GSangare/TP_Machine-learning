# OPTIMISATION ET ÉVALUATION DE MODÈLES DE RÉGRESSION : RIDGE, LASSO, ET ELASTICNET
==================================================================================

Ce projet explore trois modèles de régression régularisée : Ridge, Lasso, et ElasticNet.
Chaque modèle est optimisé en ajustant ses hyperparamètres et en évaluant ses performances
grâce à des métriques telles que l'erreur quadratique moyenne (MSE) et le coefficient de détermination (R²).

## PRÉREQUIS
---------
Avant de commencer, installez les bibliothèques Python nécessaires :
> pip install numpy matplotlib seaborn scikit-learn

## STRUCTURE DU PROJET
-------------------
1. Préparation des données : Définissez vos variables d'entrée (X) et de sortie (y).
2. Optimisation des modèles : Utilisation de GridSearchCV pour trouver les meilleurs
   hyperparamètres pour Ridge, Lasso, et ElasticNet.
3. Évaluation des modèles : Calcul du MSE, RMSE et R² pour évaluer les performances des modèles.
4. Visualisation des résultats :
   - Comparaison des valeurs réelles et prédites
   - Distribution des résidus
   - Évolution de l'erreur en fonction des hyperparamètres

## UTILISATION
-----------
1. **Optimisation des modèles** :
    - Ridge :
        best_ridge, best_alpha_ridge, grid_search_ridge = optimize_ridge(X_train, y_train)
    - Lasso :
        best_lasso, best_alpha_lasso, grid_search_lasso = optimize_lasso(X_train, y_train)
    - ElasticNet :
        best_elastic, best_params_elastic, grid_search_elastic = optimize_elasticnet(X_train, y_train)

2. **Évaluation des modèles** :
    - Ridge :
        mse_train_ridge, rmse_train_ridge, r2_train_ridge, mse_test_ridge, rmse_test_ridge, r2_test_ridge = evaluate_model(best_ridge, X_train, X_test, y_train, y_test)
    - Lasso :
        mse_train_lasso, rmse_train_lasso, r2_train_lasso, mse_test_lasso, rmse_test_lasso, r2_test_lasso = evaluate_model(best_lasso, X_train, X_test, y_train, y_test)
    - ElasticNet :
        mse_train_elastic, rmse_train_elastic, r2_train_elastic, mse_test_elastic, rmse_test_elastic, r2_test_elastic = evaluate_model(best_elastic, X_train, X_test, y_train, y_test)

3. **Visualisation des résultats** :
    - Comparaison des valeurs réelles et prédites :
        plot_pred_vs_true(y_test, y_test_pred_ridge, "Ridge")
        plot_pred_vs_true(y_test, y_test_pred_lasso, "Lasso")
        plot_pred_vs_true(y_test, y_test_pred_elastic, "ElasticNet")

    - Distribution des résidus :
        plot_residuals(y_test, y_test_pred_ridge, "Ridge")
        plot_residuals(y_test, y_test_pred_lasso, "Lasso")
        plot_residuals(y_test, y_test_pred_elastic, "ElasticNet")

    - Évolution de l'erreur en fonction de l'hyperparamètre alpha :
        plot_error_vs_alpha(grid_search_ridge)
        plot_error_vs_alpha(grid_search_lasso)
        plot_error_vs_alpha_elasticnet(grid_search_elastic)

AUTEUR
------
Ce projet a été réalisé pour analyser les performances des modèles de régression
Ridge, Lasso, et ElasticNet en utilisant des techniques de régularisation pour la prédiction.

