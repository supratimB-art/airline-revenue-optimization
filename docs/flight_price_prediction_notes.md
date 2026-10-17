# Separate notebook: flight-price prediction

This is a record of a separate exploratory machine-learning notebook, **not** the airline seat-inventory DP model.

## What the notebook records

The notebook describes scraping flight information from Yatra using Selenium and collecting **5,303 rows**. It reads a local file named `Flight_Prices.xlsx`, which is not part of the supplied source set in this repository. It uses ticket price as the regression target and works with features including airline, source, destination, number of stops, meal availability, duration, and departure/arrival time components.

The workflow includes exploratory data analysis, cleaning, feature engineering, categorical encoding, scaling, a train/test split, several tree-based ensemble regressors, and a GridSearchCV experiment for Extra Trees.

## Results recorded in the notebook

These are the notebook's displayed results, not metrics reproduced by the current repository.

| Model | Notebook-reported R² |
|---|---:|
| Decision Tree Regressor | 51.30% |
| Random Forest Regressor | 72.74% |
| Extra Trees Regressor | 77.47% |
| Gradient Boosting Regressor | 60.35% |
| XGBoost Regressor | 73.08% |
| Bagging Regressor | 69.30% |
| Tuned Extra Trees Regressor | 77.61% |

For the final Extra Trees result, the notebook records MAE of approximately **₹1,200.30**, MSE of **3,732,348.50**, and RMSE of approximately **₹1,931.93**. These scores belong to the notebook's particular dataset, feature processing, split, and software environment; they should not be presented as current or independently validated model performance.

## Reproducibility and interpretation notes

1. **Data is missing.** The notebook expects `Flight_Prices.xlsx` and saves/loads `Flight_Ticket_Price_Prediction.pkl`; neither is included in the uploaded source set. Therefore the project cannot be rerun end-to-end from the notebook alone.
2. **Scaling happens before splitting.** The notebook calls `StandardScaler.fit_transform(x)` before `train_test_split`. This lets information about the held-out feature distribution enter preprocessing. A clean rerun should put scaling and categorical preprocessing in a scikit-learn Pipeline fitted only on the training folds.
3. **Model-selection notes differ.** An earlier cell searches random states and records its best Random Forest split as 122, but the later model comparison resets the split to random state 10. The reported results correspond to the notebook's recorded run and should be accompanied by the exact split and evaluation procedure if reused.
4. **Modern compatibility may differ.** Some GridSearchCV candidates and final parameters in the notebook use legacy settings (including `criterion='mae'` and `max_features='auto'`) that are not accepted by all current scikit-learn versions. A reproducible rerun should document package versions and update the search space.
5. **Prediction is not control.** Predicting the price associated with a flight from a feature row is different from deciding whether to accept a booking at that fare given inventory, remaining time, and future fare opportunities.
6. **No claim about causal price dynamics.** The described feature table does not by itself establish how a particular ticket's price changes as departure approaches. That would require appropriately timestamped price observations and a suitable evaluation design.

The dataset itself is not redistributed here. Follow the source website's terms and applicable data-use requirements before recreating a scraped dataset.
