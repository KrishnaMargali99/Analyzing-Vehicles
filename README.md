Our analysis of the electric vehicle population in Washington, US from 1997 to 2022 has yielded the following
findings:


EDA
● The dataset contains records ranging from the year 1997 to 2023, with the majority of the electric vehicles
manufactured between 2016 and 2022 and having a range between 0 to 250 miles. (Refer Fig. 1 - Appendix A)
● Vehicles that are eligible for the CAFV program seem to have a slightly higher range, with a peak around 200 to
250 miles. The non-CAFV eligible vehicles, on the other hand, are concentrated around the 0 to 100 miles range.
(Refer Fig. 2 - Appendix A)

● The majority of battery electric vehicles (BEVs) have an electric range of less than 300 miles, while plug-in
hybrid electric vehicles (PHEVs) have a wider range distribution with some vehicles having an electric range of
over 500 miles. (Refer Fig. 3 - Appendix A)

● EVs with longer ranges tend to be more expensive, and those with higher base MSRP are eligible for CAFV
incentives. (Refer Fig. 4 - Appendix A)
Data Preprocessing

● Features VIN (1-10), DOL Vehicle ID, Vehicle Location, Electric Utility, and 2020 Census Tract have no effect on
the target variable and are hence dropped.
● The target variable is continuous and is converted into a categorical variable (‘Electric Range Category’) for
classification by defining range intervals and assigning a category to each interval.
● The features Electric Range and Base MSRP are positively skewed with 0.93 and 6.4 respectively.
● Outlier Detection using Mahalanobis distance criteria is used to remove all the discrepancies in the Electric Range
feature.
● Features with importance scores close to zero, including 'Electric Range', 'Electric Range Category', 'City', 'State',
'County', 'Postal Code', and 'Legislative District' are removed from the list of features for ML models.
ML Methods
● Classifier: The Decision Tree classifier achieves a significantly higher accuracy score of 98.52% compared to the
Logistic Regression classifier's accuracy score of 58.86%. The cross-validation scores suggest that the Decision
Tree classifier is likely to generalize better to new data than the Logistic Regression classifier.
5
● Regressor: The Random Forest Regressor achieves a much lower cross-validation MSE of 26.051983 compared
to the Linear Regressor's cross-validation MSE of 4888.003900.
● On the other hand, the R-squared score for the Linear Regressor is much lower, suggesting that the independent
variables do not explain as much of the variance in the dependent variable.
● Clustering: K-means clustering algorithm is used to group the electric vehicles into 3 clusters based on the
Electric Range variable as follows - short range, medium range, and long range. The optimal number of clusters is
found using the Elbow Method.
● Advance Method: The ensemble model achieves a much lower MSE of 5.07 compared to the Random Forest
model alone, indicating that the ensemble model is better at predicting the electric range of vehicles than the
Random Forest model alone


Based on these findings, we can suggest the following actionable insights:
● To incentivize the adoption of EVs, policymakers could consider increasing incentives for vehicles with longer
ranges, particularly those that are not eligible for the CAFV program.
● Dealerships and manufacturers can use the clustering results to target marketing efforts and product development
towards specific segments of the EV market.
● Machine learning models, particularly the Random Forest Regressor and ensemble model, can be used to predict
the electric range of new EV models with reasonable accuracy, thereby aiding in the product development
process.
