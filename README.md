# ML-HousePricePrediction
Growing unaffordability of housing has become one of the major challenges for metropolitan cities around the world. In order to gain a better understanding of the commercialized housing market we are currently facing; we want to figure out what are the top influential factors of the housing price. Apart from the more obvious driving forces such as the inflation and the scarcity of land, there are also several variables that are worth looking into. The task here is to reach a model that can closely predict the pricing of a house.

## Data Preprocessing

- The pre processing of data begins by looking out for missing values in each explainatory variables.
- For this dataset we are considering to eliminate those variables which have or have more than 45% of total data missing or showing no values. `PoolQC`, `MiscFeature`, `Fence`, and `FireplaceQu` attributes for predicting house prices have very large numbers of missing values, which are, 0.99, 0.96, 0.81 and 0.47 respectively. Since it covers almost whole columns eliminating these attributes won't cause any significant loss of values to our project. It is also important to be aware that right evaluation is required to deal with such missing values because there may be chance of loss of important data.
- `LotFrontage` will be considered in our dataset. It has 211 missing values which is significant number and will replace with Median. However, I think it has moderate relationship with house price.
