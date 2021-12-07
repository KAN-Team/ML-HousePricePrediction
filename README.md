# ML-HousePricePrediction
Growing unaffordability of housing has become one of the major challenges for metropolitan cities around the world. In order to gain a better understanding of the commercialized housing market we are currently facing; we want to figure out what are the top influential factors of the housing price. Apart from the more obvious driving forces such as the inflation and the scarcity of land, there are also several variables that are worth looking into. The task here is to reach a model that can closely predict the pricing of a house.

# Data Preprocessing

- The pre processing of data begins by looking out for missing values in each explainatory variables.
- For this dataset we are considering to eliminate those variables which have or have more than 45% of total data missing or showing no values. **PoolQC**, **MiscFeature**, **Fence**, and **FireplaceQu** attributes for predicting house prices have very large numbers of missing values, which are, 0.99, 0.96, 0.81 and 0.47 respectively. Since it covers almost whole columns eliminating these attributes won't cause any significant loss of values to our project. It is also important to be aware that right evaluation is required to deal with such missing values because there may be chance of loss of important data.
- **LotFrontage** will be considered in our dataset. It has 211 missing values which is significant number and will replace with Median. However, I think it has moderate relationship with house price.
- Before analyzing Garage related variables, it seems all variables are related with having or not having garage in the house. Garagecars do have good relationship with the price of the house. Since our aim is the least deletion of data, we will consider all other data.
  - **GarageCond**: In this variable, 'na' can be put into new category 'No Garage'. None category will describe the housees with no garage.
  - **GarageType**: This variable describes the location of garage. So, 'na' does make sense that there are no garage in the house. So, I will create new category No Garage.
  - **GarageYrBuilt**: This variable describes the year garage was built. So, I will make this variable the categorical variable. The data with na will go into 'Unknown' category.
  - **GarageFinish**: This categorical variable describes interior finishes of the garage. The 'na's indicating missing values do make sense here that it represents there are no garages. The data with na will go into 'No Garage' category.
  - **GarageQual**: This categorical variable describes quality of the garage. The 'na' indicates the houses with no garage. The na values will go into No Garage category.
- Before analyzing Basement related variables, it seems all variables are related with having or not having basement in the house. Basements do have good relationship with the price of the house. Since our aim is the least deletion of data, we will consider all other data.
  - **BsmtExposure**: The 'na' values can be put into No Basement category.
  - **BsmtFinType2**: The 'na' values can be put into No Basement category.
  - **BsmtFinType1**: The 'na' values can be put into No Basement category.
  - **BsmtCond**: It indicates condtion of the basement. The 'na' values can be put into No Basement category.
  - **BsmtQual**: It indicates overall qualties of the basement. The 'na' values can be put into No Basement category.
