# ML-HousePricePrediction
Growing unaffordability of housing has become one of the major challenges for metropolitan cities around the world. In order to gain a better understanding of the commercialized housing market we are currently facing; we want to figure out what are the top influential factors of the housing price. Apart from the more obvious driving forces such as the inflation and the scarcity of land, there are also several variables that are worth looking into. The task here is to reach a model that can closely predict the pricing of a house.

# Data Preprocessing

- The preprocessing of data begins by looking out for missing values in each explainatory variables.
- For this dataset we are considering eliminating those variables which have or have more than 45% of total data missing or showing no values. **PoolQC**, **MiscFeature**, **Fence**, and **FireplaceQu** attributes for predicting house prices have very large numbers of missing values, which are, 0.99, 0.96, 0.81 and 0.47 respectively. Since it covers almost whole columns, eliminating these attributes won't cause any significant loss of values to our project. It is also important to be aware that right evaluation is required to deal with such missing values because there may be a chance of loss of important data.
- **LotFrontage** will be considered in our dataset. It has 211 missing values, which is a significant number, and we will replace it with Median. However, I think it has a moderate relationship with the house price.
- Before analyzing Garage-related variables, it seems all variables are related to having or not having garage in the house. Garages do have good relationship with the price of the house. Since our aim is the least deletion of data, we will consider all other data. 
  - **GarageCond**: In this variable, 'na' can be put into a new category 'No Garage'. None category will describe the houses with no garage.
  - **GarageType**: This variable describes the location of the garage. So, 'na' does make sense that there is no garage in the house. So, I will create a new category, which is No Garage.
  - **GarageYrBuilt**: This variable describes the year when the garage was built. So, I will make this variable the categorical variable. The data with na will go to the 'Unknown' category.
  - **GarageFinish**: This categorical variable describes interior finishes of the garage. The 'na's' indicating missing values do make sense here that it represents there are no garages. The data with na will go to 'No Garage' category.
  - **GarageQual**: This categorical variable describes the quality of the garage. The 'na' indicates the houses with no garage. The na values will go to No Garage                         category.
- Before analyzing Basement related variables, it seems all variables are related to having or not having basement in the house. Basements do have a good relationship with the price of the house. Since our aim is the least deletion of data, we will consider all other data.
  - **BsmtExposure**: The 'na' values can be put into the No Basement category.
  - **BsmtFinType2**: The 'na' values can be put into the No Basement category.
  - **BsmtFinType1**: The 'na' values can be put into the No Basement category.
  - **BsmtCond**: It indicates, condition of the basement. The 'na' values can be put into No Basement category.
  - **BsmtQual**: It indicates overall qualities of the basement. The 'na' values can be put into No Basement category.
- Masonry veneer Masonery has been put into the house and it could be one of the significant variables that influence the selling price.
  - **MasVnrType**: We can categorize na values into the Unknown category.
  - **MasVnrArea**: Since we only have 6 missing values, we will impute those values with Median value.
- **FireplaceQu** is categorical variable; I will make another category for null values. 'No Fireplace' replaces the missing values of FireplaceQu.
