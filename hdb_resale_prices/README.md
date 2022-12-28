<h1 align='center'>HDB Resale Prices</h1>

## Project Introduction
Public housing in Singapore is managed by the Housing and Development Board. The aim of public housing is to provide affordable housing options to Singaporeans despite domestic land constraints. This is achieved through high-rise housing structures and grants that make the costs of housing affordable for the average Singaporean. In the recent years, there have been an increasing number of HDB flats making headlines for record-breaking transaction prices.

In this project, we explore the factors that might affect HDB resale prices in Singapore. Since majority of Singaporeeans own HDB units, this could give them an indication as to the current prices of their property. In this part, we will work on publicly available datasets to generate a set of possible factors that might affect resale prices. In the next part, we will make use of data analytics to examine the effect of housing prices on these factors, and use models to predict the resale prices of HDB flats in Singapore.

The completed API is available for use on Render [ here ](https://hdb-resale-prices-calculator.onrender.com/) (RIP Heroku Free Plan).

## Change Log
\[16/05/2022]
* Changed methodology for primary school score calculation, to take into account ranking of primary schools and balloting priority system based on distance of house from school.
* Updated list of shopping centres and primary schools (removed Juying Primary School due to merger)

## Table of Contents

1. [ Methods/Technologies Used ](#Methods_Techonologies_Used)
2. [ Project Description ](#Project_Description)
3. [ File Descriptions ](#File_Descriptions)
4. [ Credits ](#Credits)

<a name="Methods_Techonologies_Used"></a>
## Methods/Technologies Used

### Technologies
* Python
* pandas
* NumPy
* Matplotlib
* scikit-learn
* FastAPI
* Heroku/Render

### Methods
Preprocessing
* Standardisation: StandardScaler
* Feature Encoding: OneHotEncoder

Model Building
* scikit-learn Pipelines
* Linear Models: Linear Regression, Ridge Regression
* Tree Models: Random Forest Regressor, XGBoost Regressor
* Hyperparameter Tuning: RandomizedSearchCV, GridSearchCV

<a name="Project_Description"></a>
## Project Description

### Housing Price Dataset
* A list of 108,048 HDB resale transactions from 2017 to 2021 was extracted from data.gov.sg.
* Using the OneMap API, the addresses of these HDB resale flats were geocoded. A list of MRTs/LRTs, primary/secondary schools, and shopping malls in Singapore was also geocoded. These coordinates were then used to determine and calculate the distance of the nearest amenities to each resale flat.
* To remove flunctuations in housing prices through inflation, the Price Index for resale flats was used to adjust housing resale prices to 2021 Q2.

### Housing Price Model
* Supervised machine learning methods were applied to the final dataset, to determine the best model for application.
* Linear Regression and Ridge Regression were used for linear models. Dataset was fed through a pipeline where continuous features were standardised using StandardScaler and categorical features were encoded using OneHotEncoder.
  * For linear models, an R^2 value of .910 and a root mean squred error of 0.00962 was obtained.
* Random Forest Regressor and XGBoost Regressor were used for tree-based models. Dataset was fed through a pipeline where only categorical features were encoded using OneHotEncoder.
  * Hyperparameter tuning was also applied used RandomizedSearchCV to determine the best hyperparameters for each model.
  * For tree-based models, an R^2 value of .969 and a root mean squared error of 0.00328 was obtained.
* The model ultimately chosen for deployment was an XGBoost Regressor with tuned hyperparameters.

### Deployment
* FastAPI was used for the implmentation of the trained model into an API.
* The API also included additional features such as retrieving the nearest amenities to the input address.
* A user interface was also created to make the API more user-friendly.
* The API was then deployed onto Render for use (https://hdb-resale-prices-calculator.onrender.com/).

<a name="File_Descriptions"></a>
## File Descriptions

* <strong>[ data ](https://github.com/cheeweisoh/portfolio/tree/main/hdb_resale_prices/data)</strong> : data files used/generated
    * <strong> housing_cpi.csv </strong>: list of price indices for resale flats from 2017 to 2021, obtained from Singapore Department of Statistics
    * <strong> malls.csv </strong>: list of shopping malls in Singapore, obtained from Wikipedia
    * <strong> malls_coord.csv </strong>: list of coordinates for all shopping malls from malls.csv
    * <strong> mrt_coord.csv </strong>: list of coordinates for all MRT/LRT stations from mrt_stations.csv
    * <strong> mrt_stations.csv </strong>: list of MRT/LRT stations in Singapore, obtained from Land Transport Authority website
    * <strong> pri_sch_coord.csv </strong>: list of coordinates for all primary schools from pri_sch_ranking.csv
    * <strong> pri_sch_ranking.csv </strong>: list of primary schools and rankings based on popularity in 2021
    * <strong> resale_df_geocode.csv </strong>: list of resale flats with coordinates
    * <strong> resale_prices.csv </strong>: dataset with added features
    * <strong> resale_prices_final.csv </strong>: final dataset for model building
* <strong>[ resale_prices_deployment ](https://github.com/cheeweisoh/portfolio/tree/main/hdb_resale_prices/deployment)</strong> : deployment files to FastAPI
    * <strong> models </strong>: model/preprocessing files
        * <strong> house.py </strong>: House object for initiating input
        * <strong> malls_coord.csv </strong>: list of coordinates for all malls
        * <strong> mrt_coord.csv </strong>: list of coordinates for all MRT/LRT stations
        * <strong> pri_school_coord.csv </strong>: list of coordinates for all primary schools
        * <strong> sec_school_coord.csv </strong>: list of coordinates for all secondary schools
    * <strong> templates </strong>: HTML templates for user interface
        * <strong>input.html</strong>: Input form for adding details about resale flat
        * <strong>layout.html</strong>: General layout for all pages
        * <strong>output.html</strong>: Output page for displaying resale price and other features
    * <strong> main.py </strong>: main file for FastAPI deployment
    * <strong> Procfile </strong>: procfile for deployment on Heroku
    * <strong> requirements.txt </strong>: Package requirements for deployment on Heroku
    * <strong> xgb_best.sav </strong>: pickle file for XGBoost pipeline
* <strong>Housing Price Dataset.ipynb</strong>: retrieving dataset from source, and including new features for distance to amenities
* <strong>Housing Price Analysis.ipynb</strong>: exploratory data analysis of resale price dataset
* <strong>Housing Prices Model.ipynb</strong>: building supervised learning models to predict resale price
    
<a name="Credits"></a>
## Credits

* <strong>[ data.gov.sg ](https://data.gov.sg/)</strong>: Datasets for HDB resale prices and list of schools in Singapore
* <strong>[ Singstat Table Builder ](https://www.tablebuilder.singstat.gov.sg/publicfacing/mainMenu.action)</strong>: Data for Price Index for resale flats in Singapore
* <strong>[ OneMap API ](https://www.onemap.gov.sg/docs/#onemap-rest-apis)</strong>: API used for geocoding addresses and amenities
