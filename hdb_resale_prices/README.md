<h1 align='center'>HDB Resale Prices</h1>

## Project Introduction
Public housing in Singapore is managed by the Housing and Development Board. The aim of public housing is to provide affordable housing options to Singaporeans despite domestic land constraints. This is achieved through high-rise housing structures and grants that make the costs of housing affordable for the average Singaporean. In the recent years, there have been an increasing number of HDB flats making headlines for record-breaking transaction prices.

In this project, we explore the factors that might affect HDB resale prices in Singapore. Since majority of Singaporeeans own HDB units, this could give them an indication as to the current prices of their property. In this part, we will work on publicly available datasets to generate a set of possible factors that might affect resale prices. In the next part, we will make use of data analytics to examine the effect of housing prices on these factors, and use models to predict the resale prices of HDB flats in Singapore.

## Current Status
* [Completed] Data Collection
* [Completed] Exploratory Data Analysis
* [In Progress] Model Building

## Table of Contents

1. [ Methods/Technologies Used ](#Methods_Techonologies_Used)
2. [ Project Description ](#Project_Description)
3. [ File Descriptions ](#File_Descriptions)
4. [ Credits ](#Credits)

<a name="Methods_Techonologies_Used"></a>
## Methods/Technologies Used

### Technologies
* Python
* Pandas
* Numpy
* Matplotlib

<a name="Project_Description"></a>
## Project Description

### Housing Price Dataset
* A list of 108,048 HDB resale transactions from 2017 to 2021 was extracted from data.gov.sg.
* Using the OneMap API, the addresses of these HDB resale flats were geocoded. A list of MRTs/LRTs, primary/secondary schools, and shopping malls in Singapore was also geocoded. These coordinates were then used to determine and calculate the distance of the nearest amenities to each resale flat.
* To remove flunctuations in housing prices through inflation, the Price Index for resale flats was used to adjust housing resale prices to 2021 Q2.

<a name="File_Descriptions"></a>
## File Descriptions

* <strong>[ data ](https://github.com/cheeweisoh/portfolio/tree/main/hdb_resale_prices/data)</strong> : folder containing all data files used/generated
    * <strong> coord_data.csv </strong>: list of unique addresses and the corresponding latitudes and longitudes
    * <strong> housing_cpi.csv </strong>: list of price indices for resale flats from 2017 to 2021, obtained from Singapore Department of Statistics
    * <strong> malls.csv </strong>: list of shopping malls in Singapore, obtained from Wikipedia
    * <strong> mrt_stations.csv </strong>: list of MRT/LRT stations in Singapore, obtained from Land Transport Authority website
    * <strong> nearest_mall.csv </strong>: distance from resale flat to nearest shopping mall for each unique address
    * <strong> nearest_mrt.csv </strong>: distance from resale flat to nearest MRT/LRT station for each unique address
    * <strong> nearest_pri_sch.csv </strong>: distance from resale flat to nearest primary school for each unique address
    * <strong> nearest_sec_school.csv </strong>: distance from resale flat to nearest secondary school for each unique address
    * <strong> resale_prices.csv <strong>: final dataset with added features
    
<a name="Credits"></a>
## Credits

* <strong>[ data.gov.sg ](https://data.gov.sg/)</strong>: Datasets for HDB resale prices and list of schools in Singapore
* <strong>[ Singstat Table Builder ](https://www.tablebuilder.singstat.gov.sg/publicfacing/mainMenu.action)</strong>: Data for Price Index for resale flats in Singapore
* <strong>[ OneMap API ](https://www.onemap.gov.sg/docs/#onemap-rest-apis)</strong>: API used for geocoding addresses and amenities
