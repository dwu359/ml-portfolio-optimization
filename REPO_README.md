# Repository Description

## Repository Structure

| dataset  
&emsp;| econ_data  
&emsp;| stock_data   
| environment   
| images  
| src  
&emsp;| init_dataset.py  
&emsp;| myoptimizer.py 
&emsp;| myarima.py 
&emsp;| mykmeans.py  
| test  
| portfolo_optimization.ipynb  
| README.md  

## Structure Information

**/dataset**

This directory holds all of our data. The *econ_data* subfolder includes
information about economic indicators that are inputs to the clustering models.
The *stock_data* subfolder includes the ticker data for 10 stocks since 2007. 

**/environment**

Similar to the provided environment provided by GT CS7641, but the files are
modified to include the necessary packages for the repository.

**/images**

Directory to hold relevant images for the report.

**/src**

Directory to hold relevant model files for the report. These files should
contain most of the business logic for the ML models, while the jupyter notebook
is mainly used for quick development and visualizations. 

- **/src/init_dataset.py**
    
Logic used to initialize and conglomerate the large datasets such as
`dataset/stock_data/combined_stock_adj_closed`,
`dataset/combined_stock_data.csv`, `dataset/econ_data/combined_econ_data.csv`. 

- **/src/myoptimizer.py**
    
Baseline model that will be used to determine stock allocation that optimizes
the maximum returns from the portfolio. This model utilizes allocation from the
other models as input. 

- **/src/myarima.py**
    
The current implementation of the ARIMA model. 

- **/src/mykmeans.py**
    
Kmeans clustering model that attempts to cluster stock data across macroeconomic
indicators such as GDP, Interest Rates, etc. 

**/test**

This directory will hold unit tests for any model implementations. We currently
do not have any unit tests.  

**portfolo_optimization.ipynb**

Main jupyter notebook that executes the different models to determine how to
best optimize a given portfolio of 10 stocks.


**README.md**

This file hold our project report and acts as the source content for the Github page. 

