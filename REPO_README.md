# Repository Description

## Repository Structure

| dataset  
&emsp;| econ_data  
&emsp;&emsp;| ...  
&emsp;| stock_data  
&emsp;&emsp;| ...  
| environment  
&emsp;| ...   
| images  
&emsp;| ...   
| src  
&emsp;| myarima.py   
| test  
&emsp;| ...   
| portfolo_optimization.ipynb  
| README.md  

## Structure Information

**/dataset**

This directory holds most of our data. The *econ_data* subfolder includes
information about economic indicators that are inputs to the clustering models.
The *stock_data* subfolder includes the ticker data for 10 stocks since 2007. 

**/environment**

Similar to the provided environment provided by GT CS7641, but the files are
modified to include the necessary packages for the repository.

**/images**

Directory to hold relevant images for the report.

**/src**

Directory to hold relevant model files for the report.

- **/src/myarima.py**
    
The current implementation of the ARIMA model. 

**/test**

This directory will hold unit tests for any model implementations. We currently
do not have any unit tests.  

**portfolo_optimization.ipynb**

Main jupyter notebook that executes the different models to determine how to
best optimize a given portfolio of 10 stocks.


**README.md**

This file hold our project report and acts as the source content for the Github page. 

