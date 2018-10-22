# Predict future sales

This project is a forecasting system for timeseries.

For the prediction, it uses [spark-ts](https://github.com/sryza/spark-timeseries), 
[ts-flint](https://github.com/twosigma/flint) and [pyramid-arima](https://github.com/tgsmith61591/pyramid) 
and then it compares their R-squared score. 

## Requirements
- Python 3.6 

Python libraries (installed via ```pip3```):
- pyspark 
- sparkts
- ts-flint
- numpy
- pandas
- sklearn
- pyramid.arima
- matplotlib

## How to run
1. Clone the repository:
```
git clone https://github.com/murruFlavia/ARE2project.git
```
2. Run (in top-level directory):
```
python3 main.py
```
