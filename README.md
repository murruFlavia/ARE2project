# Predict future sales

This project is a forecasting system for timeseries.

For the prediction, it uses [spark-ts](https://github.com/sryza/spark-timeseries), 
[ts-flint](https://github.com/twosigma/flint) and [pyramid-arima](https://github.com/tgsmith61591/pyramid) 
and then it compares their R-squared score. 

## Requirements
- Ubuntu 18.04
- Python 3.6 with pip:

```sudo apt-get update
   sudo apt-get install python3.6  
   sudo apt install python3-pip
```

Python libraries: 
- pyspark: ```pip3 install pyspark```
- sparkts: ```pip3 install sparkts```
- ts-flint (see the [installation guide](https://github.com/twosigma/flint/blob/master/python/README.md))
- numpy: ```pip3 install numpy```
- pandas: ```pip3 install pandas```
- sklearn: ```pip3 install sklearn```
- pyramid-arima: ```pip3 install pyramid-arima```
- matplotlib: ```pip3 install matplotlib```

## How to run
1. Clone the repository:
```
git clone https://github.com/murruFlavia/ARE2project.git
```
2. Run (in top-level directory):
```
python3 main.py
```
