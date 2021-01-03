import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import itertools
import warnings
import os


import statsmodels.api as sm
import statsmodels.graphics.tsaplots as sgt 
import statsmodels.tsa.stattools as sts 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from scipy.stats.distributions import chi2


org_path = "/home/lsm/Desktop/Prac/Blog_/Time series"
os.chdir(org_path)
sns.set()
warnings.filterwarnings("ignore")

## LLR test
def LLR_test(mod_1, mod_2, DF=1):
    L1 = mod_1.llf
    L2 = mod_2.llf
    LR = (2*(L2-L1))
    p = chi2.sf(LR, DF).round(3)
    return p


## Data Loading
snp= pd.read_csv("snp_index.csv")
nikkei = pd.read_csv("N225.csv")
print("snp:", len(snp),"nikkei:", len(nikkei))

## Changing type of Date column
snp.Date = pd.to_datetime(snp.Date, dayfirst = False)
nikkei.Date = pd.to_datetime(nikkei.Date, dayfirst = False)

## Setting the start date and end date  
snp = snp.loc[snp["Date"] != "1994-01-03"]
nikkei=nikkei.loc[nikkei["Date"] < "2020-01-01"]

## Setting index to make two dataframe with same length 
snp.set_index("Date", inplace = True)
snp=snp.asfreq(freq="B")

nikkei.set_index("Date", inplace = True)
nikkei=nikkei.asfreq(freq="B")

print("snp:", len(snp),"nikkei:", len(nikkei))

## Merge 
df = pd.merge(snp, nikkei, on="Date")
# df.set_index("Date", inplace = True)
len(df)
df=df.fillna(method='ffill')
df = df[["Close_x", "Close_y"]]
df.columns = ["SNP", "Nikkei"]

## diff
df["diff_SNP"]=df["SNP"].diff()
df["diff_Nikkei"]=df["Nikkei"].diff()

## return
df["return_snp"]=df["SNP"].pct_change(1)*100
df["return_nikkei"]=df["Nikkei"].pct_change(1)*100


## Train, test
size = int(len(df)*0.8)
df_train = df.iloc[:size]
df_test = df.iloc[size:]


##Decompose

decomposition =seasonal_decompose(df_train["return_nikkei"][1:], period =1)
fig = decomposition.plot()
fig.set_size_inches(10,10)
plt.show()

## ACF, PACF
fig = plt.figure()
ax1 = fig.add_subplot(2, 1, 1)
ax2 = fig.add_subplot(2, 1, 2)

sgt.plot_acf(df_train["return_nikkei"][1:], lags = 20, zero = False, ax=ax1)
ax1.set_title("ACF Nikkei") ##MA2


sgt.plot_pacf(df_train["return_nikkei"][1:], lags = 20, zero = False, method = ('ols'), ax=ax2)
ax2.set_title("PACF Nikkei") ##AR6

fig.set_size_inches(10,10)
plt.show()
## ADF
sts.adfuller(df_train["return_nikkei"][1:])

## selecting lag

p = range(0,7)
d = range(0,1)
q = range(0,7)

pdq = list(itertools.product(p,d,q))

dict_model ={}
for i in pdq:
    
    try:
        model = ARIMA(df["return_nikkei"][1:], order=(i))
        print(i)
        model_fit = model.fit( maxiter = 100)
        dict_model[i]=[model_fit.llf, model_fit.aic]
        
    except:
        print("error lag:",i)

information=pd.DataFrame.from_dict(dict_model, orient ="index", columns =["llf", "AIC"])
information.loc[information["llf"] == information["llf"].max()]
information.loc[information["AIC"] == information["AIC"].min()]

##(6,0,4)
model_ar_6_ma_4 = ARIMA(df_train["return_nikkei"][1:], order=(6,0,4))
results_ar_6_ma_4 = model_ar_6_ma_4.fit()
results_ar_6_ma_4.summary()

##(6,0,6)
model_ar_6_ma_6 = ARIMA(df_train["return_nikkei"][1:], order=(6,0,6))
results_ar_6_ma_6 = model_ar_6_ma_6.fit()
results_ar_6_ma_6.summary()

LLR_test(results_ar_6_ma_4,results_ar_6_ma_6, DF=2) 

## ARIMA
model_ar_1_ma_1 = ARIMA(df_train["return_nikkei"][1:], order=(1,0,1))
results_ar_1_ma_1 = model_ar_1_ma_1.fit()
results_ar_1_ma_1.summary()

## ARIMAX
model_ar_1_ma_1_X = ARIMA(df_train["return_nikkei"][1:], exog = df_train["return_snp"][1:],order=(1,0,1))
results_ar_1_ma_1_X = model_ar_1_ma_1_X.fit()
results_ar_1_ma_1_X.summary()

## SARIMA & SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

model_sarima = SARIMAX(df_train["return_nikkei"][1:], order=(1,0,1), seasonal_order = (1,0,1,10),initialization='approximate_diffuse')
results_sarima = model_sarima.fit()
results_sarima.summary()

# sgt.plot_acf(results_sarima.resid, zero = False, lags = 50)
# plt.title("ACF Of Residuals for SARIMA", size=50)
# plt.show()

# sm.stats.acorr_ljungbox(results_sarima.resid, lags=[20], return_df=True)

model_sarimax = SARIMAX(df_train["return_nikkei"][1:], exog = df_train["return_snp"][1:], order=(1,0,1), seasonal_order = (1,0,1,10),initialization='approximate_diffuse')
results_sarimax = model_sarimax.fit()
results_sarimax.summary()

# sgt.plot_acf(results_sarimax.resid, zero = False, lags = 20)
# plt.title("ACF Of Residuals for SARIMAX", size=20)
# plt.show()

# sm.stats.acorr_ljungbox(results_sarimax.resid, lags=[20], return_df=True)

# ## Preidcting
start_date = '1994-01-05'
end_date = '2014-10-16'


df_train["prediction_arima"] = results_ar_1_ma_1.predict(start = start_date, end = end_date)
df_train["prediction_arimax"] = results_ar_1_ma_1_X.predict(start = start_date, end = end_date, exog = df_train["return_snp"])
df_train["prediction_sarima"] = results_sarima.predict(start = start_date, end = end_date)
df_train["prediction_sarimax"] = results_sarimax.predict(start = start_date, end = end_date , exog = df_train["return_snp"])

fig= plt.figure(figsize = (20,20))

plt.plot(df_train["return_nikkei"][:30], "b-", label = "Actual", alpha = 0.5)
plt.plot(df_train["prediction_arima"][:30] , "r-", label = "ARIMA" , alpha = 0.5)
plt.plot(df_train["prediction_arimax"][:30] , "g-" , label = "ARIMAX" , alpha = 0.5) 
plt.plot(df_train["prediction_sarima"][:30] , "y-" , label = "SARIMA" , alpha = 0.5) 
plt.plot(df_train["prediction_sarimax"][:30] , "k-" , label = "SARIMAX" , alpha = 0.5) 
plt.legend()

plt.show()


## Forecasting
start_date = '2014-10-16'
end_date = '2019-12-30'
len(df_train)+len(df_test)
len(results_sarima.predict(start = 5236, end = 6544))
results_sarima.forecast(1309)

df_test["prediction_arima"] = results_ar_1_ma_1.forecast(1309)[0]
df_test["prediction_arimax"] = results_ar_1_ma_1_X.forecast(1309, exog = df_test["return_snp"])[0] 
df_test["prediction_sarima"] = results_sarima.forecast(1309).values
df_test["prediction_sarimax"] = results_sarimax.forecast(1309, exog = df_test[["return_snp"]]).values


fig= plt.figure(figsize = (20,20))
plt.plot(df_test["return_nikkei"][:30], "b-", label = "Actual", alpha = 0.5)
plt.plot(df_test["prediction_arima"][:30] , "r-", label = "ARIMA" , alpha = 0.5)
plt.plot(df_test["prediction_arimax"][:30] , "g-" , label = "ARIMAX" , alpha = 0.5) 
plt.plot(df_test["prediction_sarima"][:30] , "y-" , label = "SARIMA" , alpha = 0.5) 
plt.plot(df_test["prediction_sarimax"][:30] , "k-" , label = "SARIMAX" , alpha = 0.5) 
plt.legend()

plt.show()

def rmse_(real, pred):
    diff = real-pred
    diff2=np.sqrt(diff**2)
    return diff2.mean()

rmse_(df_test["return_nikkei"],df_test["prediction_arima"])
rmse_(df_test["return_nikkei"],df_test["prediction_arimax"])
rmse_(df_test["return_nikkei"],df_test["prediction_sarima"])
rmse_(df_test["return_nikkei"],df_test["prediction_sarimax"])


# import pmdarima as pm

# model_auto = pm.auto_arima()

# model_auto.summary()
# model_auto.predict(n_periods =1356, exogenous=df_test[["return_snp"]])
# len(df_test[["return_snp"]])
