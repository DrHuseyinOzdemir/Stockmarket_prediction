# Ensemble of Time Series Models  
# 1. ARIMA (AR+I+MA models) 
# 2. Neural Network 
# 3. Smooting Method ETS
# 4. Random- walk on Hadoop


#Autoregressive Models (AR), Moving Average Models (MA) & Seasonal Regression Models
#Distributed Lags Models (I) & Neural Net Models & ETS Models

#The ARIMA model combines three basic methods:

#AutoRegression (AR) – In auto-regression the values of a given time series data are
#                      regressed on their own lagged values, which is indicated by the
#                      “p” value in the model.
#Differencing (I-for Integrated) – This involves differencing the time series data to
#                      remove the trend and convert a non-stationary time series to a
#                      stationary one. This is indicated by the “d” value in the model.
#                      If d = 1, it looks at the difference between two time series 
#                      entries, if d = 2 it looks at the differences of the differences
#                      obtained at d =1, and so forth.
#Moving Average (MA) – The moving average nature of the model is represented by the “q”
#                      value which is the number of lagged values of the error term.

library(quantmod)
library(tseries)
library(ggplot2)
library(timeSeries)
library(forecast)
library(xts)
library(timeDate)
library(data.table)
library(tidyverse)
library(lubridate)
library(formattable)
library(MLmetrics)
library(prophet)
library(tsfknn)

stock_name <- 'AMZN' # Input variable: Get stock name
analysis_date <- 5*365 # Input variable: This indicates the number of days to determine the length of analysis period 
forecast_day <- 30 # Input variable: Number of days for forecast
test_length <- 100  # Input variable: This determines how many days are used for backtest 

end_date <- Sys.Date()
start_date <- end_date - analysis_date

MyZone = "UTC"
Sys.setenv(TZ = MyZone)

DayCal = timeSequence(from = end_date,  to = end_date + 2*forecast_day, by="day", zone = MyZone)

TradingCal = DayCal[isBizday(DayCal, holidayNYSE())]
forecast_Dates <- as.Date(TradingCal)
forecast_Dates <- forecast_Dates[1:forecast_day]


# Pull data from Yahoo finance 
stock_allprice = getSymbols(stock_name, from=start_date, to=end_date, auto.assign = FALSE)
stock_allprice = na.omit(stock_allprice)

# Select the relevant close price series
stock_prices = stock_allprice[,4]
names(stock_prices) <- "Close"
#Autoplot


# Arima Forecast of closing price of next N days

arima <- forecast(auto.arima(stock_prices),h=forecast_day, level=95)


mean_arima <- round(as.vector(arima$mean[1:forecast_day]),2) 
lower_arima <- round(as.vector(arima$lower[1:forecast_day]),2)  
upper_arima <- round(as.vector(arima$upper[1:forecast_day]),2)  


sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)

arima_mean_value <- c(rep(NA,length(stock_prices)), mean_arima)
arima_lower_value <- c(rep(NA,length(stock_prices)), lower_arima)
arima_upper_value <- c(rep(NA,length(stock_prices)), upper_arima)

sample_value <- c(coredata(stock_prices),rep(NA,forecast_day))

arima_data <- data.frame(all_date,sample_value, arima_mean_value,arima_lower_value,arima_upper_value)

### Arima Plot ####

fig1 <- ggplot(arima_data,aes(x=all_date))+
        geom_line(aes(y=sample_value), size=1, color="steelblue")+  
        geom_line(aes(y=arima_mean_value), size=0.8, color="red") +
        geom_ribbon(aes(ymin=arima_lower_value, ymax=arima_upper_value), fill="firebrick", linetype=2, alpha=0.3)+
        labs(title = paste("The daily price forecast of", stock_name, 
                     "for next",forecast_day,"days"),
       subtitle = paste("Method:",arima$method),
       x = "",
       y = "")+
       theme_minimal()

fig1


### Arima Table ####


tab1 <- formattable(data.table(Date = forecast_Dates,
                               Forecast = mean_arima,
                               "Lower band"=lower_arima,
                               "Upper band"=upper_arima), align=c("l","c","c","c"),
                    list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
                    ))

tab1


######################## Neural Network Forecast ########################

#Neural Net prediction for closing price of next N days

#nn_method <- forecast(nnetar(stock_prices,p=arimaorder(auto.arima(stock_prices))[1]),h=forecast_day)

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)


dnn <- forecast(nnetar(stock_prices,lambda= "auto"),h=forecast_day, PI=TRUE, level= 90,)

mean_dnn <- round(as.vector(dnn$mean[1:forecast_day]),2) 
lower_dnn <- round(as.vector(dnn$lower[1:forecast_day]),2)  
upper_dnn <- round(as.vector(dnn$upper[1:forecast_day]),2) 

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)

dnn_mean_value <- c(rep(NA,length(stock_prices)), mean_dnn)
dnn_lower_value <- c(rep(NA,length(stock_prices)), lower_dnn)
dnn_upper_value <- c(rep(NA,length(stock_prices)), upper_dnn)

sample_value <- c(coredata(stock_prices),rep(NA,forecast_day))

dnn_data <- data.frame(all_date,sample_value, dnn_mean_value,dnn_lower_value,dnn_upper_value)


fig2 <- ggplot(dnn_data,aes(x=all_date))+
  geom_line(aes(y=sample_value), size=1, color="steelblue")+  
  geom_line(aes(y=dnn_mean_value), size=0.8, color="red") +
  geom_ribbon(aes(ymin=dnn_lower_value, ymax=dnn_upper_value), fill="firebrick", linetype=2, alpha=0.3)+
  labs(title = paste("The daily price forecast of", stock_name, 
                     "for next",forecast_day,"days"),
       subtitle = paste("Method:",dnn$method),
       x = "",
       y = "")+
  theme_minimal()

fig2


### Neural network table ####


tab2 <- formattable(data.table(Date = forecast_Dates,
                               Forecast = mean_dnn,
                               "Lower band"=lower_dnn,
                               "Upper band"=upper_dnn), align=c("l","c","c","c"),
                    list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
                    ))

tab2


######################## ETS Smoothing model MAN ########################

man <- forecast(ets(model="MAN",stock_prices),h=forecast_day,level=95)

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)


mean_man <- round(as.vector(man$mean[1:forecast_day]),2) 
lower_man <- round(as.vector(man$lower[1:forecast_day]),2)  
upper_man <- round(as.vector(man$upper[1:forecast_day]),2) 

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)

man_mean_value <- c(rep(NA,length(stock_prices)), mean_man)
man_lower_value <- c(rep(NA,length(stock_prices)), lower_man)
man_upper_value <- c(rep(NA,length(stock_prices)), upper_man)

sample_value <- c(coredata(stock_prices),rep(NA,forecast_day))

man_data <- data.frame(all_date,sample_value, man_mean_value,man_lower_value,man_upper_value)

### man Plot ####

fig3 <- ggplot(man_data,aes(x=all_date))+
  geom_line(aes(y=sample_value), size=1, color="steelblue")+  
  geom_line(aes(y=man_mean_value), size=0.8, color="red") +
  geom_ribbon(aes(ymin=man_lower_value, ymax=man_upper_value), fill="firebrick", linetype=2, alpha=0.3)+
  labs(title = paste("The daily price forecast of", stock_name, 
                     "for next",forecast_day,"days"),
       subtitle = paste("Method:",man$method),
       x = "",
       y = "")+
  theme_minimal()

fig3


### Man Table ####


tab3 <- formattable(data.table(Date = forecast_Dates,
                               Forecast = mean_man,
                               "Lower band"=lower_man,
                               "Upper band"=upper_man), align=c("l","c","c","c"),
                    list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
                    ))

tab3



######################## ETS Smoothing model AAN ########################

aan <- forecast(ets(model="AAN",stock_prices),h=forecast_day,level=95)

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)

mean_aan <- round(as.vector(aan$mean[1:forecast_day]),2) 
lower_aan <- round(as.vector(aan$lower[1:forecast_day]),2)  
upper_aan <- round(as.vector(aan$upper[1:forecast_day]),2) 

aan_mean_value <- c(rep(NA,length(stock_prices)), mean_aan)
aan_lower_value <- c(rep(NA,length(stock_prices)), lower_aan)
aan_upper_value <- c(rep(NA,length(stock_prices)), upper_aan)

sample_value <- c(coredata(stock_prices),rep(NA,forecast_day))

aan_data <- data.frame(all_date,sample_value, aan_mean_value,aan_lower_value,aan_upper_value)

### AAN Plot ####

fig4 <- ggplot(aan_data,aes(x=all_date))+
  geom_line(aes(y=sample_value), size=1, color="steelblue")+  
  geom_line(aes(y=aan_mean_value), size=0.8, color="red") +
  geom_ribbon(aes(ymin=aan_lower_value, ymax=aan_upper_value), fill="firebrick", linetype=2, alpha=0.3)+
  labs(title = paste("The daily price forecast of", stock_name, 
                     "for next",forecast_day,"days"),
       subtitle = paste("Method:",aan$method),
       x = "",
       y = "")+
  theme_minimal()

fig4


### AAN Table ####


tab4 <- formattable(data.table(Date = forecast_Dates,
                               Forecast = mean_aan,
                               "Lower band"=lower_aan,
                               "Upper band"=upper_aan), align=c("l","c","c","c"),
                    list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
                    ))

tab4



######################## Random walk with drift model ####################################################

cp_rwf <- rwf((stock_prices),h=forecast_day,level=95)

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)

mean_rwf <- round(as.vector(cp_rwf$mean[1:forecast_day]),2) 
lower_rwf <- round(as.vector(cp_rwf$lower[1:forecast_day]),2)  
upper_rwf <- round(as.vector(cp_rwf$upper[1:forecast_day]),2) 

rwf_mean_value <- c(rep(NA,length(stock_prices)), mean_rwf)
rwf_lower_value <- c(rep(NA,length(stock_prices)), lower_rwf)
rwf_upper_value <- c(rep(NA,length(stock_prices)), upper_rwf)

sample_value <- c(coredata(stock_prices),rep(NA,forecast_day))

rwf_data <- data.frame(all_date,sample_value, rwf_mean_value,rwf_lower_value,rwf_upper_value)

### rwf Plot ####

fig5 <- ggplot(rwf_data,aes(x=all_date))+
  geom_line(aes(y=sample_value), size=1, color="steelblue")+  
  geom_line(aes(y=rwf_mean_value), size=0.8, color="red") +
  geom_ribbon(aes(ymin=rwf_lower_value, ymax=rwf_upper_value), fill="firebrick", linetype=2, alpha=0.3)+
  labs(title = paste("The daily price forecast of", stock_name, 
                     "for next",forecast_day,"days"),
       subtitle = paste("Method:",cp_rwf$method),
       x = "",
       y = "")+
  theme_minimal()



### rwf Table ####

tab5 <- formattable(data.table(Date = forecast_Dates,
                        Forecast = mean_rwf,
                        "Lower band"=lower_rwf,
                        "Upper band"=upper_rwf), align=c("l","c","c","c"),
                    list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
                    ))

tab5

#### Facebook Prophet Model Prediction and Daily Future Forecast 


prodf <- data.frame(ds = index(stock_prices),
                 y = as.numeric(stock_prices))


prophetpred <- prophet(prodf,daily.seasonality=TRUE)


future <- make_future_dataframe(prophetpred, periods = forecast_day)
forecastprophet <- predict(prophetpred, future)


yhat <- forecastprophet$yhat
yhat_lower <- forecastprophet$yhat_lower
yhat_upper <- forecastprophet$yhat_upper
sample_value <- c(coredata(stock_prices),rep(NA,forecast_day))

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)


prophet_plotdata <- data.frame(all_date,sample_value,yhat,
                               yhat_lower, yhat_upper)

fig6 <- ggplot(prophet_plotdata,aes(x=all_date))+
        geom_point(aes(y=sample_value), size=1, color="darkgray")+  
        geom_line(aes(y=yhat), size=0.8, color="red") +
        geom_ribbon(aes(ymin=yhat_lower, ymax=yhat_upper), fill="firebrick", linetype=2, alpha=0.3)+
        labs(title = paste("The daily price forecast of", stock_name, "for next",forecast_day,"days"),
        subtitle = paste("Method: Prophet"),
        x = "",
        y = "")+
        theme_minimal()
  

fig6

### Prophet Table ####


tab6 <- formattable(data.table(Date = forecast_Dates,
                               Forecast = yhat[(length(stock_prices)+1):(length(stock_prices)+forecast_day)],
                               "Lower band"=yhat_lower[(length(stock_prices)+1):(length(stock_prices)+forecast_day)],
                               "Upper band"=yhat_upper[(length(stock_prices)+1):(length(stock_prices)+forecast_day)]), align=c("l","c","c","c"),
                    list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
                    ))

tab6


# KNN Forecast of closing price of next N days

knndf <- data.frame(ds = index(stock_prices),
                    y = as.numeric(stock_prices))

predknn <- knn_forecasting(knndf$y, h = forecast_day, k=c(3,5,7))

sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)

mean_knn <- round(coredata(predknn$prediction),2)

knn_mean_value <- c(rep(NA,length(stock_prices)),mean_knn)

sample_value <- c(coredata(stock_prices),rep(NA,forecast_day))

knn_data <- data.frame(all_date,sample_value, knn_mean_value)

### KNN Plot ####

fig7 <- ggplot(knn_data,aes(x=all_date))+
  geom_line(aes(y=sample_value), size=1, color="steelblue")+  
  geom_line(aes(y=knn_mean_value), size=0.8, color="red") +
  labs(title = paste("The daily price forecast of", stock_name, 
                     "for next",forecast_day,"days"),
       subtitle = paste("Method: K-nearest neighbor"),
       x = "",
       y = "")+
  theme_minimal()

fig7

### KNN Table ####


tab7 <- formattable(data.table(Date = forecast_Dates,
                               Forecast = mean_knn), align=c("l","l"),
                    list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
                    ))

tab7



## Save Plot 

pdf('forecast_plot.pdf',height = 6, width = 8)
plot(fig1)
plot(fig2)
plot(fig3) 
plot(fig4)
plot(fig5)
plot(fig6)
plot(fig7)
dev.off()



############# Arima Forecast accuracy rate and backtest table #####################################################

k <- length(stock_prices)
test <- NULL
forecast_return_direction <- NULL

for (i in 1:test_length) {
  
  train_f <- forecast(auto.arima(stock_prices[i:(k-test_length-1+i)]), h=1) # Model predicts price for next day
  test[i] <- round(train_f$mean,2)
  
}

compare_test <- round(stock_prices[(k-test_length+1):k],2)
compare_test$Forecast <- test

# Compute the log returns for the stock
stock_return <- diff(log(stock_prices),lag=1)
stock_return <- stock_return[!is.na(stock_return)]
stock_return <- coredata(stock_return)

actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
actual_return_direction <- as.vector(actual_return_direction[(k-test_length):(k-1)])

for (i in 1:test_length) {
  
  forecast_return_direction[i] <- ifelse(compare_test$Forecast[[i]] > stock_prices[[k-test_length-1+i]], 'Up','Down')
  
}

forcast_table <- as.data.frame(compare_test)
forcast_table$actual_rd <- actual_return_direction
forcast_table$forecast_rd <- forecast_return_direction

names(forcast_table) <- c("Close price","Forecast price","Actual return","Forecast return")

accuracy_rate <- ifelse(forcast_table$`Actual return` == forcast_table$`Forecast return`, 1, 0)

formattable(forcast_table, align="l")

accuracy_arima <- print(paste0(round(sum(accuracy_rate)/test_length*100,2)," %"))

arima_table <- c(round(MAPE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                 round(MAE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                 round(RMSE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                 accuracy_arima)

############# Neural network Forecast accuracy rate and backtest table #####################################################

k <- length(stock_prices)
test <- NULL
forecast_return_direction <- NULL


for (i in 1:test_length) {
  
  train_f <- forecast(nnetar(stock_prices[i:(k-test_length-1+i)],lambda="auto"),h=1) # Model predicts price for next day
  test[i] <- round(train_f$mean,2)
  
}


compare_test <- round(stock_prices[(k-test_length+1):k],2)
compare_test$Forecast <- test

# Compute the log returns for the stock
stock_return <- diff(log(stock_prices),lag=1)
stock_return <- stock_return[!is.na(stock_return)]
stock_return <- coredata(stock_return)

actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
actual_return_direction <- as.vector(actual_return_direction[(k-test_length):(k-1)])

for (i in 1:test_length) {
  
  forecast_return_direction[i] <- ifelse(compare_test$Forecast[[i]] > stock_prices[[k-test_length-1+i]], 'Up','Down')
  
}

forcast_table <- as.data.frame(compare_test)
forcast_table$actual_rd <- actual_return_direction
forcast_table$forecast_rd <- forecast_return_direction

names(forcast_table) <- c("Close price","Forecast price","Actual return","Forecast return")

accuracy_rate <- ifelse(forcast_table$`Actual return` == forcast_table$`Forecast return`, 1, 0)

formattable(forcast_table, align="l")
accuracy_nnar <- print(paste0(round(sum(accuracy_rate)/test_length*100,2)," %"))

nnar_table <- c(round(MAPE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                round(MAE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                round(RMSE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                accuracy_nnar)

############# ETS Smoothing MAN Forecast accuracy rate and backtest table #####################################################


k <- length(stock_prices)
test <- NULL
forecast_return_direction <- NULL

for (i in 1:test_length) {
  
  train_f <- forecast(ets(model="MAN",stock_prices[i:(k-test_length-1+i)]),h=1) # Model predicts price for next day
  test[i] <- round(train_f$mean,2)
  
}

compare_test <- round(stock_prices[(k-test_length+1):k],2)
compare_test$Forecast <- test

# Compute the log returns for the stock
stock_return <- diff(log(stock_prices),lag=1)
stock_return <- stock_return[!is.na(stock_return)]
stock_return <- coredata(stock_return)

actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
actual_return_direction <- as.vector(actual_return_direction[(k-test_length):(k-1)])

for (i in 1:test_length) {
  
  forecast_return_direction[i] <- ifelse(compare_test$Forecast[[i]] > stock_prices[[k-test_length-1+i]], 'Up','Down')
  
}

forcast_table <- as.data.frame(compare_test)
forcast_table$actual_rd <- actual_return_direction
forcast_table$forecast_rd <- forecast_return_direction

names(forcast_table) <- c("Close price","Forecast price","Actual return","Forecast return")

accuracy_rate <- ifelse(forcast_table$`Actual return` == forcast_table$`Forecast return`, 1, 0)

formattable(forcast_table, align="l")

accuracy_man <- print(paste0(round(sum(accuracy_rate)/test_length*100,2)," %"))

man_table <- c(round(MAPE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(MAE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(RMSE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               accuracy_man)

############# ETS Smoothing AAN Forecast accuracy rate and backtest table #####################################################


k <- length(stock_prices)
test <- NULL
forecast_return_direction <- NULL

for (i in 1:test_length) {
  
  train_f <- forecast(ets(model="AAN",stock_prices[i:(k-test_length-1+i)]),h=1) # Model predicts price for next day
  test[i] <- round(train_f$mean,2)
  
}

compare_test <- round(stock_prices[(k-test_length+1):k],2)
compare_test$Forecast <- test

# Compute the log returns for the stock
stock_return <- diff(log(stock_prices),lag=1)
stock_return <- stock_return[!is.na(stock_return)]
stock_return <- coredata(stock_return)

actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
actual_return_direction <- as.vector(actual_return_direction[(k-test_length):(k-1)])

for (i in 1:test_length) {
  
  forecast_return_direction[i] <- ifelse(compare_test$Forecast[[i]] > stock_prices[[k-test_length-1+i]], 'Up','Down')
  
}

forcast_table <- as.data.frame(compare_test)
forcast_table$actual_rd <- actual_return_direction
forcast_table$forecast_rd <- forecast_return_direction

names(forcast_table) <- c("Close price","Forecast price","Actual return","Forecast return")

accuracy_rate <- ifelse(forcast_table$`Actual return` == forcast_table$`Forecast return`, 1, 0)

formattable(forcast_table, align="l")

accuracy_aan <- print(paste0(round(sum(accuracy_rate)/test_length*100,2)," %"))


aan_table <- c(round(MAPE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(MAE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(RMSE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               accuracy_aan)

############# RWD Forecast accuracy rate and backtest table #####################################################

k <- length(stock_prices)
test <- NULL
forecast_return_direction <- NULL

for (i in 1:test_length) {
  
  train_f <- rwf((stock_prices[i:(k-test_length-1+i)]),h=1) # Model predicts price for next day
  test[i] <- round(train_f$mean,2)
  
}

compare_test <- round(stock_prices[(k-test_length+1):k],2)
compare_test$Forecast <- test

# Compute the log returns for the stock
stock_return <- diff(log(stock_prices),lag=1)
stock_return <- stock_return[!is.na(stock_return)]
stock_return <- coredata(stock_return)

actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
actual_return_direction <- as.vector(actual_return_direction[(k-test_length):(k-1)])

for (i in 1:test_length) {
  
  forecast_return_direction[i] <- ifelse(compare_test$Forecast[[i]] > stock_prices[[k-test_length-1+i]], 'Up','Down')
  
}

forcast_table <- as.data.frame(compare_test)
forcast_table$actual_rd <- actual_return_direction
forcast_table$forecast_rd <- forecast_return_direction

names(forcast_table) <- c("Close price","Forecast price","Actual return","Forecast return")

accuracy_rate <- ifelse(forcast_table$`Actual return` == forcast_table$`Forecast return`, 1, 0)

formattable(forcast_table, align="l")

accuracy_rwd <- print(paste0(round(sum(accuracy_rate)/test_length*100,2)," %"))


rwd_table <- c(round(MAPE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(MAE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(RMSE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               accuracy_rwd)

Accuracy(forcast_table$`Forecast price`, forcast_table$`Close price`)


############# Prophet Forecast accuracy rate and backtest table #####################################################


k <- length(stock_prices)
test <- NULL
forecast_return_direction <- NULL


compare_test <- round(stock_prices[(k-test_length+1):k],2)
compare_test$Forecast <- round(yhat[(k-test_length+1):k],2)

# Compute the log returns for the stock
stock_return <- diff(log(stock_prices),lag=1)
stock_return <- stock_return[!is.na(stock_return)]
stock_return <- coredata(stock_return)

actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
actual_return_direction <- as.vector(actual_return_direction[(k-test_length):(k-1)])

for (i in 1:test_length) {
  
  forecast_return_direction[i] <- ifelse(compare_test$Forecast[[i]] > stock_prices[[k-test_length-1+i]], 'Up','Down')
  
}

forcast_table <- as.data.frame(compare_test)
forcast_table$actual_rd <- actual_return_direction
forcast_table$forecast_rd <- forecast_return_direction

names(forcast_table) <- c("Close price","Forecast price","Actual return","Forecast return")

accuracy_rate <- ifelse(forcast_table$`Actual return` == forcast_table$`Forecast return`, 1, 0)

formattable(forcast_table, align="l")

accuracy_prophet <- print(paste0(round(sum(accuracy_rate)/test_length*100,2)," %"))

prophet_table <- c(round(MAPE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                   round(MAE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                   round(RMSE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
                   accuracy_prophet)


############# KNN Forecast accuracy rate and backtest table #####################################################

k <- length(stock_prices)
test <- NULL
forecast_return_direction <- NULL

for (i in 1:test_length) {
  
  test[i] <- round(knn_forecasting(knndf$y[i:(k-test_length-1+i)], h = 1, k=c(3,5,7))$prediction[[1]],2)
  
}

compare_test <- round(stock_prices[(k-test_length+1):k],2)
compare_test$Forecast <- test

# Compute the log returns for the stock
stock_return <- diff(log(stock_prices),lag=1)
stock_return <- stock_return[!is.na(stock_return)]
stock_return <- coredata(stock_return)

actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
actual_return_direction <- as.vector(actual_return_direction[(k-test_length):(k-1)])

for (i in 1:test_length) {
  
  forecast_return_direction[i] <- ifelse(compare_test$Forecast[[i]] > stock_prices[[k-test_length-1+i]], 'Up','Down')
  
}

forcast_table <- as.data.frame(compare_test)
forcast_table$actual_rd <- actual_return_direction
forcast_table$forecast_rd <- forecast_return_direction

names(forcast_table) <- c("Close price","Forecast price","Actual return","Forecast return")

accuracy_rate <- ifelse(forcast_table$`Actual return` == forcast_table$`Forecast return`, 1, 0)

formattable(forcast_table, align="l")

accuracy_arima <- print(paste0(round(sum(accuracy_rate)/test_length*100,2)," %"))

knn_table <- c(round(MAPE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(MAE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               round(RMSE(forcast_table$`Forecast price`, forcast_table$`Close price`),3),
               accuracy_arima)



####### Total backtest  result ######


accuracy_table <- rbind(arima_table,nnar_table, man_table, aan_table, rwd_table,prophet_table,knn_table)
colnames(accuracy_table) <- c("MAPE","MAE","RMSE","Forecast Rate Accuracy")
rownames(accuracy_table) <- c("ARIMA","NNTAR","MAN","AAN","RWD","Prophet","KNN")
accuracy_table <- as.data.frame(accuracy_table)


formattable(accuracy_table,
            align="l",
            list('Date' = formatter("span", style = ~ style(color = "grey",font.weight = "bold"))
            ))

