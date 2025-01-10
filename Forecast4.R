# Ensemble of Time Series Models  
# 1. ARIMA (AutoRegressive Integrated Moving Average)
# 2. Neural Network (NNetAR)
# 3. ETS Smoothing Methods (MAN & AAN models)
# 4. Random Walk with Drift (RWD)
# 5. Facebook Prophet
# 6. K-Nearest Neighbors (KNN) on Hadoop

# -------------------- Load Necessary Libraries --------------------
# Install any missing packages before loading
required_packages <- c("quantmod", "tseries", "ggplot2", "timeSeries", "forecast",
                       "xts", "timeDate", "data.table", "tidyverse", "lubridate",
                       "formattable", "MLmetrics", "prophet", "tsfknn")

installed_packages <- rownames(installed.packages())
for(pkg in required_packages){
  if(!pkg %in% installed_packages){
    install.packages(pkg, dependencies = TRUE)
  }
  library(pkg, character.only = TRUE)
}

# -------------------- Define Input Variables --------------------
stock_name <- 'AMZN'          # Stock symbol to analyze
analysis_period_days <- 5*365 # Number of days for the analysis period (approx. 5 years)
forecast_day <- 30            # Number of days to forecast
test_length <- 100            # Number of days used for backtesting

# -------------------- Set Date Range --------------------
end_date <- Sys.Date()
start_date <- end_date - analysis_period_days

# -------------------- Set Time Zone --------------------
MyZone <- "UTC"
Sys.setenv(TZ = MyZone)

# -------------------- Define Trading Days for Forecast --------------------
DayCal <- timeSequence(from = end_date, to = end_date + 2*forecast_day, by = "day", zone = MyZone)
TradingCal <- DayCal[isBizday(DayCal, holidayNYSE())] # Exclude NYSE holidays
forecast_Dates <- as.Date(TradingCal)[1:forecast_day] # Select the first 'forecast_day' trading days

# -------------------- Fetch Stock Data from Yahoo Finance --------------------
stock_allprice <- getSymbols(stock_name, from = start_date, to = end_date, auto.assign = FALSE)
stock_allprice <- na.omit(stock_allprice) # Remove any missing values

# Select the 'Close' price series
stock_prices <- stock_allprice[,4]
names(stock_prices) <- "Close"

# Combine historical dates with forecast dates for plotting
sample_date <- index(stock_prices)
all_date <- c(sample_date, forecast_Dates)

# -------------------- Define Forecasting Function --------------------
# This function performs forecasting, plotting, and tabulation for a given model
forecast_model <- function(model_name, forecast_result, all_date, stock_prices, forecast_day, method_label){
  
  # Extract forecast components
  mean_forecast <- round(as.vector(forecast_result$mean), 2)
  lower_forecast <- round(as.vector(forecast_result$lower), 2)
  upper_forecast <- round(as.vector(forecast_result$upper), 2)
  
  # Create vectors for plotting
  mean_value <- c(rep(NA, length(stock_prices)), mean_forecast)
  lower_value <- c(rep(NA, length(stock_prices)), lower_forecast)
  upper_value <- c(rep(NA, length(stock_prices)), upper_forecast)
  sample_value <- c(coredata(stock_prices), rep(NA, forecast_day))
  
  # Create data frame for plotting
  forecast_data <- data.frame(all_date, sample_value, mean_value, lower_value, upper_value)
  
  # Generate forecast plot
  plot_fig <- ggplot(forecast_data, aes(x = all_date)) +
    geom_line(aes(y = sample_value), size = 1, color = "steelblue") +  
    geom_line(aes(y = mean_value), size = 0.8, color = "red") +
    geom_ribbon(aes(ymin = lower_value, ymax = upper_value), fill = "firebrick", linetype = 2, alpha = 0.3) +
    labs(title = paste("Daily Price Forecast of", stock_name, "for Next", forecast_day, "Days"),
         subtitle = paste("Method:", method_label),
         x = "Date",
         y = "Price") +
    theme_minimal()
  
  print(plot_fig)
  
  # Create forecast table
  forecast_table <- formattable(data.table(Date = forecast_Dates,
                                           Forecast = mean_forecast,
                                           "Lower Band" = lower_forecast,
                                           "Upper Band" = upper_forecast),
                                align = c("l", "c", "c", "c"),
                                list('Date' = formatter("span", style = ~ style(color = "grey", font.weight = "bold"))))
  
  print(forecast_table)
  
  return(list(plot = plot_fig, table = forecast_table))
}

# -------------------- Forecast Using Different Models --------------------

# 1. ARIMA Model
arima_model <- auto.arima(stock_prices) # Automatically select ARIMA parameters
arima_forecast <- forecast(arima_model, h = forecast_day, level = 95) # Forecast next 'forecast_day' days
arima_results <- forecast_model("ARIMA", arima_forecast, all_date, stock_prices, forecast_day, arima_forecast$method)

# 2. Neural Network (NNetAR) Model
nnetar_model <- nnetar(stock_prices, lambda = "auto") # Fit Neural Network model with automatic Box-Cox transformation
nnetar_forecast <- forecast(nnetar_model, h = forecast_day, PI = TRUE, level = 90) # Forecast with prediction intervals
nnetar_results <- forecast_model("Neural Network", nnetar_forecast, all_date, stock_prices, forecast_day, nnetar_forecast$method)

# 3. ETS Smoothing Models (MAN & AAN)

# ETS MAN Model
ets_man_model <- ets(stock_prices, model = "MAN") # Fit ETS model with multiplicative error, additive trend, and no seasonality
ets_man_forecast <- forecast(ets_man_model, h = forecast_day, level = 95) # Forecast
ets_man_results <- forecast_model("ETS MAN", ets_man_forecast, all_date, stock_prices, forecast_day, ets_man_forecast$method)

# ETS AAN Model
ets_aan_model <- ets(stock_prices, model = "AAN") # Fit ETS model with additive error, additive trend, and no seasonality
ets_aan_forecast <- forecast(ets_aan_model, h = forecast_day, level = 95) # Forecast
ets_aan_results <- forecast_model("ETS AAN", ets_aan_forecast, all_date, stock_prices, forecast_day, ets_aan_forecast$method)

# 4. Random Walk with Drift (RWD) Model
rwf_model <- rwf(stock_prices, h = forecast_day, level = 95) # Fit Random Walk with Drift model
rwf_results <- forecast_model("Random Walk with Drift", rwf_model, all_date, stock_prices, forecast_day, rwf_model$method)

# 5. Facebook Prophet Model
# Prepare data for Prophet
prophet_df <- data.frame(ds = index(stock_prices),
                         y = as.numeric(stock_prices))

prophet_model <- prophet(prophet_df, daily.seasonality = TRUE) # Fit Prophet model
future_prophet <- make_future_dataframe(prophet_model, periods = forecast_day) # Create future dataframe
prophet_forecast <- predict(prophet_model, future_prophet) # Forecast

# Extract Prophet forecast for plotting
yhat <- prophet_forecast$yhat
yhat_lower <- prophet_forecast$yhat_lower
yhat_upper <- prophet_forecast$yhat_upper

# Create data frame for Prophet plot
prophet_plotdata <- data.frame(all_date, sample_value = c(coredata(stock_prices), rep(NA, forecast_day)),
                               yhat = yhat,
                               yhat_lower = yhat_lower,
                               yhat_upper = yhat_upper)

# Generate Prophet forecast plot
fig_prophet <- ggplot(prophet_plotdata, aes(x = all_date)) +
  geom_point(aes(y = sample_value), size = 1, color = "darkgray") +  
  geom_line(aes(y = yhat), size = 0.8, color = "red") +
  geom_ribbon(aes(ymin = yhat_lower, ymax = yhat_upper), fill = "firebrick", linetype = 2, alpha = 0.3) +
  labs(title = paste("Daily Price Forecast of", stock_name, "for Next", forecast_day, "Days"),
       subtitle = "Method: Prophet",
       x = "Date",
       y = "Price") +
  theme_minimal()

print(fig_prophet)

# Create Prophet forecast table
prophet_table <- formattable(data.table(Date = forecast_Dates,
                                        Forecast = yhat[(length(stock_prices)+1):(length(stock_prices)+forecast_day)],
                                        "Lower Band" = yhat_lower[(length(stock_prices)+1):(length(stock_prices)+forecast_day)],
                                        "Upper Band" = yhat_upper[(length(stock_prices)+1):(length(stock_prices)+forecast_day)]),
                             align = c("l", "c", "c", "c"),
                             list('Date' = formatter("span", style = ~ style(color = "grey", font.weight = "bold"))))

print(prophet_table)

# 6. K-Nearest Neighbors (KNN) Model
knn_df <- data.frame(ds = index(stock_prices),
                     y = as.numeric(stock_prices))

knn_forecast_result <- knn_forecasting(knn_df$y, h = forecast_day, k = c(3,5,7)) # Perform KNN forecasting

mean_knn <- round(coredata(knn_forecast_result$prediction), 2) # Extract and round KNN predictions

# Create data frame for KNN plot
knn_data <- data.frame(all_date, sample_value = c(coredata(stock_prices), rep(NA, forecast_day)),
                       knn_mean = mean_knn)

# Generate KNN forecast plot
fig_knn <- ggplot(knn_data, aes(x = all_date)) +
  geom_line(aes(y = sample_value), size = 1, color = "steelblue") +  
  geom_line(aes(y = knn_mean), size = 0.8, color = "red") +
  labs(title = paste("Daily Price Forecast of", stock_name, "for Next", forecast_day, "Days"),
       subtitle = "Method: K-Nearest Neighbors",
       x = "Date",
       y = "Price") +
  theme_minimal()

print(fig_knn)

# Create KNN forecast table
knn_table <- formattable(data.table(Date = forecast_Dates,
                                    Forecast = mean_knn),
                         align = c("l", "l"),
                         list('Date' = formatter("span", style = ~ style(color = "grey", font.weight = "bold"))))

print(knn_table)

# -------------------- Save All Plots to a PDF --------------------
pdf('forecast_plot.pdf', height = 6, width = 8)
print(fig1)
print(fig2)
print(fig3)
print(fig4)
print(fig5)
print(fig_prophet)
print(fig_knn)
dev.off()

# -------------------- Backtesting and Forecast Accuracy Calculation --------------------
# Function to perform backtesting for a given forecasting method
backtest_forecast <- function(forecast_func, model_label, stock_prices, test_length){
  k <- length(stock_prices)
  test_forecasts <- numeric(test_length)
  forecast_return_direction <- character(test_length)
  
  for (i in 1:test_length) {
    # Define the training window
    train_start <- i
    train_end <- k - test_length - 1 + i
    train_data <- window(stock_prices, start = time(stock_prices)[train_start], end = time(stock_prices)[train_end])
    
    # Generate forecast using the provided forecasting function
    forecast_result <- forecast_func(train_data, h = 1)
    
    # Store the forecasted value
    test_forecasts[i] <- round(as.vector(forecast_result$mean), 2)
  }
  
  # Actual values for comparison
  actual_values <- round(stock_prices[(k - test_length + 1):k], 2)
  
  # Calculate actual return directions
  stock_return <- diff(log(stock_prices), lag = 1)
  stock_return <- stock_return[!is.na(stock_return)]
  actual_return_direction <- ifelse(stock_return > 0, 'Up', 'Down')
  actual_return_direction <- as.vector(actual_return_direction[(k - test_length):(k - 1)])
  
  # Calculate forecast return directions
  for (i in 1:test_length) {
    forecast_return_direction[i] <- ifelse(test_forecasts[i] > stock_prices[[k - test_length -1 + i]], 'Up', 'Down')
  }
  
  # Create comparison table
  compare_test <- data.table(Close_Price = coredata(actual_values),
                             Forecast_Price = test_forecasts,
                             Actual_Return = actual_return_direction,
                             Forecast_Return = forecast_return_direction)
  
  # Calculate accuracy rate
  accuracy_rate <- mean(compare_test$Actual_Return == compare_test$Forecast_Return) * 100
  
  # Calculate error metrics
  mape <- round(MAPE(compare_test$Forecast_Price, compare_test$Close_Price), 3)
  mae <- round(MAE(compare_test$Forecast_Price, compare_test$Close_Price), 3)
  rmse <- round(RMSE(compare_test$Forecast_Price, compare_test$Close_Price), 3)
  
  # Compile results
  result <- c(MAPE = mape, MAE = mae, RMSE = rmse, Accuracy = paste0(round(accuracy_rate, 2), " %"))
  
  return(result)
}

# Define forecasting functions for each model
forecast_functions <- list(
  "ARIMA" = function(x, h) { forecast(auto.arima(x), h = h) },
  "NNetAR" = function(x, h) { forecast(nnetar(x, lambda = "auto"), h = h) },
  "ETS MAN" = function(x, h) { forecast(ets(x, model = "MAN"), h = h) },
  "ETS AAN" = function(x, h) { forecast(ets(x, model = "AAN"), h = h) },
  "RWD" = function(x, h) { forecast(rwf(x, h = h), h = h) },
  "Prophet" = function(x, h) { 
    df <- data.frame(ds = index(x), y = as.numeric(x))
    m <- prophet(df, daily.seasonality = TRUE)
    future <- make_future_dataframe(m, periods = h)
    forecast(m, future)
  },
  "KNN" = function(x, h) { 
    knn_forecasting(as.numeric(x), h = h, k = c(3,5,7))$prediction[[1]]
  }
)

# Perform backtesting for each model
backtest_results <- sapply(names(forecast_functions), function(model) {
  backtest_forecast(forecast_functions[[model]], model, stock_prices, test_length)
})

# Convert results to a data frame for better readability
accuracy_table <- as.data.frame(t(backtest_results))
colnames(accuracy_table) <- c("MAPE", "MAE", "RMSE", "Forecast Rate Accuracy")
rownames(accuracy_table) <- names(forecast_functions)

# Display the accuracy table
formattable(accuracy_table,
            align = "l",
            list('Forecast Rate Accuracy' = formatter("span", style = ~ style(color = "grey", font.weight = "bold"))))
