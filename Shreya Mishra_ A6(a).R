# Load necessary libraries
install.packages(forecast)
library(forecast)
library(tseries)
library(ggplot2)
library(caret)
library(randomForest)
library(e1071)

# Load your data
# df <- read.csv('/Users/shreyamishra/Desktop/DABUR.BO.csv')
# Convert the date column to Date type
df$Date <- as.Date(df$Date)
# Set the date column as row names
rownames(df) <- df$Date
df$Date <- NULL

# 2.1 Plot the time series
ggplot(df, aes(x = index(df), y = Adj.Close)) + 
  geom_line() + 
  labs(title = 'NFLX Adj Close Price', x = 'Date', y = 'Adj Close Price')

# 2.2 Decomposition of Time Series
decomposed <- stl(ts(df$Adj.Close, frequency = 12), s.window = "periodic")
plot(decomposed)

# Split the data into training and test sets
train_data <- head(df, round(0.8 * nrow(df)))
test_data <- tail(df, round(0.2 * nrow(df)))

### 3. Univariate Forecasting - Conventional Models/Statistical Models

# 3.1 Holt-Winters Model
monthly_data <- aggregate(Adj.Close ~ format(index(df), "%Y-%m"), df, mean)
monthly_data_ts <- ts(monthly_data$Adj.Close, frequency = 12)

# Fit the Holt-Winters model
hw_model <- HoltWinters(monthly_data_ts)

# Forecast for the next year (12 months)
hw_forecast <- forecast(hw_model, h = 12)

# Plot the forecast
plot(hw_forecast)

# Evaluate the model
y_pred <- hw_forecast$mean
rmse <- sqrt(mean((tail(monthly_data_ts, 12) - y_pred)^2))
mae <- mean(abs(tail(monthly_data_ts, 12) - y_pred))
mape <- mean(abs((tail(monthly_data_ts, 12) - y_pred) / y_pred)) * 100
r2 <- 1 - sum((tail(monthly_data_ts, 12) - y_pred)^2) / sum((tail(monthly_data_ts, 12) - mean(tail(monthly_data_ts, 12)))^2)
cat("RMSE:", rmse, "\nMAE:", mae, "\nMAPE:", mape, "\nR-squared:", r2, "\n")

# 3.2 ARIMA Monthly Data
auto_arima_model <- auto.arima(monthly_data_ts, seasonal = TRUE)

# Forecast with ARIMA
arima_forecast <- forecast(auto_arima_model, h = 8)

# Plot the forecast
plot(arima_forecast)

# Evaluate the model
y_pred <- arima_forecast$mean
rmse <- sqrt(mean((tail(monthly_data_ts, 8) - y_pred)^2))
mae <- mean(abs(tail(monthly_data_ts, 8) - y_pred))
mape <- mean(abs((tail(monthly_data_ts, 8) - y_pred) / y_pred)) * 100
r2 <- 1 - sum((tail(monthly_data_ts, 8) - y_pred)^2) / sum((tail(monthly_data_ts, 8) - mean(tail(monthly_data_ts, 8)))^2)
cat("RMSE:", rmse, "\nMAE:", mae, "\nMAPE:", mape, "\nR-squared:", r2, "\n")

# 3.3 ARIMA Daily Data
daily_data_ts <- ts(df$Adj.Close, frequency = 365)
auto_arima_model <- auto.arima(daily_data_ts, seasonal = TRUE)

# Forecast with ARIMA
arima_forecast <- forecast(auto_arima_model, h = 60)

# Plot the forecast
plot(arima_forecast)

# 4. Multivariate Forecasting - Machine Learning Models

# Normalize the data
preprocess_params <- preProcess(df, method = c("center", "scale"))
scaled_df <- predict(preprocess_params, df)

# Create sequences
create_sequences <- function(data, target_col, sequence_length) {
  sequences <- list()
  labels <- c()
  for (i in seq(1, nrow(data) - sequence_length)) {
    seq_data <- data[i:(i + sequence_length - 1), ]
    sequences <- append(sequences, list(seq_data))
    labels <- c(labels, data[i + sequence_length, target_col])
  }
  return(list(sequences = sequences, labels = labels))
}

# Define parameters
sequence_length <- 30
target_col <- which(names(scaled_df) == "Adj.Close")

# Create sequences
sequences_data <- create_sequences(scaled_df, target_col, sequence_length)
X <- sequences_data$sequences
y <- sequences_data$labels

# Split into train and test sets
train_indices <- 1:round(0.8 * length(y))
X_train <- X[train_indices]
X_test <- X[-train_indices]
y_train <- y[train_indices]
y_test <- y[-train_indices]

# Train Decision Tree model
dt_model <- train(y_train ~ ., data = do.call(rbind, X_train), method = "rpart")
y_pred_dt <- predict(dt_model, newdata = do.call(rbind, X_test))

# Evaluate the model
rmse <- sqrt(mean((y_test - y_pred_dt)^2))
mae <- mean(abs(y_test - y_pred_dt))
mape <- mean(abs((y_test - y_pred_dt) / y_pred_dt)) * 100
r2 <- 1 - sum((y_test - y_pred_dt)^2) / sum((y_test - mean(y_test))^2)
cat("Decision Tree - RMSE:", rmse, "\nMAE:", mae, "\nMAPE:", mape, "\nR-squared:", r2, "\n")

# Train Random Forest model
rf_model <- randomForest(y_train ~ ., data = do.call(rbind, X_train))
y_pred_rf <- predict(rf_model, newdata = do.call(rbind, X_test))

# Evaluate the model
rmse <- sqrt(mean((y_test - y_pred_rf)^2))
mae <- mean(abs(y_test - y_pred_rf))
mape <- mean(abs((y_test - y_pred_rf) / y_pred_rf)) * 100
r2 <- 1 - sum((y_test - y_pred_rf)^2) / sum((y_test - mean(y_test))^2)
cat("Random Forest - RMSE:", rmse, "\nMAE:", mae, "\nMAPE:", mape, "\nR-squared:", r2, "\n")

# Plot the predictions vs true values for Decision Tree
plot(y_test, type = "l", col = "blue", main = "Decision Tree: Predictions vs True Values", xlab = "Index", ylab = "Adj Close Price")
lines(y_pred_dt, col = "red")
legend("topright", legend = c("True Values", "Predicted Values"), col = c("blue", "red"), lty = 1)

# Plot the predictions vs true values for Random Forest
plot(y_test, type = "l", col = "blue", main = "Random Forest: Predictions vs True Values", xlab = "Index", ylab = "Adj Close Price")
lines(y_pred_rf, col = "red")
legend("topright", legend = c("True Values", "Predicted Values"), col = c("blue", "red"), lty = 1)
