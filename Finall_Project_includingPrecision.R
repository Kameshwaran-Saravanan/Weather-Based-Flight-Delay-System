# ===============================
# 1. LOAD LIBRARIES
# ===============================
library(data.table)
library(dplyr)
library(caret)
library(xgboost)
library(randomForest)
library(pROC)
library(ggplot2)
library(reshape2)
library(shiny)

# ===============================
# 2. LOAD DATA
# ===============================
setwd("/Users/kamesh/Desktop/Weather Based Flight Delay")
data <- fread("DelayedFlights.csv")



# ===============================
# 3. SAMPLE (MEMORY SAFE)
# ===============================
set.seed(123)
data <- data[sample(nrow(data), 100000), ]

# ===============================
# 4. TARGET VARIABLE
# ===============================
data$delay_flag <- ifelse(data$ArrDelay > 15, 1, 0)

# ===============================
# 5. FEATURE ENGINEERING
# ===============================
data$weather_flag <- ifelse(data$WeatherDelay > 0,1,0)
data$carrier_flag <- ifelse(data$CarrierDelay > 0,1,0)
data$late_aircraft_flag <- ifelse(data$LateAircraftDelay > 0,1,0)

# ===============================
# 6. SELECT FEATURES
# ===============================
data_model <- data %>%
  select(
    DepDelay,
    Distance,
    WeatherDelay,
    CarrierDelay,
    LateAircraftDelay,
    weather_flag,
    carrier_flag,
    late_aircraft_flag,
    delay_flag
  ) %>%
  na.omit()

# ===============================
# 7. TRAIN TEST SPLIT
# ===============================
set.seed(123)
index <- createDataPartition(data_model$delay_flag, p=0.8, list=FALSE)

train <- data_model[index, ]
test  <- data_model[-index, ]

# ===============================
# 8. FORCE NUMERIC
# ===============================
train <- data.frame(lapply(train, as.numeric))
test  <- data.frame(lapply(test, as.numeric))

# ===============================
# 9. MATRICES
# ===============================
x_train <- data.matrix(train[, names(train) != "delay_flag"])
y_train <- train$delay_flag

x_test  <- data.matrix(test[, names(test) != "delay_flag"])
y_test  <- test$delay_flag

# ===============================
# 10. XGBOOST MODEL
# ===============================
scale_pos_weight <- sum(y_train==0)/sum(y_train==1)

dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest  <- xgb.DMatrix(data = x_test)

xgb_model <- xgb.train(
  params=list(
    objective="binary:logistic",
    eval_metric="auc",
    max_depth=6,
    eta=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight
  ),
  data=dtrain,
  nrounds=150,
  verbose=0
)

xgb_prob <- predict(xgb_model, dtest)
xgb_pred <- ifelse(xgb_prob > 0.5, 1, 0)

# ===============================
# 11. RANDOM FOREST
# ===============================
train_rf <- train
train_rf$delay_flag <- as.factor(train_rf$delay_flag)

rf_model <- randomForest(delay_flag ~ ., data=train_rf, ntree=100)

rf_prob <- predict(rf_model, test, type="prob")[,2]
rf_pred <- ifelse(rf_prob > 0.5, 1, 0)

# ===============================
# 12. LOGISTIC REGRESSION
# ===============================
log_model <- glm(delay_flag ~ ., data=train, family="binomial")

log_prob <- predict(log_model, test, type="response")
log_pred <- ifelse(log_prob > 0.5, 1, 0)

# ===============================
# 13. EVALUATION (UPDATED)
# ===============================
evaluate <- function(actual, pred, prob){
  
  actual <- factor(actual, levels=c(0,1))
  pred   <- factor(pred, levels=c(0,1))
  
  cm <- table(actual, pred)
  
  accuracy  <- sum(diag(cm)) / sum(cm)
  
  precision <- ifelse(sum(cm[,2])==0,0, cm[2,2]/sum(cm[,2]))
  
  recall    <- ifelse(sum(cm[2,])==0,0, cm[2,2]/sum(cm[2,]))
  
  f1        <- ifelse((precision+recall)==0,0,
                      2*((precision*recall)/(precision+recall)))
  
  auc_val <- auc(roc(actual, prob))
  
  return(c(accuracy, precision, recall, f1, auc_val))
}
# ===============================
# 14. RESULTS (UPDATED WITH PRECISION)
# ===============================

log_metrics <- evaluate(y_test, log_pred, log_prob)
rf_metrics  <- evaluate(y_test, rf_pred, rf_prob)
xgb_metrics <- evaluate(y_test, xgb_pred, xgb_prob)

results <- data.frame(
  Model = c("Logistic", "RandomForest", "XGBoost"),
  
  Accuracy  = c(log_metrics[1], rf_metrics[1], xgb_metrics[1]),
  Precision = c(log_metrics[2], rf_metrics[2], xgb_metrics[2]),
  Recall    = c(log_metrics[3], rf_metrics[3], xgb_metrics[3]),
  F1_Score  = c(log_metrics[4], rf_metrics[4], xgb_metrics[4]),
  AUC       = c(log_metrics[5], rf_metrics[5], xgb_metrics[5])
)

print(results)

# ===============================
# 15. HEATMAP (FINAL CORRECTED)
# ===============================

library(ggplot2)
library(reshape2)


corr_matrix <- cor(data_model)


corr_melt <- reshape2::melt(as.matrix(corr_matrix))


colnames(corr_melt) <- c("Feature1", "Feature2", "Correlation")


ggplot(corr_melt, aes(x = Feature1, y = Feature2, fill = Correlation)) +
  
  # Tiles
  geom_tile(color = "white", linewidth = 0.4) +
  
  # Values inside tiles
  geom_text(aes(label = round(Correlation, 2)),
            size = 3.5, fontface = "bold", color = "black") +
  

  scale_fill_gradient2(
    low = "#001f3f",
    mid = "#0074D9",
    high = "#7FDBFF",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Correlation"
  ) +
  
  # Clean theme
  theme_minimal(base_size = 13) +
  theme(
    plot.background = element_rect(fill = "#f5f7fa", color = NA),
    panel.background = element_rect(fill = "#f5f7fa"),
    
    axis.text.x = element_text(angle = 45, hjust = 1, size = 11),
    axis.text.y = element_text(size = 11),
    
    plot.title = element_text(
      size = 18,
      face = "bold",
      hjust = 0.5,
      color = "#003366"
    ),
    
    legend.title = element_text(face = "bold"),
    
    panel.grid = element_blank()
  ) +
  
  labs(
    title = "✈️ Flight Delay Feature Correlation Heatmap",
    x = "Features",
    y = "Features"
  )
# ===============================
# 16. SHINY GUI
# ===============================
ui <- fluidPage(
  
  tags$head(
    tags$style(HTML("
      body { background-color: #e0f7fa; }
      .title { text-align:center; font-size:30px; color:#006064; }
      .card { background:white; padding:20px; border-radius:10px; }
      .btn-custom { background:#00acc1; color:white; }
    "))
  ),
  
  div(class="title","✈️ Flight Delay Prediction System"),
  
  fluidRow(
    column(4,
           div(class="card",
               numericInput("dep","Departure Delay",10),
               numericInput("dist","Distance",500),
               numericInput("weather","Weather Delay",0),
               numericInput("carrier","Carrier Delay",0),
               numericInput("late","Late Aircraft Delay",0),
               actionButton("predict","Predict",class="btn-custom")
           )),
    
    column(8,
           div(class="card",
               h3("Result"),
               textOutput("result"),
               textOutput("prob")
           ))
  )
)
server <- function(input, output){
  
  observeEvent(input$predict,{
    
    user_data <- data.frame(
      DepDelay = input$dep,
      Distance = input$dist,
      WeatherDelay = input$weather,
      CarrierDelay = input$carrier,
      LateAircraftDelay = input$late,
      weather_flag = ifelse(input$weather>0,1,0),
      carrier_flag = ifelse(input$carrier>0,1,0),
      late_aircraft_flag = ifelse(input$late>0,1,0)
    )
    
    pred <- predict(xgb_model, xgb.DMatrix(data.matrix(user_data)))
    
    # ✅ RESTORED ORIGINAL ICONS
    result <- ifelse(pred > 0.5, "Delayed ✈️❌", "On-Time ✈️✅")
    
    output$result <- renderText({
      paste("Flight Status:", result)
    })
    
    output$prob <- renderText({
      paste("Probability:", round(pred,3))
    })
    
  })
}
shinyApp(ui, server)

# ===============================
# 17. MODEL COMPARISON GRAPH (FIXED)
# ===============================

library(reshape2)
library(ggplot2)

# ✅ FIXED MELT (no conflict)
results_long <- reshape2::melt(
  as.data.frame(results),
  id.vars = "Model"
)

# Plot
ggplot(results_long, aes(x = Model, y = value, fill = variable)) +
  
  geom_bar(stat = "identity", position = "dodge") +
  
  # SAME COLORS
  scale_fill_manual(values = c(
    "Accuracy" = "#1f77b4",
    "Precision" = "#2ca02c",
    "Recall" = "#ff7f0e",
    "F1_Score" = "#d62728",
    "AUC" = "#9467bd"
  )) +
  
  labs(
    title = "📊 Model Performance Comparison",
    x = "Models",
    y = "Score",
    fill = "Metrics"
  ) +
  
  theme_minimal(base_size = 13) +
  theme(
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5)
  )

