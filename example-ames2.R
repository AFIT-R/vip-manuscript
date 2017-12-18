# The Ames Housing dataset was compiled by Dean De Cock for use in data science 
# education. It's an incredible alternative for data scientists looking for a 
# modernized and expanded version of the often cited Boston Housing dataset.
#
# The goal is to build a model to predict the sales price for each house. The 
# metric used in the associated Kaggle comptetiopn is the Root-Mean-Squared-
# Error (RMSE) between the logarithm of the predicted value and the logarithm of 
# the observed sales price. (Taking logs means that errors in predicting 
# expensive houses and cheap houses will affect the result equally.)


################################################################################
# Setup
################################################################################

# Load required packages
library(gbm)
library(ggplot2)
library(h2o)
library(vip)

# Load the data set
ames2 <- AmesHousing::make_ames()

# Log transform sale price
ames2$Log_Sale_Price <- log(ames2$Sale_Price)
ames2$Sale_Price <- NULL


################################################################################
# RF, GBM, and RF + GBM (i.e., stacked ensemble/super learner)
################################################################################

# Initialize and connect to H2O
h2o.init(nthreads = -1)

# Random seed
seed <- 1458

# Variable names
x <- names(subset(ames2, select = -Log_Sale_Price))
y <- "Log_Sale_Price"

# Convert training data to an H2OFrame
trn <- as.h2o(ames2)

# # Fit a GBM
# ames2_gbm <- h2o.gbm(
#   x = x,
#   y = y,
#   training_frame = trn,
#   model_id = "ames2_gbm",
#   nfolds = 10,
#   fold_assignment = "Modulo",
#   keep_cross_validation_predictions = TRUE,
#   ntrees = 10000,
#   max_depth = 5,
#   learn_rate = 0.01,
#   stopping_rounds = 5,
#   stopping_metric = "RMSE",
#   stopping_tolerance = 0.001,
#   seed = seed
# )
# 
# # Fit an RF
# ames2_rf <- h2o.randomForest(
#   x = x,
#   y = y,
#   training_frame = trn,
#   model_id = "ames2_rf",
#   nfolds = 10,
#   fold_assignment = "Modulo",
#   keep_cross_validation_predictions = TRUE,
#   ntrees = 1000,
#   stopping_rounds = 5,
#   stopping_metric = "RMSE",
#   stopping_tolerance = 0.001,
#   seed = seed
# )
# 
# # Train a stacked ensemble using the previously fit RF and GBM
# ames2_ensemble <- h2o.stackedEnsemble(
#   x = x,
#   y = y,
#   training_frame = trn,
#   model_id = "ames2_ensemble",
#   base_models = list(ames2_rf, ames2_gbm)
# )
# 
# # # Save the models
# h2o.saveModel(object = ames2_rf, path = getwd(), force = TRUE)
# h2o.saveModel(object = ames2_gbm, path = getwd(), force = TRUE)
# h2o.saveModel(object = ames2_ensemble, path = getwd(), force = TRUE)

# Load the models
ames2_rf <- h2o.loadModel("ames2_rf")
ames2_gbm <- h2o.loadModel("ames2_gbm")
ames2_ensemble <- h2o.loadModel("ames2_ensemble")

# Extract variable importance scores from base models
ames2_vi_rf <- as.data.frame(h2o.varimp(ames2_rf))[, -(3L:4L)]
ames2_vi_gbm <- as.data.frame(h2o.varimp(ames2_gbm))[, -(3L:4L)]
ames2_vi_rf$model <- "rf"
ames2_vi_gbm$model <- "gbm"
names(ames2_vi_rf) <- c("variable", "importance", "model")
names(ames2_vi_gbm) <- c("variable", "importance", "model")
ames2_vi_all <- rbind(ames2_vi_rf, ames2_vi_gbm)

# Compute partial dependence values and variable importance scores
# ames2_pd_ensemble <- h2o.partialPlot(ames2_ensemble, data = trn, nbins = 30,
#                                      plot = FALSE, plot_stddev = FALSE)
# save(ames2_pd_ensemble, file = "ames2_pd_ensemble.RData")
load("ames2_pd_ensemble.RData")
ames2_vi_ensemble <- unlist(lapply(ames2_pd_ensemble, FUN = function(x) {
  if (is.numeric(x[[1L]])) {
    sd(x[["mean_response"]])
  } else {
    diff(range(x[["mean_response"]])) / 4
  }
}))
names(ames2_vi_ensemble) <- x
ames2_vi_ensemble <- sort(ames2_vi_ensemble, decreasing = TRUE)
ames2_vi_ensemble <- data.frame(
  variable = names(ames2_vi_ensemble),
  importance = ames2_vi_ensemble,
  model = "ensemble"
)
ames2_vi_all <- rbind(ames2_vi_all, ames2_vi_ensemble)

# Save the results
write.csv(ames2_vi_all, file = "ames2_vi_all.csv", row.names = FALSE)
