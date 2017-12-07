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
ames <- read.csv("ames.csv", header = TRUE)[, -1L]

# Log transform sale price
ames$LogSalePrice <- log(ames$SalePrice)
ames$SalePrice <- NULL


################################################################################
# GBM
################################################################################

# Fit a generalized boosted regression model
set.seed(1138)
ames.gbm <- gbm(LogSalePrice ~ ., data = ames,
                distribution = "gaussian",
                n.trees = 3000,
                interaction.depth = 5,
                shrinkage = 0.01,
                bag.fraction = 1,
                train.fraction = 1,
                cv.folds = 10,
                verbose = TRUE)

# Compute "optimal" number of iterations based on CV results
best.iter <- gbm.perf(ames.gbm, method = "cv")
print(best.iter)

# Plot relative influence of each predictor
summary(ames.gbm, n.trees = best.iter)

# Plot variable importance scores
p1 <- vip(ames.gbm)
p2 <- vip(ames.gbm, partial = TRUE, n.trees = best.iter)
grid.arrange(p1, p2, ncol = 2)

vi.gbm <- vi(ames.gbm)
vi.gbm <- vi.gbm[vi.gbm$Importance > 0, ]

# Partial depence plots
ames.vi <- vi(ames.gbm, partial = TRUE, keep.partial = TRUE, 
              n.trees = best.iter)
vars <- as.character(vi.gbm$Variable[c(1L:3L, (nrow(vi.gbm) - 2):nrow(vi.gbm))])
ames.pd <- attr(ames.vi, "partial")[vars]
ames.pd <- plyr::ldply(ames.pd, .id = "x.name", .fun = function(x) {
  names(x)[1L] <- "x.value"
  x
})
p <- ggplot(ames.pd, aes(x = x.value, y = yhat)) +
  geom_line() +
  # geom_point(size = 1.5) +
  # geom_smooth(se = FALSE, linetype = "dashed") +
  geom_hline(yintercept = mean(ames$LogSalePrice), linetype = "dashed") +
  facet_wrap( ~ x.name, scales = "free_x") +
  theme_light() +
  xlab("") +
  ylab("Partial dependence")
p


################################################################################
# RF, GBM, and RF + GBM (i.e., stacked ensemble/super learner)
################################################################################

# Initialize and connect to H2O
h2o.init(nthreads = -1)

# Random seed
seed <- 2101

# Variable names
x <- names(subset(ames, select = -LogSalePrice))
y <- "LogSalePrice"

# Convert training data to an H2OFrame
trn <- as.h2o(ames)

# # Fit an RF
# ames_rf <- h2o.randomForest(
#   x = x, 
#   y = y, 
#   training_frame = trn, 
#   model_id = "ames_rf", 
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
# # Fit a GBM
# ames_gbm <- h2o.gbm(
#   x = x, 
#   y = y, 
#   training_frame = trn, 
#   model_id = "ames_gbm", 
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
# # Train a stacked ensemble using the previously fit RF and GBM
# ames_ensemble <- h2o.stackedEnsemble(
#   x = x,
#   y = y,
#   training_frame = trn,
#   model_id = "ames_ensemble",
#   base_models = list(ames_rf, ames_gbm)
# )

# # Save the models
# h2o.saveModel(object = ames_rf, path = getwd(), force = TRUE)
# h2o.saveModel(object = ames_gbm, path = getwd(), force = TRUE)
# h2o.saveModel(object = ames_ensemble, path = getwd(), force = TRUE)

# Load the models
ames_rf <- h2o.loadModel("ames_rf")
ames_gbm <- h2o.loadModel("ames_gbm")
ames_ensemble <- h2o.loadModel("ames_ensemble")

# Extract variable importance scores from base models
vi_ames_rf <- as.data.frame(h2o.varimp(ames_rf))[, -(3L:4L)]
vi_ames_gbm <- as.data.frame(h2o.varimp(ames_gbm))[, -(3L:4L)]
vi_ames_rf$model <- "rf"
vi_ames_gbm$model <- "gbm"
names(vi_ames_rf) <- c("variable", "importance", "model")
names(vi_ames_gbm) <- c("variable", "importance", "model")
vi_ames_all <- rbind(vi_ames_rf, vi_ames_gbm)

# Compute partial dependence values and variable importance scores
pd_ames_ensemble <- h2o.partialPlot(ames_ensemble, data = trn, nbins = 25, 
                                    plot = FALSE, plot_stddev = FALSE)
vi_ames_ensemble <- unlist(lapply(pd_ames_ensemble, FUN = function(x) {
  sd(x[["mean_response"]])
}))
names(vi_ames_ensemble) <- x
vi_ames_ensemble <- sort(vi_ames_ensemble, decreasing = TRUE)
vi_ames_ensemble <- data.frame(
  variable = names(vi_ames_ensemble), 
  importance = vi_ames_ensemble, 
  model = "ensemble"
)
vi_ames_all <- rbind(vi_ames_all, vi_ames_ensemble)

# Save the results
write.csv(vi_ames_all, file = "vi_ames_all.csv", row.names = FALSE)
