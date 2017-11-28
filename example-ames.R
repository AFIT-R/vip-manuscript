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
library(xgboost)

# Load the data set
ames <- read.csv("ames.csv", header = TRUE)[, -1L]

# Log transform sale price
ames$LogSalePrice <- log(ames$SalePrice)
ames$SalePrice <- NULL


################################################################################
# Exploratory data analysis
################################################################################

# Histogram of response
hist(ames$LogSalePrice, 50)


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

so# Plot variable importance scores
p1 <- vip(ames.gbm)
p2 <- vip(ames.gbm, partial = TRUE, n.trees = best.iter)
grid.arrange(p1, p2, ncol = 2)

# Partial depence plots
ames.vi <- vi(ames.gbm, partial = TRUE, keep.partial = TRUE, 
              n.trees = best.iter)
ames.pd <- attr(ames.vi, "partial")[ames.vi$Variable[1L:16L]]
ames.pd <- plyr::ldply(ames.pd, .id = "x.name", .fun = function(x) {
  names(x)[1L] <- "x.value"
  x
})
p <- ggplot(ames.pd, aes(x = x.value, y = yhat)) +
  geom_line() +
  # geom_point(size = 0.5) +
  # geom_smooth(se = FALSE, linetype = "dashed") +
  geom_hline(yintercept = mean(log(ames2$SalePrice)), linetype = "dashed") +
  facet_wrap( ~ x.name, scales = "free_x") +
  theme_light() +
  xlab("") +
  ylab("Partial dependence")
p


################################################################################
# Stacked
################################################################################

# Initialize and connect to H2O
h2o.init(nthreads = -1)

# Random seed
seed <- 2101

# Variable names
x <- names(subset(ames, select = -LogSalePrice))
y <- "LogSalePrice"
trn <- as.h2o(ames)
# trn <- trn[, -1L]  # ??

# Fit a random forest model
RFd <- h2o.randomForest(
  x, y, 
  training_frame = trn, 
  model_id = "RF_defaults", 
  nfolds = 10,
  fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE,
  ntrees = 1000,
  stopping_rounds = 5,
  stopping_metric = "RMSE",
  stopping_tolerance = 0.001,
  seed = seed
)

# Fit a GBM
GBMd <- h2o.gbm(
  x, y, 
  training_frame = trn, 
  model_id = "GBM_defaults", 
  nfolds = 10,
  fold_assignment = "Modulo", 
  keep_cross_validation_predictions = TRUE,
  ntrees = 10000,
  max_depth = 5,
  learn_rate = 0.01,
  stopping_rounds = 5,
  stopping_metric = "RMSE",
  stopping_tolerance = 0.001,
  seed = seed
)

# Extract variable importance
vi.RFd <- as.data.frame(h2o.varimp(RFd))[, -(3L:4L)]
vi.GBMd <- as.data.frame(h2o.varimp(GBMd))[, -(3L:4L)]
vi.RFd$model <- "RFd"
vi.GBMd$model <- "GBMd"
vi.all <- rbind(vi.RFd, vi.GBMd)
names(vi.RFd) <- c("variable", "importance", "model")
names(vi.GBMd) <- c("variable", "importance", "model")

# Train a stacked ensemble using the RF and GBMabove
ensemble <- h2o.stackedEnsemble(
  x = x,
  y = y,
  training_frame = trn,
  base_models = list(RFd, GBMd)
)

# Compute partial dependence values and variable importance scores
pds <- h2o.partialPlot(ensemble, data = trn, nbins = 25, plot = FALSE, 
                       plot_stddev = FALSE)
vi <- unlist(lapply(pds, FUN = function(x) {
  sd(x[["mean_response"]])
}))
names(vi) <- x
vi <- sort(vi, decreasing = TRUE)
vi.ensemble <- data.frame(variable = names(vi), importance = vi, 
                          model = "ensemble")
vi.all <- rbind(vi.all, vi.ensemble)

library(ggplot2)
p1 <- ggplot(head(vi.RFd, n = 10), 
             aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  theme_light() +
  xlab("") +
  ylab("Importance")
p2 <- ggplot(head(vi.GBMd, n = 10), 
             aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  theme_light() +
  xlab("") +
  ylab("Importance")
p3 <- ggplot(head(vi.ensemble, n = 10), 
             aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  theme_light() +
  xlab("") +
  ylab("Importance")
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)

# Save the results
write.csv(vi.all, file = "ames_vi_all.csv", row.names = FALSE)

# GLMd <- h2o.glm(x, y, trn, model_id = "GLM_defaults", nfolds = 10,
#                 fold_assignment = "Modulo", 
#                 keep_cross_validation_predictions = TRUE)
# DLd <- h2o.deeplearning(x, y, trn, model_id = "DL_defaults", nfolds = 10,
#                         fold_assignment = "Modulo", 
#                         keep_cross_validation_predictions = TRUE)

