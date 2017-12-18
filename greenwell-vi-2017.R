################################################################################
# Setup
################################################################################

# Install required packages
# pkgs <- c("caret", "devtools", "dplyr", "gbm", "NeuralNetTools", "nnet", "pdp", 
          # "randomForest")
# install.packages(pkgs)
# devtools::install_github("AFIT-R/vip")

# Load required packages
library(caret)           # for model tuning/training (also loads ggplot2)
library(dplyr)           # for data manipulation
library(ggplot2)         # for fancier plots
library(h2o)
library(NeuralNetTools)  # for Garson and Olden's algorithms
library(nnet)            # for fitting neural networks
library(pdp)             # for constructing partial dependence plots
library(randomForest)    # for fitting random forests
library(vip)             # for constructing variable importance plots

# Load data sets
ames <- read.csv("ames.csv", header = TRUE)[, -1L]  # rm ID column

# Log transform sale price
ames$LogSalePrice <- log(ames$SalePrice)
ames$SalePrice <- NULL

# Colors
set1 <- RColorBrewer::brewer.pal(9, "Set1")


################################################################################
# Section 2.4: The Ames housing data set
################################################################################

# Initialize and connect to H2O
h2o.init(nthreads = -1)

# Variable names
x <- names(subset(ames, select = -LogSalePrice))
y <- "LogSalePrice"
trn <- as.h2o(ames)

# Random seed
seed <- 2101

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

# Load the fitted GBM model
ames_gbm <- h2o.loadModel("ames_gbm")

# Extract variable importance scores
vi_ames_gbm <- h2o.varimp(ames_gbm)
vi_ames_gbm <- vi_ames_gbm[vi_ames_gbm$relative_importance > 0, ]

# Figure 1
ames_gbm_top_15 <- vi_ames_gbm$variable[1L:15L]
# ames_gbm_bottom_15 <- rev(vi_ames_gbm$variable)[1L:15L]
# p1 <- vip(ames_gbm, pred.var = ames_gbm_top_15, horizontal = TRUE) +
#   theme_light()
# p2 <- vip(ames_gbm, pred.var = ames_gbm_bottom_15, horizontal = TRUE) +
#   theme_light()
pdf(file = "ames-gbm-vip.pdf", width = 7, height = 5)
vip(ames_gbm, pred.var = ames_gbm_top_15, horizontal = TRUE) +
  theme_light()
dev.off()

# Compute partial dependence for top/bottom three predictors
vars <- c(head(vi_ames_gbm$variable, 3), tail(vi_ames_gbm$variable, 3))
pd_list <- h2o.partialPlot(ames_gbm, data = trn, cols = vars, nbins = 25, 
                           plot = FALSE, plot_stddev = FALSE)
names(pd_list) <- vars
rng <- range(unlist(lapply(pd_list, FUN = function(x) range(x[[2L]]))))
ref_line <- mean(ames$LogSalePrice)
pd_plots <- plyr::llply(pd_list, .fun = function(x) {
  x[[3L]] <- NULL
  x[[1L]] <- as.numeric(as.factor(x[[1L]]))
  onames <- names(x)
  names(x) <- c("xvar", "yvar")
  p <- ggplot(x, aes(xvar, yvar)) +
    geom_hline(yintercept = ref_line, color = set1[1L], linetype = "dashed") +
    geom_line() +
    ylim(rng) +
    xlab(onames[1L]) +
    ylab("Partial dependence") +
    theme_light()
  p
})

# Figure 2
pdf(file = "ames-gbm-pdps.pdf", width = 7, height = 5)
gridExtra::grid.arrange(grobs = pd_plots, ncol = 3)
dev.off()


################################################################################
# Section 3: A partial dependence-based variable importance measure
################################################################################

# Compute partial dependence variable importance scores
# imp <- vi(ames_gbm, partial = TRUE, data = trn, nbins = 25)
# save(imp, file = "ames-gbm-vi.RData")
load("ames-gbm-vi.RData")
imp15 <- imp[1L:15L, ]

# Variable importance plot
p <- ggplot(imp15, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  xlab("") +
  ylab("Importance") +
  coord_flip() +
  theme_light()

# Figure 3
pdf(file = "ames-gbm-vip-pd.pdf", width = 7, height = 5)
p
dev.off()

imp2 <- vi(ames_gbm, pred.var = x)
p1 <- ggplot(imp2, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_point(size = 1, col = "grey35") +
  geom_segment(aes(x = reorder(Variable, Importance), 
                   xend = reorder(Variable, Importance), 
                   y = 0, yend = Importance), 
               col = "grey35", size = 0.1) +
  xlab("") +
  ylab("Importance") +
  coord_flip() +
  theme_light() + 
  theme(axis.text = element_text(size = 6))
p2 <- ggplot(imp, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_point(size = 1, col = "grey35") +
  geom_segment(aes(x = reorder(Variable, Importance), 
                   xend = reorder(Variable, Importance), 
                   y = 0, yend = Importance), 
               col = "grey35", size = 0.1) +
  xlab("") +
  ylab("Importance") +
  coord_flip() +
  theme_light() + 
  theme(axis.text = element_text(size = 6))

# Figure ?
pdf(file = "ames-gbm-vi-both.pdf", width = 7, height = 7)
grid.arrange(p1, p2, ncol = 2)
dev.off()


################################################################################
# Section 3.1: Linear models
################################################################################

# Simulate data
n <- 1000
set.seed(101)
x1 <- runif(n, min = 0, max = 1)
x2 <- runif(n, min = 0, max = 1)
d <- data.frame(x1, x2, y = 1 + 3*x1 - 5*x2 + rnorm(n, sd = 0.1))
pairs(d)

# Fit a simple linear model
fit <- lm(y ~ x1 + x2, data = d)

# Estimated and true partial dependence plots
pd1 <- partial(fit, pred.var = "x1")
pd2 <- partial(fit, pred.var = "x2")

# Figure 4
pdf(file = "lm-pdps.pdf", width = 8, height = 4)
grid.arrange(
  autoplot(pd1, pdp.size = 3.2, alpha = 0.5) + 
    geom_abline(slope = 3, intercept = -3/2, col = "red") +
    xlab(expression(X[1])) +
    ylab("Partial dependence") +
    ylim(-2.5, 2.5) +
    theme_light(), 
  autoplot(pd2, pdp.size = 1.2, alpha = 0.5) + 
    geom_abline(slope = -5, intercept = 5/2, col = "red") +
    xlab(expression(X[2])) +
    ylab("Partial dependence") +
    ylim(-2.5, 2.5) +
    theme_light(), 
  ncol = 2
)
dev.off()


################################################################################
# Section 4: Friedman's regression problem
################################################################################

# Simulate the data
set.seed(101)  # for reproducibility
trn <- as.data.frame(mlbench::mlbench.friedman1(n = 500, sd = 1))

# # Setup for k-fold cross-validation
# ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
# set.seed(103)
# trn.nn.tune <- train(
#   x = subset(trn, select = -y),
#   y = trn$y,
#   method = "nnet",
#   trace = FALSE,
#   linout = TRUE,
#   maxit = 1000,
#   trControl = ctrl,
#   tuneGrid = expand.grid(size = 1:20, decay = c(0, 0.0001, 0.001, 0.01, 0.1))
# )
# plot(trn.nn.tune)
#    size decay     RMSE  Rsquared     RMSESD  RsquaredSD
# 39    8  0.01 1.205598 0.9443347 0.08825044 0.005865337

# Fit a neural network to the Firedman 1 data set
set.seed(103)
trn.nn <- nnet(y ~ ., data = trn, size = 8, linout = TRUE, decay = 0.01,
               maxit = 1000, trace = FALSE)

# vip(trn.nn, pred.var = paste0("x.", 1:10), FUN = var)
# vip(trn.nn, pred.var = paste0("x.", 1:10), FUN = mad)

# VIP: partial dependence algorithm
p1 <- vip(trn.nn, use.partial = TRUE, pred.var = paste0("x.", 1:10)) +
  theme_light() +
  ylab("Importance (partial dependence)")

# VIP: Garson's algorithm
trn.nn.garson <- garson(trn.nn, bar_plot = FALSE) %>%
  tibble::rownames_to_column("Variable") %>%
  select(Variable, Importance = rel_imp)
p2 <- ggplot(trn.nn.garson, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  xlab("") +
  ylab("Importance (Garson's algorithm)") +
  coord_flip() +
  theme_light()

# VIP: Olden's algorithm
trn.nn.olden <- olden(trn.nn, bar_plot = FALSE) %>%
  tibble::rownames_to_column("Variable") %>%
  select(Variable, Importance = importance)
p3 <- ggplot(trn.nn.olden, aes(x = reorder(Variable, Importance), y = Importance)) +
  geom_col() +
  xlab("") +
  ylab("Importance (Olden's algorithm)") +
  coord_flip() +
  theme_light()

# Figure 5
# pdf(file = "network-vip.pdf", width = 12, height = 6)
grid.arrange(p1, p2, p3, ncol = 3)
# dev.off()

# vint <- function(x) {
#   pd <- partial(trn.nn, pred.var = c(x[1L], x[2L]))
#   c(sd(tapply(pd$yhat, INDEX = pd[[x[1L]]], FUN = sd)),
#     sd(tapply(pd$yhat, INDEX = pd[[x[2L]]], FUN = sd)))
# }
# combns <- combn(paste0("x.", 1:10), m = 2)
# res <- plyr::aaply(combns, .margins = 2, .fun = vint, .progress = "text")
# plot(rowMeans(res), type = "h")
# int <- data.frame(x = paste0(combns[1L, ], "*", combns[2L, ]), y = rowMeans(res))
# int <- int[order(int$y, decreasing = TRUE), ]
# save(int, file = "interaction-statistics.RData")
load("interaction-statistics.RData")

# Figure 6
# pdf(file = "network-int.pdf", width = 8, height = 4)
labs <- c(
  expression(x[1]*x[2]), expression(x[1]*x[3]), expression(x[3]*x[10]), 
  expression(x[1]*x[8]), expression(x[3]*x[8]), expression(x[4]*x[5]),
  expression(x[3]*x[4]), expression(x[1]*x[4]), expression(x[2]*x[4]),
  expression(x[1]*x[5])
)
ggplot(int[1:10, ], aes(reorder(x, -y), y)) +
  geom_col(width = 0.75) +
  xlab("") +
  ylab("Interaction") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  scale_x_discrete("", labels = labs) +
  theme_light()
# dev.off()

# Simulation -------------------------------------------------------------------
# simSD <- function(pred.var, pd.fun) {
#   x <- y <- seq(from = 0, to = 1, length = 100)
#   xy <- expand.grid(x, y)
#   z <- apply(xy, MARGIN = 1, FUN = function(x) {
#     pd.fun(x[1L], x[2L])
#   })
#   res <- as.data.frame(cbind(xy, z))
#   names(res) <- c(pred.var, "yhat")
#   form <- as.formula(paste("yhat ~", paste(paste(pred.var, collapse = "*"))))
#   p1 <- levelplot(form, data = res, col.regions = viridis::viridis,
#                   xlab = expression(x[1]), ylab = expression(x[4]))
#   approxVar.x <- function(x = 0.5, n = 100000) {
#     y <- runif(n, min = 0, max = 1)
#     sd(pd.fun(x, y))
#   }
#   approxVar.y <- function(y = 0.5, n = 100000) {
#     x <- runif(n, min = 0, max = 1)
#     sd(pd.fun(x, y))
#   }
#   x <- seq(from = 0, to = 1, length = 100)
#   y1 <- sapply(x, approxVar.x)
#   y2 <- sapply(x, approxVar.y)
#   p2 <- xyplot(SD ~ x, data = data.frame(x = x, SD = y1), type = "l",
#                lwd = 1, col = "black",
#                ylim = c(min(y1, na.rm = TRUE) - 1, max(y1, na.rm = TRUE) + 1),
#                xlab = expression(x[1]), 
#                ylab = expression(imp ~ (x[4]*" | "*x[1])))
#   p3 <- xyplot(SD ~ x, data = data.frame(x = x, SD = y2), type = "l",
#                lwd = 1, col = "black",
#                ylim = c(min(y2, na.rm = TRUE) - 1, max(y2, na.rm = TRUE) + 1),
#                xlab = expression(x[4]), 
#                ylab = expression(imp ~ (x[1]*" | "*x[4])))
#   grid.arrange(p1, p2, p3, ncol = 3)
# }
# p1 <- simSD(c("x.1", "x.2"), pd.fun = function(x1, x2) {
#   5 * (pi * x1 * (12 * x2 + 5) - 12 * cos(pi * x1) + 12) / (6 * pi * x1)
# })
# p2 <- simSD(c("x.1", "x.4"), pd.fun = function(x1, x2) {
#   10 * sin(pi * x1 * x2) + 55 / 6
# })
# 
# # Figure 7
# pdf(file = "interaction-simulation.pdf", width = 12, height = 8)
# grid.arrange(p1, p2, nrow = 2)
# dev.off()
# ------------------------------------------------------------------------------


################################################################################
# Section 4.1: Friedman's H-statistic
################################################################################

# Fit a GBM
set.seed(937)
trn.gbm <- gbm(y ~ ., data = trn, distribution = "gaussian", n.trees = 25000,
               shrinkage = 0.01, interaction.depth = 2, bag.fraction = 1,
               train.fraction = 0.8, cv.folds = 5, verbose = TRUE)
best.iter <- gbm.perf(trn.gbm, method = "cv")
print(best.iter)

# Variable importance plots
summary(trn.gbm, n.trees = best.iter)
vip.gbm <- vip(trn.gbm, pred.var = paste0("x.", 1:10), n.trees = best.iter)
print(vip.gbm)

# Friedman's H-statistic
combns <- t(combn(paste0("x.", 1:10), m = 2))
int.h <- numeric(nrow(combns))
for (i in 1:nrow(combns)) {
  print(paste("iter", i, "of", nrow(combns)))
  int.h[i] <- interact.gbm(trn.gbm, data = trn, i.var = combns[i, ], 
                           n.trees = best.iter)
}
int.h <- data.frame(x = paste0(combns[, 1L], "*", combns[, 2L]), y = int.h)
dotchart(int.h$y, labels = int.h$x)
int.h <- int.h[order(int.h$y, decreasing = TRUE), ]
# dotchart(int.h$y, labels = int.h$x)

# Variable importance-based interaction statistic
vint <- function(x) {
  pd <- partial(trn.gbm, pred.var = c(x[1L], x[2L]), n.trees = best.iter)
  c(sd(tapply(pd$yhat, INDEX = pd[[x[1L]]], FUN = sd)),
    sd(tapply(pd$yhat, INDEX = pd[[x[2L]]], FUN = sd)))
}
int.i <- plyr::aaply(combns, .margins = 1, .fun = vint, .progress = "text")
int.i <- data.frame(x = paste0(combns[, 1L], "*", combns[, 2L]), 
                    y = rowMeans(int.i))
int.i <- int.i[order(int.i$y, decreasing = TRUE), ]

# Construct plot
p1 <- ggplot(int.h[1:10, ], aes(reorder(x, y), y)) +
  geom_col(width = 0.75) +
  xlab("") +
  ylab("Interaction (Friedman's H-statistic)") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  # scale_x_discrete("", labels = labs) +
  theme_light() +
  # scale_fill_viridis() +
  coord_flip()

# Construct plots
p2 <- ggplot(int.i[1:10, ], aes(reorder(x, y), y)) +
  geom_col(width = 0.75) +
  xlab("") +
  ylab("Interaction (variable importance)") +
  theme(axis.text.x = element_text(angle = 45, vjust = 1)) +
  # scale_x_discrete("", labels = labs) +
  theme_light() +
  coord_flip()

# Figure 8
# pdf(file = "gbm-int.pdf", width = 7, height = 7)
grid.arrange(p1, p2, ncol = 2)
# dev.off()


################################################################################
# Section 5: Stacked ensembles
################################################################################

# Load the Ames variable importance scores
vi_ames_all <- read.csv("vi_ames_all.csv", header = TRUE)

# Extract the top 15 from each model for plotting
vi_ames_rf <- vi_ames_all[vi_ames_all$model == "rf", ][1L:15L, ]
vi_ames_gbm <- vi_ames_all[vi_ames_all$model == "gbm", ][1L:15L, ]
vi_ames_ensemble <- vi_ames_all[vi_ames_all$model == "ensemble", ][1L:15L, ]

# Rescale variable importance scores
vi_ames_rf$importance <- vi_ames_rf$importance / max(vi_ames_rf$importance)
vi_ames_gbm$importance <- vi_ames_gbm$importance / max(vi_ames_gbm$importance)
vi_ames_ensemble$importance <- vi_ames_ensemble$importance / max(vi_ames_ensemble$importance)

# Plot the top 15 predcitors from each model
p1 <- ggplot(vi_ames_rf, aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  theme_light() +
  xlab("") +
  ylab("Scaled importance") +
  ggtitle("Random forest")
p2 <- ggplot(vi_ames_gbm, aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  theme_light() +
  xlab("") +
  ylab("Scaled importance") +
  ggtitle("GBM")
p3 <- ggplot(vi_ames_ensemble, aes(x = reorder(variable, importance), y = importance)) +
  geom_col() +
  coord_flip() +
  theme_light() +
  xlab("") +
  ylab("Scaled importance") +
  ggtitle("Stacked ensemble")

# Figure 9
pdf(file = "ames-ensemble-vip.pdf", width = 14, height = 7)
gridExtra::grid.arrange(p1, p2, p3, ncol = 3)
dev.off()


# # Load data
# data(pima, package = "pdp")
# pima <- na.omit(pima)
# 
# # Setup for repeated k-fold cross-validation
# ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 10, 
#                      classProbs = TRUE, summaryFunction = twoClassSummary,
#                      verboseIter = TRUE)
# 
# # Tune the model
# set.seed(1256)
# pima.tune <- train(
#   x = subset(pima, select = -diabetes),
#   y = pima$diabetes,
#   method = "nnet",
#   trace = FALSE,
#   maxit = 2000,
#   metric = "ROC",
#   trControl = ctrl,
#   tuneLength = 5
# )
# plot(pima.tune)  # plot tuning results
# 
# # Compute partial dependence for each predictor
# pd.all <- NULL
# for (i in 1:length(xnames)) {
#   pd <- partial(pima.tune, pred.var = xnames[i])
#   pd <- cbind(xnames[i], pd)
#   names(pd) <- c("Feature", "X", "Y")
#   pd.all <- rbind(pd.all, pd)
# }
# 
# # Figure 9
# pdf(file = "pima-pdps.pdf", width = 12, height = 4)
# ggplot(pd.all, aes(x = X, y = Y)) +
#   geom_line() +
#   facet_grid( ~ Feature, scales = "free_x") +
#   xlab("") +
#   ylab("Partial dependence") +
#   theme_light()
# dev.off()
# 
# # Figure 10
# xnames <- names(subset(pima, select = -diabetes))
# # pdf(file = "pima-vip.pdf", width = 7, height = 5)
# vip(pima.tune, pred.var = xnames)
# # dev.off()



################################################################################
# Example from ICE curve paper
################################################################################

# set.seed(101)
# n <- 1000
# x1 <- runif(n, min = -1, max = 1)
# x2 <- runif(n, min = -1, max = 1)
# x3 <- runif(n, min = -1, max = 1)
# y <- 0.2 * x1 - 5 * x2 + 10 * ifelse(x3 >= 0, x2, 0) + rnorm(n, sd = 0.1)
# plot(x2, y)
# d <- data.frame(x1, x2, x3, y)
# 
# library(gbm)
# set.seed(102)
# fit <- gbm(y ~ ., data = d, distribution = "gaussian", n.trees = 10000, 
#            interaction.depth = 3, cv.folds = 5, shrinkage = 0.1,
#            bag.fraction = 1, train.fraction = 1, verbose = TRUE)
# best.iter <- gbm.perf(fit, method = "cv")
# print(best.iter)
# 
# library(pdp)
# pd.2 <- partial(fit, pred.var = "x2", n.trees = best.iter)
# plotPartial(pd.2, ylim = c(-6, 6))
# 
# partial(fit, pred.var = "x2", n.trees = best.iter, ice = TRUE, plot = TRUE,
#         progress = "text")
# 
# partial(fit, pred.var = c("x2", "x3"), plot = TRUE, n.trees = best.iter)
# 
# pd.23 <- partial(fit, pred.var = c("x2", "x3"), n.trees = best.iter)
# plotPartial(pd.23, levelplot = FALSE, scales = list(arrows = FALSE),
#             drape = TRUE, colorkey = TRUE,
#             screen = list(z = -20, x = -60))
# 
# vint <- function(x) {
#   pd <- partial(fit, pred.var = c(x[1L], x[2L]), n.trees = best.iter)
#   c(sd(tapply(pd$yhat, INDEX = pd[[x[1L]]], FUN = sd)),
#     sd(tapply(pd$yhat, INDEX = pd[[x[2L]]], FUN = sd)))
# }
# combns <- t(combn(paste0("x", 1:3), m = 2))
# int.i <- plyr::aaply(combns, .margins = 1, .fun = vint, .progress = "text")
# int.i <- data.frame(x = paste0(combns[, 1L], "*", combns[, 2L]), 
#                     y = rowMeans(int.i))
# int.i <- int.i[order(int.i$y, decreasing = TRUE), ]
# dotchart(int.i$y, labels = int.i$x, pch = 19)
# 
# # Using GBM
# interact.gbm(fit, data = d, i.var = c("x1", "x2"), n.trees = best.iter)
# interact.gbm(fit, data = d, i.var = c("x1", "x3"), n.trees = best.iter)
# interact.gbm(fit, data = d, i.var = c("x2", "x3"), n.trees = best.iter)
