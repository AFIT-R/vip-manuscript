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
library(gbm)             # for fitting generalized boosted regression models
library(NeuralNetTools)  # for Garson and Olden's algorithms
library(nnet)            # for fitting neural networks
library(pdp)             # for constructing partial dependence plots
library(randomForest)    # for fitting random forests
library(vip)             # for constructing variable importance plots

# Load data sets
# data(boston, package = "pdp")
# data(pima, package = "pdp")
ames <- read.csv("ames.csv", header = TRUE)[, -1L]  # rm ID column

# Colors
set1 <- RColorBrewer::brewer.pal(9, "Set1")


################################################################################
# Section 2.4: Ames housing data
################################################################################

# Fit a generalized boosted regression model
set.seed(1138)
ames.gbm <- gbm(log(SalePrice) ~ ., data = ames,
                distribution = "gaussian",
                n.trees = 5000,
                interaction.depth = 6,
                shrinkage = 0.01,
                bag.fraction = 1,
                train.fraction = 1,
                cv.folds = 5,
                verbose = TRUE)

# Compute "optimal" number of iterations based on CV results
best.iter <- gbm.perf(ames.gbm, method = "cv")
print(best.iter)

# Plot relative influence of each predictor
summary(ames.gbm, n.trees = best.iter)

# Figure 1
# pred.var <- as.character(vi(ames.gbm)[1L:20L, ]$Variable)
pdf(file = "ames-gbm-vip.pdf", width = 7, height = 5)
# vip(ames.gbm, pred.var = pred.var, n.trees = best.iter)
vip(ames.gbm, n.trees = best.iter)
dev.off()

# Partial depence plots
ames.ri <- vi(ames.gbm, partial = FALSE, n.trees = best.iter)
ames.vi <- vi(ames.gbm, partial = TRUE, keep.partial = TRUE, 
              n.trees = best.iter)
ames.pd <- attr(ames.vi, "partial")[as.character(ames.ri$Variable[1L:16L])]
ames.pd <- plyr::ldply(ames.pd, .id = "x.name", .fun = function(x) {
  names(x)[1L] <- "x.value"
  x
})
p <- ggplot(ames.pd, aes(x = x.value, y = yhat)) +
  geom_line() +
  # geom_point(size = 0.5) +
  # geom_smooth(se = FALSE, linetype = "dashed") +
  geom_hline(yintercept = mean(log(ames$SalePrice)), linetype = "dashed") +
  facet_wrap( ~ x.name, scales = "free_x") +
  theme_light() +
  xlab("") +
  ylab("Partial dependence")
p


################################################################################
# Section 3: A partial dependence-based variable importance measure
################################################################################

# # Partial dependence-based variable importance scores
# boston.rf.vi <- vi(boston.rf, pred.var = names(subset(boston, select = -cmedv)))
# p <- vip(ames.gbm, partial = TRUE, n.trees = best.iter)

# Figure 3
# pdf(file = "ames-gbm-vip-pd.pdf", width = 7, height = 4)
p
# dev.off()


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
# Section 5: The Pima Indians diabetes data
################################################################################

# Load data
data(pima, package = "pdp")
pima <- na.omit(pima)

# Setup for repeated k-fold cross-validation
ctrl <- trainControl(method = "repeatedcv", number = 5, repeats = 10, 
                     classProbs = TRUE, summaryFunction = twoClassSummary,
                     verboseIter = TRUE)

# Tune the model
set.seed(1256)
pima.tune <- train(
  x = subset(pima, select = -diabetes),
  y = pima$diabetes,
  method = "nnet",
  trace = FALSE,
  maxit = 2000,
  metric = "ROC",
  trControl = ctrl,
  tuneLength = 5
)
plot(pima.tune)  # plot tuning results

# Compute partial dependence for each predictor
pd.all <- NULL
for (i in 1:length(xnames)) {
  pd <- partial(pima.tune, pred.var = xnames[i])
  pd <- cbind(xnames[i], pd)
  names(pd) <- c("Feature", "X", "Y")
  pd.all <- rbind(pd.all, pd)
}

# Figure 9
pdf(file = "pima-pdps.pdf", width = 12, height = 4)
ggplot(pd.all, aes(x = X, y = Y)) +
  geom_line() +
  facet_grid( ~ Feature, scales = "free_x") +
  xlab("") +
  ylab("Partial dependence") +
  theme_light()
dev.off()

# Figure 10
xnames <- names(subset(pima, select = -diabetes))
# pdf(file = "pima-vip.pdf", width = 7, height = 5)
vip(pima.tune, pred.var = xnames)
# dev.off()



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
