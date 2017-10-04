################################################################################
# Setup
################################################################################

# Load required packages
library(caret)  # also loads ggplot2
library(dplyr)
library(gbm)
library(nnet)
library(pdp)
library(randomForest)
library(vip)

# Load the (corrected) Boston housing data
data(boston, package = "pdp")


################################################################################
# Boston housing example: random forest
################################################################################

# Fit a random forest to the Boston Housing data (mtry was tuned using cross-
# validation)
set.seed(101)
boston.rf <- randomForest(cmedv ~ ., data = boston, mtry = 6, ntree = 1000,
                          importance = TRUE)

# Figure ?
imp1 <- importance(boston.rf, type = 1) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Variable")
imp2 <- importance(boston.rf, type = 2) %>%
  as.data.frame() %>%
  tibble::rownames_to_column("Variable")
p1 <- ggplot(imp1, aes(x = reorder(Variable, `%IncMSE`), y = `%IncMSE`)) +
  geom_col() +
  coord_flip() +
  xlab("") +
  theme_light()
p2 <- ggplot(imp2, aes(x = reorder(Variable, IncNodePurity), y = IncNodePurity)) +
  geom_col() +
  coord_flip() +
  xlab("") +
  theme_light()
pdf(file = "boston-rf-vip.pdf", width = 8, height = 5)
grid.arrange(p1, p2, ncol = 2)
dev.off()


# Partial dependence plots for lstat, rm, and zn
pd1 <- partial(boston.rf, pred.var = "lstat")
pd2 <- partial(boston.rf, pred.var = "rm")
pd3 <- partial(boston.rf, pred.var = "zn")
pd.range <- range(c(pd1$yhat, pd2$yhat, pd3$yhat))
p1 <- autoplot(pd1) +
  ylim(pd.range[1L], pd.range[2L]) +
  theme_light() +
  geom_hline(yintercept = mean(boston$cmedv), linetype = 2, col = set1[1L],
             alpha = 0.5) +
  ylab("Partial dependence")
p2 <- autoplot(pd2) +
  ylim(pd.range[1L], pd.range[2L]) +
  theme_light() +
  geom_hline(yintercept = mean(boston$cmedv), linetype = 2, col = set1[1L],
             alpha = 0.5) +
  ylab("Partial dependence")
p3 <- autoplot(pd3) +
  ylim(pd.range[1L], pd.range[2L]) +
  theme_light() +
  geom_hline(yintercept = mean(boston$cmedv), linetype = 2, col = set1[1L],
             alpha = 0.5) +
  ylab("Partial dependence")

# Figure ?
pdf(file = "boston-rf-pdps.pdf", width = 12, height = 4)
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

# Variable importance scores (partial dependence)
boston.rf.vi <- vi(boston.rf, pred.var = names(subset(boston, select = -cmedv)))
p <- ggplot(boston.rf.vi, aes(x = reorder(Variable, -Importance), y = Importance)) +
  geom_col() +
  xlab("") +
  theme_light()

# Variable importance plots
p1 <- vip(boston.rf, pred.var = names(subset(boston, select = -cmedv)), FUN = sd)
p2 <- vip(boston.rf, pred.var = names(subset(boston, select = -cmedv)), FUN = mad)

# Figure ?
pdf(file = "boston-rf-vip-pd.pdf", width = 7, height = 4)
print(p)
dev.off()


################################################################################
# Boston housing example: neural network
################################################################################

ctrl <- trainControl(method = "cv", number = 5, verboseIter = TRUE)
set.seed(4578)
boston.nn.tune <- train(
  x = subset(boston, select = -cmedv),
  y = boston$cmedv,
  method = "nnet",
  trace = FALSE,
  linout = TRUE,
  maxit = 1000,
  trControl = ctrl,
  tuneGrid = expand.grid(size = 1:20, decay = c(0, 0.0001, 0.001, 0.01, 0.1))
)
plot(boston.nn.tune)

# Fit a neural network with "optimal" tuning parameters
set.seed(301)
boston.nn <- nnet(cmedv ~ ., data = boston, size = 15, decay = 0.1, 
                  linout = TRUE, maxit = 10000)

# Variable importance plots
vip(boston.nn, pred.var = names(subset(boston, select = -cmedv)), 
    quantiles = TRUE, probs = 10:90/100)
vip(boston.nn, pred.var = names(subset(boston, select = -cmedv)), FUN = var)
vip(boston.nn, pred.var = names(subset(boston, select = -cmedv)), FUN = IQR)

# Variable importance plots with custom metric
pred.var <- names(subset(boston, select = -cmedv))
vip(boston.rf, pred.var = pred.var, FUN = function(x) {
  # max(abs(x - mean(boston$cmedv)))
  sqrt(mean((x - mean(boston$cmedv)) ^ 2))
})


################################################################################
# Friedman 1 data set
################################################################################

# Simulate the data
set.seed(101)  # for reproducibility
trn <- as.data.frame(mlbench::mlbench.friedman1(n = 500, sd = 1))


# Stochastic gradient boosting and Friedman's H-statistic ----------------------

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
dotchart(int.h$y, labels = int.h$x)

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

# Figure ?
pdf(file = "gbm-int.pdf", width = 7, height = 7)
grid.arrange(p1, p2, ncol = 2)
dev.off()


# Neural network ---------------------------------------------------------------

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

vip(trn.nn, pred.var = paste0("x.", 1:10), FUN = var)
vip(trn.nn, pred.var = paste0("x.", 1:10), FUN = mad)

# Figure ?
pdf(file = "network.pdf", width = 12, height = 6)
plotnet(trn.nn, circle_col = "lightgrey")
dev.off()

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

# Figure ?
pdf(file = "network-vip.pdf", width = 12, height = 6)
grid.arrange(p1, p2, p3, ncol = 3)
dev.off()

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

# Figure ?
pdf(file = "network-int.pdf", width = 8, height = 4)
labs <-  c(
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
dev.off()


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
# # Figure ?
# pdf(file = "interaction-simulation.pdf", width = 12, height = 8)
# grid.arrange(p1, p2, nrow = 2)
# dev.off()
# ------------------------------------------------------------------------------



################################################################################
# Linear model
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


varImp(fit)
vi(fit, use.partial = TRUE)
5 / 3  # absolute ratio of slopes


################################################################################
# Pima indians diabetes data
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

# Variable importance plots
xnames <- names(subset(pima, select = -diabetes))
pdf(file = "pima-vip.pdf", width = 7, height = 5)
vip(pima.tune, pred.var = xnames)
dev.off()

pd.all <- NULL
for (i in 1:length(xnames)) {
  pd <- partial(pima.tune, pred.var = xnames[i])
  pd <- cbind(xnames[i], pd)
  names(pd) <- c("Feature", "X", "Y")
  pd.all <- rbind(pd.all, pd)
}

# Figure ?
pdf(file = "pima-pdps.pdf", width = 12, height = 4)
ggplot(pd.all, aes(x = X, y = Y)) +
  geom_line() +
  facet_grid( ~ Feature, scales = "free_x") +
  xlab("") +
  ylab("Partial dependence") +
  theme_light()
dev.off()

# Compare to random forest
set.seed(0156)
pima.rf <- randomForest(diabetes ~ ., data = na.omit(pima), importance = TRUE)
plot(pima.rf)
varImpPlot(pima.rf)


################################################################################
# Example from ICE curve paper
################################################################################

set.seed(101)
n <- 1000
x1 <- runif(n, min = -1, max = 1)
x2 <- runif(n, min = -1, max = 1)
x3 <- runif(n, min = -1, max = 1)
y <- 0.2 * x1 - 5 * x2 + 10 * ifelse(x3 >= 0, x2, 0) + rnorm(n, sd = 0.1)
plot(x2, y)
d <- data.frame(x1, x2, x3, y)

library(gbm)
set.seed(102)
fit <- gbm(y ~ ., data = d, distribution = "gaussian", n.trees = 10000, 
           interaction.depth = 3, cv.folds = 5, shrinkage = 0.1,
           bag.fraction = 1, train.fraction = 1, verbose = TRUE)
best.iter <- gbm.perf(fit, method = "cv")
print(best.iter)

library(pdp)
pd.2 <- partial(fit, pred.var = "x2", n.trees = best.iter)
plotPartial(pd.2, ylim = c(-6, 6))

partial(fit, pred.var = "x2", n.trees = best.iter, ice = TRUE, plot = TRUE,
        progress = "text")

partial(fit, pred.var = c("x2", "x3"), plot = TRUE, n.trees = best.iter)

pd.23 <- partial(fit, pred.var = c("x2", "x3"), n.trees = best.iter)
plotPartial(pd.23, levelplot = FALSE, scales = list(arrows = FALSE),
            drape = TRUE, colorkey = TRUE,
            screen = list(z = -20, x = -60))

vint <- function(x) {
  pd <- partial(fit, pred.var = c(x[1L], x[2L]), n.trees = best.iter)
  c(sd(tapply(pd$yhat, INDEX = pd[[x[1L]]], FUN = sd)),
    sd(tapply(pd$yhat, INDEX = pd[[x[2L]]], FUN = sd)))
}
combns <- t(combn(paste0("x", 1:3), m = 2))
int.i <- plyr::aaply(combns, .margins = 1, .fun = vint, .progress = "text")
int.i <- data.frame(x = paste0(combns[, 1L], "*", combns[, 2L]), 
                    y = rowMeans(int.i))
int.i <- int.i[order(int.i$y, decreasing = TRUE), ]
dotchart(int.i$y, labels = int.i$x, pch = 19)

# Using GBM
interact.gbm(fit, data = d, i.var = c("x1", "x2"), n.trees = best.iter)
interact.gbm(fit, data = d, i.var = c("x1", "x3"), n.trees = best.iter)
interact.gbm(fit, data = d, i.var = c("x2", "x3"), n.trees = best.iter)
