# Load required packages
library(ggplot2)
library(h2o)
library(pdp)

# Initialize and connect to H2O
h2o.init(nthreads = -1, max_mem_size = "4g")

# Load the data
url <- paste0("http://archive.ics.uci.edu/ml/machine-learning-databases/",
              "00291/airfoil_self_noise.dat")
airfoil <- read.table(url, header = FALSE)
names(airfoil) <- c(
  "frequency", 
  "angle_of_attack", 
  "chord_length", 
  "free_stream_velocity", 
  "suction_side_displacement_thickness", 
  "scaled_sound_pressure_level"
)

# Setup
x <- names(airfoil)[1L:5L]
y <- names(airfoil)[6L]
trn <- as.h2o(airfoil)

# Auto ML
aml <- h2o.automl(
  x = x, 
  y = y,
  training_frame = trn,
  max_runtime_secs = 60 * 15,
  stopping_metric = "RMSE",
  seed = 1347
)
h2o.saveModel(aml, path = "C:\\Users\\greenweb\\Desktop\\")
lb <- as.data.frame(aml@leaderboard)
save(lb, file = "C:\\Users\\greenweb\\Desktop\\aml_lb.RData")

# # Partial dependence
# pfun <- function(object, newdata) {
#   nd <- as.h2o(newdata)
#   mean(h2o.predict(object, newdata = nd))
# }
# pd_list <- lapply(x, FUN = function(xx) {
#   partial(aml, pred.var = xx, train = airfoil, pred.fun = pfun, progress = "text")
# })
# names(pd_list) <- x
# pd_df <- plyr::ldply(pd_list, .id = "x_name", .fun = function(x) {
#   names(x) <- c("x_value", "y_value")
#   x
# })
# 
# # Save results
# save(lb, pd_list, pd_df, 
#      file = "C:\\Users\\greenweb\\Desktop\\aml_results.RData")

# What types of models were stacked together?
table(stringi::stri_extract(str = lb$model_id, regex = "^([A-Z]|[a-z])*_"))

# The Automatic Machine Learning (AutoML) function automates the supervised 
# machine learning model training process. The current version of AutoML trains 
# and cross-validates a Random Forest, an Extremely-Randomized Forest, a random 
# grid of Gradient Boosting Machines (GBMs), a random grid of Deep Neural Nets, 
# and then trains a Stacked Ensemble using all of the models.
#
# For this example, the H2O AutoML algorithm fit one RF, one XRT, a random grid
# of 376 GBMs, and a random grid of four DNNs. The final stacked ensemble 
# acheieved a five-fold cross-validated RMSE and R-squared of 1.7131835 and 
# 0.93870246, respectively.

# x-axis labels
lbls <- names(airfoil) <- c(
  "Frequency", 
  "Angle of attack", 
  "Chord length", 
  "Free stream velocity", 
  "Suction side displacement thickness", 
  "Scaled sound pressure level"
)

# Load partial dependence data
load("aml_results.RData")

# Construct display of partial dependence plots
ylim <- range(unlist(lapply(pd_list, FUN = function(x) range(x$yhat))))
p1 <- ggplot(pd_list[[1L]], aes(x = pd_list[[1L]][[1L]], y = pd_list[[1L]][[2L]])) +
  geom_line() +
  # geom_point() +
  theme_light() +
  xlab(lbls[1L]) +
  ylab("Partial dependence") +
  ylim(ylim)
p2 <- ggplot(pd_list[[2L]], aes(x = pd_list[[2L]][[1L]], y = pd_list[[2L]][[2L]])) +
  geom_line() +
  # geom_point() +
  theme_light() +
  xlab(lbls[2L]) +
  ylab("Partial dependence") +
  ylim(ylim)
p3 <- ggplot(pd_list[[3L]], aes(x = pd_list[[3L]][[1L]], y = pd_list[[3L]][[2L]])) +
  geom_line() +
  # geom_point() +
  theme_light() +
  xlab(lbls[3L]) +
  ylab("Partial dependence") +
  ylim(ylim)
p4 <- ggplot(pd_list[[4L]], aes(x = pd_list[[4L]][[1L]], y = pd_list[[4L]][[2L]])) +
  geom_line() +
  # geom_point() +
  theme_light() +
  xlab(lbls[4L]) +
  ylab("Partial dependence") +
  ylim(ylim)
p5 <- ggplot(pd_list[[5L]], aes(x = pd_list[[5L]][[1L]], y = pd_list[[5L]][[2L]])) +
  geom_line() +
  # geom_point() +
  theme_light() +
  xlab(lbls[5L]) +
  ylab("Partial dependence") +
  ylim(ylim)
pdf(file = "C:\\Users\\greenweb\\Desktop\\aml_pdps.pdf", width = 15, height = 3)
gridExtra::grid.arrange(p1, p5, p3, p4, p2, ncol = 5)
dev.off()

# Variable importance plot
aml_vi <- sort(unlist(lapply(pd_list, FUN = function(x) sd(x$yhat))))
aml_vi <- data.frame("Variable" = names(aml_vi),
                     "Importance" = aml_vi)
rownames(aml_vi) <- NULL
p6 <- ggplot(aml_vi, aes(x = Variable, y = Importance)) +
  geom_col() +
  xlab("") +
  theme_light()

gridExtra::grid.arrange(p1, p5, p3, p4, p2, p6, ncol = 3)

