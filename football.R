# Load required packages
library(h2o)

# Start H2O
h2o.init()

# Path to data sets
path <- "C:\\Users\\greenweb\\Desktop\\h2o-bk\\datasets\\"

# Load the football data
trn <- read.csv(paste0(path, "football.train2.csv"), header = TRUE)
val <- read.csv(paste0(path, "football.valid2.csv"), header = TRUE)
tst <- read.csv(paste0(path, "football.test2.csv"), header = TRUE)
