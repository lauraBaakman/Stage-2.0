# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");
source("./results.R");

# Load libraries

# FileNames
data_set_file = "../data/simulated/normal/baakman_1_60000.txt"
parzen_file = "../results/normal/silverman/baakman_1_60000_parzen.txt"
mbe_file = "../results/normal/silverman/baakman_1_60000_mbe_silverman.txt"
sambe_file = "../results/normal/silverman/baakman_1_60000_sambe_silverman.txt"

readResultSet <- function(data_set_file, parzen_file, mbe_file, sambe_file){
  # read dataset file
  list[data , trueDensities, ] = readDataSet(data_set_file);
  data$trueDensities = trueDensities;
  
  # read parzen file
  parzen_data = readResults(parzen_file);
  data$parzenDensities = parzen_data$computedDensity;
  data$parzenNumUsedPatterns = parzen_data$numUsedPatterns;
  
  # read mbe file
  mbe_data = readResults(mbe_file);
  data$mbeDensities = mbe_data$computedDensity;
  data$mbeNumUsedPatterns = mbe_data$numUsedPatterns;  
  
  # read sambe file
  sambe_data = readResults(sambe_file);
  data$sambeDensities = sambe_data$computedDensity;
  data$sambeNumUsedPatterns = sambe_data$numUsedPatterns;  
  
  data;
}

computeLimits<-function(data){
  yMin = min(data$trueDensities, data$mbeDensities, data$sambeDensities);
  yMax = max(data$trueDensities, data$mbeDensities, data$sambeDensities);
  
  xMin = min(data$trueDensities);
  xMax = max(data$trueDensities);
  
  c(xMin = xMin, xMax = xMax,
    yMin = yMin, yMax = yMax);
}

generateResultPlot <- function(data, computedDensities, outPath){
  plotData <- data.frame(
    points.x=data$x, points.y=data$y, points.z=data$z,
    trueDensity=data$trueDensities, computedDensity=computedDensities);
  
  distribution <- table(data$component);
  
  limits <- computeLimits(data);
  
  plot <- plotResult(plotData, outPath, distribution, limits);
}

# Execute on Source
data <- readResultSet(data_set_file, parzen_file, mbe_file, sambe_file)
# generateResultPlot(data, data$sambeDensities, "~/Desktop/temp.png")