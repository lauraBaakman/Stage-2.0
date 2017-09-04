# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");
source("./results.R");

# Load libraries

# FileNames
data_set_file = "../data/simulated/normal/baakman_4_60000.txt"
parzen_file = "../results/normal/silverman/baakman_4_60000_parzen.txt"
mbe_file = "../results/normal/silverman/baakman_4_60000_mbe_silverman.txt"
sambe_file = "../results/normal/silverman/baakman_4_60000_sambe_silverman.txt"

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

mse <-function(trueDensities, estimatedDensities){
  error <- trueDensities - estimatedDensities;
  mse <- mean(error^2)
  mse;
}

componentMSE<-function(data, componentNumber){
  component1 <- data[data$component == componentNumber, ]
  printf("Component %d:\n %s\t& %s\n", 
         componentNumber,
         formatC(mse(component1$trueDensities, component1$mbeDensities), digits = 15, format = "e"),
         formatC(mse(component1$trueDensities, component1$sambeDensities), digits = 15, format = "e")); 
}

ferdosi1<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/ferdosi_1_60000.txt", 
    parzen_file="../results/normal/silverman/ferdosi_1_60000_parzen.txt", 
    mbe_file="../results/normal/silverman/ferdosi_1_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/ferdosi_1_60000_sambe_silverman.txt"
  )    
}

ferdosi2<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/ferdosi_2_60000.txt", 
    parzen_file="../results/normal/silverman/ferdosi_2_60000_parzen.txt", 
    mbe_file="../results/normal/silverman/ferdosi_2_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/ferdosi_2_60000_sambe_silverman.txt"
  )    
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
}

ferdosi3<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/ferdosi_3_120000.txt", 
    parzen_file="../results/normal/silverman/ferdosi_3_120000_parzen.txt", 
    mbe_file="../results/normal/silverman/ferdosi_3_120000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/ferdosi_3_120000_sambe_silverman.txt"
  )    
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
  componentMSE(data, 3);
  componentMSE(data, 4);
}

baakman2<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_2_60000.txt", 
    parzen_file="../results/normal/silverman/baakman_2_60000_parzen.txt", 
    mbe_file="../results/normal/silverman/baakman_2_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/baakman_2_60000_sambe_silverman.txt"
  )    
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
}

baakman3<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_3_120000.txt", 
    parzen_file="../results/normal/silverman/baakman_3_120000_parzen.txt", 
    mbe_file="../results/normal/silverman/baakman_3_120000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/baakman_3_120000_sambe_silverman.txt"
  )    
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
  componentMSE(data, 3);
  componentMSE(data, 4);
}

baakman5 <-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_5_60000.txt", 
    parzen_file="../results/normal/silverman/baakman_5_60000_parzen.txt", 
    mbe_file="../results/normal/silverman/baakman_5_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/baakman_5_60000_sambe_silverman.txt"
  )  
  
  # Remove too low densities
  data <- data[data$sambeDensities > 0.0, ];
  # Remove too high densities
  data <- data[data$sambeDensities < 0.3, ];               
  
  # Generate plots
  generateResultPlot(data, data$mbeDensities, "~/Desktop/results_baakman_5_60000_mbe_silverman_no_outliers.png")
  generateResultPlot(data, data$sambeDensities, "~/Desktop/results_baakman_5_60000_sambe_silverman_no_outliers.png")  
}

baakman4 <-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_4_60000.txt", 
    parzen_file="../results/normal/silverman/baakman_4_60000_parzen.txt", 
    mbe_file="../results/normal/silverman/baakman_4_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/baakman_4_60000_sambe_silverman.txt"
  )  
  
  # Remove too low densities
  data <- data[data$sambeDensities > 0.0, ];
  # Remove too high densities
  data <- data[data$sambeDensities < 0.15, ];               
  
  # Generate plots
  generateResultPlot(data, data$mbeDensities, "~/Desktop/results_baakman_4_60000_mbe_silverman_no_outliers.png")
  generateResultPlot(data, data$sambeDensities, "~/Desktop/results_baakman_4_60000_sambe_silverman_no_outliers.png")  
}

baakman1 <- function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_1_60000.txt", 
    parzen_file="../results/normal/silverman/baakman_1_60000_parzen.txt", 
    mbe_file="../results/normal/silverman/baakman_1_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/silverman/baakman_1_60000_sambe_silverman.txt"
  )
  
  # Remove too low densities
  data <- data[data$sambeDensities > 0.0, ];
  # Remove too high densities
  data <- data[data$sambeDensities < 0.07, ];               
  
  # Generate plots
  generateResultPlot(data, data$mbeDensities, "~/Desktop/results_baakman_1_60000_mbe_silverman_no_outliers.png")
  generateResultPlot(data, data$sambeDensities, "~/Desktop/results_baakman_1_60000_sambe_silverman_no_outliers.png")
}

# Execute on Source
# data <- readResultSet(data_set_file, parzen_file, mbe_file, sambe_file)
# generateResultPlot(data, data$sambeDensities, "~/Desktop/temp.png")
# formatC(min(data$sambeDensities), digits = 15, format = "f")
# head(data[order(data$sambeDensities, decreasing = TRUE), ], n=10)