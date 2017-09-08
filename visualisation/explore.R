# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");
source("./results.R");

# Load libraries
library(scatterplot3d)

# FileNames
data_set_file = "../data/simulated/normal/baakman_4_60000.txt"
parzen_file = "../results/normal/baakman_4_60000_parzen.txt"
mbe_file = "../results/normal/baakman_4_60000_mbe_silverman.txt"
sambe_file = "../results/normal/baakman_4_60000_sambe_silverman.txt"

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

addXisResults <- function(xsdata, sambe_file){
  # read parzen file
  # Only contains x, y, z that can also be found in sambe_file.
  
  # read mbe file
  # Only contains x, y, z, local bandwidths, can also be found in sambe_file
  
  # read sambe file
  xisdata = readXisData(sambe_file);
  
  data <- merge(xsdata, xisdata, by=c("x","y", "z"));
}

computeLimits<-function(data){
  yMin = min(data$trueDensities, data$mbeDensities, data$sambeDensities);
  yMax = max(data$trueDensities, data$mbeDensities, data$sambeDensities);
  
  xMin = min(data$trueDensities);
  xMax = max(data$trueDensities);
  
  c(xMin = xMin, xMax = xMax,
    yMin = yMin, yMax = yMax);
}

computeLimits2<-function(data){
  minimum = min(data$mbeDensities, data$sambeDensities);
  maximum = max(data$mbeDensities, data$sambeDensities);
  
  c(xMin = minimum, xMax = maximum,
    yMin = minimum, yMax = maximum);  
}

generateResultPlot <- function(data, computedDensities, outPath){
  plotData <- data.frame(
    points.x=data$x, points.y=data$y, points.z=data$z,
    trueDensity=data$trueDensities, computedDensity=computedDensities);
  
  distribution <- table(data$component);
  
  limits <- computeLimits(data);
  
  plot <- plotResult(plotData, outPath, distribution, limits);
}

generateMBEvsSAMBEPlot <- function(data, outPath){
  plotData <- data.frame(
    points.x=data$x, points.y=data$y, points.z=data$z,
    trueDensity=data$mbeDensities, computedDensity=data$sambeDensities);
  
  distribution <- table(data$component);  
  limits <- computeLimits2(data);
  
  plot <- plotResult(plotData, outPath, distribution, limits, xlabel='MBE', ylabel='SAMBE', addMSE = FALSE);
}

mse <-function(trueDensities, estimatedDensities){
  error <- trueDensities - estimatedDensities;
  mse <- mean(error^2)
  mse;
}

componentMSE<-function(data, componentNumber){
  component1 <- data[data$component == componentNumber, ]
  printf("Component %d (MBE, SAMBE):\n %s\t& %s\n", 
         componentNumber,
         formatC(mse(component1$trueDensities, component1$mbeDensities), digits = 15, format = "e"),
         formatC(mse(component1$trueDensities, component1$sambeDensities), digits = 15, format = "e")); 
}

distanceToEigenValueMean<-function(data){
  meanEigenValue = (data$eigen_value_1 + data$eigen_value_2 + data$eigen_value_3) / 3;
  meanDifferences = (
    (data$eigen_value_1 - meanEigenValue) + 
      (data$eigen_value_2 - meanEigenValue) + 
      (data$eigen_value_3 - meanEigenValue)
    ) / 3;
}

ferdosi1<-function(){
  xsdata <- readResultSet(
    data_set_file="../data/simulated/normal/ferdosi_1_60000.txt", 
    parzen_file="../results/normal/ferdosi_1_60000_parzen.txt",
    mbe_file="../results/normal/ferdosi_1_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/ferdosi_1_60000_sambe_silverman.txt"
  ) 
  data <- addXisResults(
    xsdata = xsdata,
    sambe_file="../results/normal/ferdosi_1_60000_sambe_silverman_xis.txt"    
  )
  data$mbeSambeDiff = data$mbeDensities - data$sambeDensities;
  data$meanEigDiff = distanceToEigenValueMean(data);

  generateMBEvsSAMBEPlot(data, "../paper/discussion/img/ferdosi_1_60000_mbe_sambe.png")
  plotShapeAdaptedData(data, "../paper/discussion/img/ferdosi_1_60000_pointsWithShapeAdaptedKernels.pdf", minDifference = 1.5)
  
  componentMSE(data, 0);
  componentMSE(data, 1);

    # noise = data[data$component == 1, ] 
  # gaussian=data[data$component == 0, ]
  
  data;
}

ferdosi2<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/ferdosi_2_60000.txt", 
    parzen_file="../results/normal/ferdosi_2_60000_parzen.txt",
    mbe_file="../results/normal/ferdosi_2_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/ferdosi_2_60000_sambe_silverman.txt"
  )    
  data <- addXisResults(
    xsdata = data,
    sambe_file="../results/normal/ferdosi_2_60000_sambe_silverman_xis.txt"    
  )
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
  
  data;
}

ferdosi3<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/ferdosi_3_120000.txt", 
    parzen_file="../results/normal/ferdosi_3_120000_parzen.txt",
    mbe_file="../results/normal/ferdosi_3_120000_mbe_silverman.txt", 
    sambe_file="../results/normal/ferdosi_3_120000_sambe_silverman.txt"
  )   
  data <- addXisResults(
    xsdata = data,
    sambe_file="../results/normal/ferdosi_3_120000_sambe_silverman_xis.txt"    
  )
  
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
  componentMSE(data, 3);
  componentMSE(data, 4);
  
  data;
}

baakman2<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_2_60000.txt", 
    parzen_file="../results/normal/baakman_2_60000_parzen.txt",
    mbe_file="../results/normal/baakman_2_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/baakman_2_60000_sambe_silverman.txt"
  )   
  data <- addXisResults(
    xsdata = data,
    sambe_file="../results/normal/baakman_2_60000_sambe_silverman_xis.txt"    
  )
  
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
  
  data;
}

baakman3<-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_3_120000.txt", 
    parzen_file="../results/normal/baakman_3_120000_parzen.txt",
    mbe_file="../results/normal/baakman_3_120000_mbe_silverman.txt", 
    sambe_file="../results/normal/baakman_3_120000_sambe_silverman.txt"
  )    
  data <- addXisResults(
    xsdata = data,
    sambe_file="../results/normal/baakman_3_120000_sambe_silverman_xis.txt"    
  )
  
  componentMSE(data, 0);  
  componentMSE(data, 1);
  componentMSE(data, 2);
  componentMSE(data, 3);
  componentMSE(data, 4);
  
  data;
}

baakman5 <-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_5_60000.txt", 
    parzen_file="../results/normal/baakman_5_60000_parzen.txt",
    mbe_file="../results/normal/baakman_5_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/baakman_5_60000_sambe_silverman.txt"
  )  
  data <- addXisResults(
    xsdata = data,
    sambe_file="../results/normal/baakman_5_60000_sambe_silverman_xis.txt"    
  )  
  
  data$mbeSambeDiff = data$mbeDensities - data$sambeDensities;
  data$meanEigDiff = distanceToEigenValueMean(data);
  
  generateMBEvsSAMBEPlot(data, "../paper/discussion/img/baakman_5_60000_mbe_sambe.png")
  # plotShapeAdaptedData(data, "../paper/discussion/img/baakman_5_60000_pointsWithShapeAdaptedKernels.pdf")
  
  componentMSE(data, 0);
  componentMSE(data, 1); 
  
  data;
}

plotShapeAdaptedData <- function(allData, outputFile='~/Desktop/shapeAdapted.pdf', minDifference = 0.8){
  data <- allData[(allData$eigen_value_1 - allData$eigen_value_2) > minDifference | 
                  (allData$eigen_value_1 - allData$eigen_value_2) > minDifference |
                  (allData$eigen_value_2 - allData$eigen_value_3) > minDifference, ]
  # Plot The Actual Data
  allData = allData[order(allData$component, decreasing = FALSE), ]
  distribution = table(allData$component);
  theColors = add.alpha(generateColours(distribution), alpha = 0.15);
  pdf(outputFile);
  s3d <- scatterplot3d(
    x = allData$x, y = allData$y, z = allData$z,
    xlab='x', ylab='y', zlab='z',
    pch=16,
    color=theColors,
    grid=FALSE,
    lty.hide=4,
    mar=c(2.4, 3, 0, 2),
    type='p',
    cex.symbols = 0.4
  )  
  # Plot the points of interest
  data = data[order(data$component, decreasing = FALSE), ]
  distribution = table(data$component);
  theColors = add.alpha(generateColours(distribution), alpha = 0.5);  
  s3d$points3d(x=data$x, y=data$y, z=data$z,
    pch=16,
    col=theColors
  );
  dev.off();
}

baakman4 <-function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_4_60000.txt", 
    parzen_file="../results/normal/baakman_4_60000_parzen.txt",
    mbe_file="../results/normal/baakman_4_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/baakman_4_60000_sambe_silverman.txt"
  )  
  data <- addXisResults(
    xsdata = data,
    sambe_file="../results/normal/baakman_4_60000_sambe_silverman_xis.txt"    
  )    
  data$mbeSambeDiff = data$mbeDensities - data$sambeDensities;
  data$meanEigDiff = distanceToEigenValueMean(data);
  
  generateMBEvsSAMBEPlot(data, "~/Desktop/baakman_4_60000_mbe_sambe.png");
  componentMSE(data, 0);
  componentMSE(data, 1);  
  
  data;
}

baakman1 <- function(){
  data <- readResultSet(
    data_set_file="../data/simulated/normal/baakman_1_60000.txt", 
    parzen_file="../results/normal/baakman_1_60000_parzen.txt",
    mbe_file="../results/normal/baakman_1_60000_mbe_silverman.txt", 
    sambe_file="../results/normal/baakman_1_60000_sambe_silverman.txt"
  )
  data <- addXisResults(
    xsdata = data,
    sambe_file="../results/normal/baakman_1_60000_sambe_silverman_xis.txt"    
  )     
  
  data$mbeSambeDiff = data$mbeDensities - data$sambeDensities;
  data$meanEigDiff = distanceToEigenValueMean(data);
  
  generateMBEvsSAMBEPlot(data, "../paper/discussion/img/baakman_1_60000_mbe_sambe.png")
  # plotShapeAdaptedData(data, "../paper/discussion/img/baakman_1_60000_pointsWithShapeAdaptedKernels.pdf")
  
  componentMSE(data, 0);
  componentMSE(data, 1);  
  
  data;
}

# Execute on Source
# data <- readResultSet(data_set_file, parzen_file, mbe_file, sambe_file)
# generateResultPlot(data, data$sambeDensities, "~/Desktop/temp.png")
# formatC(min(data$sambeDensities), digits = 15, format = "e")
# head(data[order(data$sambeDensities, decreasing = TRUE), ], n=10)

# head(data[order(data$meanEigDiff, decreasing = TRUE), c('x', 'y', 'z','eigen_value_1', 'eigen_value_2', 'eigen_value_3', 'local_bandwidth', 'scaling_factor')], n = 10)
# head(data[order(abs(data$mbeSambeDiff), decreasing=TRUE), c('x', 'y', 'z', 'component', 'trueDensities', 'mbeDensities', 'mbeNumUsedPatterns', 'sambeDensities', 'sambeNumUsedPatterns', 'mbeSambeDiff')], 20)