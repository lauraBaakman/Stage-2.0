# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");

# Load libraries
source('./scatterplot3d.R')

computeAxisLimit <- function(data){
  c(round(min(data)), round(max(data)))
}

plotDataSet <- function(data, outputFile, numberOfPatternsPerSubSet){
  if(length(colours) < length(numberOfPatternsPerSubSet)){
    message = sprintf('The number of subsets (%d) should be greater than or equal to the number of defined colours (%d).', length(numberOfPatternsPerSubSet), length(colours));
    stop(message)
  }
  patternColours = generateColours(numberOfPatternsPerSubSet);
  patternSymbols = generateSymbols(numberOfPatternsPerSubSet);
  printf("outputFile: %s\n", outputFile)
  pdf(outputFile);
  scatterplot3d(
    x = data$x, y = data$y, z = data$z,
    xlab='x', ylab='y', zlab='z',
    xlim = computeAxisLimit(data$x), ylim = computeAxisLimit(data$y), zlim = computeAxisLimit(data$z),
    pch=patternSymbols,
    color=patternColours,
    grid=FALSE, axis=TRUE, tick.marks=FALSE, label.tick.marks = FALSE,
    lty.hide=4,
    mar=c(2.4, 3, 0, 2),
    type='p',
    cex.symbols = 0.5, cex.lab = 3
  )
  dev.off();
}

plotDataSets <- function(dataSetFilePaths){
  # Plot the datasets in grids of 1 by 1.
  # resetPar();
  par(mfrow = c(1,1));
  
  for (file in dataSetFilePaths){
    list[data, , numberOfPatternsPerSubSet] = readDataSet(file);
    outputFile = outputFilePath(file, 'datasetplot_');
    plotDataSet(data, outputFile, numberOfPatternsPerSubSet);
  }
  printf("Wrote images of the datasets in %s to %s.\n", dirname(file), dirname(outputFile));
}

mainDataSets <- function(){
  plotDataSets(getDataSetPaths());
}