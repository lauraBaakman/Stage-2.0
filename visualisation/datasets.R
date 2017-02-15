# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");

# Load libraries
library.path <- cat(.libPaths())
library(scatterplot3d, lib.loc = library.path)

plotDataSet <- function(data, outputFile, numberOfPatternsPerSubSet){
  if(length(colours) < length(numberOfPatternsPerSubSet)){
    message = sprintf('The number of subsets (%d) should be greater than or equal to the number of defined colours (%d).', length(numberOfPatternsPerSubSet), length(colours));
    stop(message)
  }
  patternColours = generateColours(numberOfPatternsPerSubSet);
  printf("outputFile: %s\n", outputFile)
  pdf(outputFile);
  scatterplot3d(
    x = data$x, y = data$y, z = data$z,
    xlab='x', ylab='y', zlab='z',
    pch='.',
    color=patternColours,
    grid=FALSE,
    lty.hide=4,
    cex.lab=2.5,
    label.tick.marks=FALSE,
    mar=c(2.4, 3, 0, 2),
    cex.symbols = 2
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