# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");

# Load libraries
library(ggplot2);
library(extrafont);
library(stringr);
library(gridExtra);

computeMSE <- function(data){
  error <- data$trueDensity - data$computedDensity;
  mse <- mean(error^2)
  mse;
}

isSingleDensityDataSet <- function(trueDensities){
  length(unique(trueDensities)) <= 5;
}

plotResultOfMultipleDensityDataSet <-function(data, outputFile, distribution, limits){
  cols = generateColours(distribution);
  
  plot <- ggplot(data) +
    theme(
      axis.ticks = element_blank(),
      axis.text = element_blank(),
      
      plot.title = element_text(family = font.family, size=font.size),
      
      text=element_text(family=font.family, size=font.size),
      
      panel.border = element_rect(colour = "black", fill=NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank()
    );
  plot <- plot + 	geom_point(aes(x=trueDensity, y=computedDensity), size=0.7, colour=cols, shape=21, stroke=0.2);
  plot <- plot + 	geom_line(aes(x=trueDensity, y=trueDensity));
  
  plot <- plot + 	xlab('true density') +
    ylab('computed density');
  
  plot <- plot + ggtitle(sprintf("MSE = %.7e", computeMSE(data)));
  
  plot <- plot + 	xlim(limits[[1]], limits[[2]]) +
    ylim(limits[[1]], limits[[2]]);
  #print(plot)
  ggsave(
    outputFile,
    plot,
    width=(tex.textwidth/2) - 1, height=(tex.textwidth/2) - 1, unit="cm"
  )
  Sys.setenv(R_GSCMD = "/usr/local/bin/gs");
  embed_fonts(outputFile);
}

plotResultOfSingleDensityDataSet <-function(trueDensities, computedDensities, outputFile, distribution){
  printf("Plotting %s as single denisty\n", outputFile);
}

plotResult <- function(data, outputFile, distribution, limits){
  if(isSingleDensityDataSet(data$trueDensity)){
    plotResultOfSingleDensityDataSet(data, outputFile, distribution)
  } else{
    plotResultOfMultipleDensityDataSet(data, outputFile, distribution, limits)
  }
}

findPlotLimits <- function(data){
  minimum = Inf;
  maximum = -Inf;
  
  for(df in data){
    minimum = min(minimum, df$trueDensity, df$computedDensity);
    maximum = max(maximum, df$trueDensity, df$computedDensity);
  }
  c(minimum = minimum, maximum = maximum);
}

extractEstimatorSensitivityDataset <- function(file_name){
  matches = str_match(file_name, 'results_([[:alpha:]]+_[[:digit:]])_([[:digit:]]+)_([[:alpha:]]+)_([[:alpha:]]+)')
  dataset = matches[2];  
  size = matches[3];
  estimator = matches[4];
  sensitivity = matches[5];
  
  data.frame(estimator=estimator, sensitivity=sensitivity, dataset=dataset, size=size, stringsAsFactors=TRUE)
}

updateResultTable <- function(df, dataset, outputfilename){
  
  newRow = extractEstimatorSensitivityDataset(basename(outputfilename));
  newRow['mse'] = computeMSE(dataset);
  
  df <- rbind(df, newRow);
  df;
}

mainResults <- function(){
  filePairs = getFiles();
  
  overview <- data.frame(estimator=character(), sensitivity=character(), dataset=character(), mse=double(), size=integer(), stringsAsFactors=TRUE)
  
  for (filePair in filePairs){
    list[dataPoints , trueDensities, distribution] = readDataSet(filePair$dataFile)
    data <- list();
    outputFiles <- list();
    idx = 1;
    
    # Read the results
    for(resultPath in filePair$associatedResults){
      computedDensities = readResults(resultPath);
      
      printf('Processing: %s\n', basename(resultPath));
      
      outputFiles[[idx]] = outputFilePath(resultPath, "results_");
      data[[idx]] = data.frame(points=dataPoints, trueDensities=trueDensities, computedDensities=computedDensities);
      
      overview <- updateResultTable(overview, data[[idx]], outputFiles[[idx]]);
      
      idx <- idx + 1;
    }
    
    limits = findPlotLimits(data);
    
    # Plot the results
    for (i in seq(1, length(data))){
      plotResult(data[[i]], outputFiles[[i]], distribution, limits);
    }
    
  }
  
  write.csv(file=overviewFilePath(resultPath), x=overview);
  print(overview)
  
  
}

mainResults()

