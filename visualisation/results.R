# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");

# Load libraries
library(ggplot2);
library(scales);
library(extrafont);
library(stringr);
library(gridExtra);

global.exponent = -4

add.alpha <- function(col, alpha=1){
  if(missing(col))
    stop("Please provide a vector of colours.")
  apply(sapply(col, col2rgb)/255, 2, 
        function(x) 
          rgb(x[1], x[2], x[3], alpha=alpha))  
}

computeMSE <- function(data){
  error <- data$trueDensity - data$computedDensity;
  mse <- mean(error^2)
  mse;
}

isSingleDensityDataSet <- function(trueDensities){
  length(unique(trueDensities)) <= 5;
}

fancy_scientificLabels <- function(breaks) {
  # # turn in to character string in scientific notation
  # l <- format(breaks, scientific = TRUE)
  # # Find the all exponents
  # exponents <- unlist(sapply(str_match_all(l, "e([+,-][0-9]+)"), function(x) x[,2]))
  # min_exponent = min(as.numeric(exponents))
  min_exponent = global.exponent
  # Multiply everyone with the smallest exponent
  new_breaks <- (breaks * 1/(10^(min_exponent)))
  # Call format again
  l <- format(new_breaks)
  # return this as an expression
  parse(text=l)
}

buildLabel <- function(labelText, exponent){
  library(latex2exp)
  TeX(
    sprintf("%s $\\left( \\times 10^{%d} \\right)$", labelText, exponent)
  )
}

plotResultOfMultipleDensityDataSet <-function(data, outputFile=NULL, distribution, limits, xlabel="true density", ylabel='estimated density', addMSE=TRUE){
  cols = generateColours(distribution);
  cols = add.alpha(cols, alpha=1);
  symbols = generateSymbols(distribution);
  
  if (addMSE){
    margin = c(0,2.5,0,0)
  } else {
    margin = c(0, 0, 0, 0)
  }
  
  plot <- ggplot(data) +
    theme(
      plot.title = element_text(family = font.family, size=font.size, hjust = 0.5),
      
      text=element_text(family=font.family, size=font.size),
      
      panel.border = element_rect(colour = "black", fill=NA),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      
      plot.margin=unit(margin, "mm")
    );
  plot <- plot + geom_point(aes(x=trueDensity, y=computedDensity), size=0.7, colour=cols, shape=symbols, stroke=0.2);
  plot <- plot + geom_line(aes(x=trueDensity, y=trueDensity));
  plot <- plot + xlab(buildLabel(xlabel, global.exponent)) + ylab(buildLabel(ylabel, global.exponent));
  plot <- plot + scale_x_continuous(labels = fancy_scientificLabels, limits = c(limits['xMin'], limits['xMax'])) 
  plot <- plot + scale_y_continuous(labels = fancy_scientificLabels, limits = c(limits['yMin'], limits['yMax']));

  if (addMSE){
    plot <- plot + ggtitle(sprintf("MSE = %.4e", computeMSE(data)));  
  }
  print(plot)
  if (!is.null(outputFile)){
    ggsave(
      outputFile,
      plot,
      width=(tex.textwidth/2) - 1, height=(tex.textwidth/2) - 1, unit="cm"
    )
    # Sys.setenv(R_GSCMD = "/usr/local/bin/gs");
    # embed_fonts(outputFile);    
  }

  
  plot;
}

plotResultOfSingleDensityDataSet <-function(trueDensities, computedDensities, outputFile, distribution){
  printf("Plotting %s as single denisty\n", outputFile);
}

plotResult <- function(data, outputFile, distribution, limits, xlabel='true density', ylabel='estimated density', addMSE=TRUE){
  if(isSingleDensityDataSet(data$trueDensity)){
    plotResultOfSingleDensityDataSet(data, outputFile, distribution)
  } else{
    plotResultOfMultipleDensityDataSet(data, outputFile, distribution, limits, xlabel, ylabel, addMSE)
  }
}

findPlotLimits <- function(data){
  minimum = Inf;
  maximum = -Inf;
  
  for(df in data){
    minimum = min(minimum, df$computedDensity);
    maximum = max(maximum, df$computedDensity);
  }
  
  trueDensities = data[1][[1]]$trueDensity
  c(xMin = min(trueDensities), xMax = max(trueDensities), 
    yMin = minimum, yMax = maximum);
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
      result = readResults(resultPath);
      result$numUsedPatterns <- NULL
      
      printf('Processing: %s\n', basename(resultPath));
      
      outputFiles[[idx]] = outputFilePath(resultPath, "results_", '.png');
      data[[idx]] = data.frame(points=dataPoints, trueDensities=trueDensities, computedDensities=result);
      
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

testMainResults <- function(){
  test.N = 100
  test.data <- data.frame(
    points.x = runif(test.N, 0, 100), points.y = runif(test.N, 0, 100), points.z = runif(test.N, 0, 100),
    trueDensity = runif(test.N, 0, 1.0), computedDensity = runif(test.N, 0, 1.0),
    points.component = round(runif(test.N, 0, 2))
  )
  test.data <- test.data[order(test.data$points.component), ]
  test.distribution <- table(test.data$points.component)
  test.limits <- c(xMin = min(test.data$trueDensity), xMax = max(test.data$trueDensity), 
                   yMin = min(test.data$computedDensity), yMax = max(test.data$computedDensity));
  plotResultOfMultipleDensityDataSet(data=test.data, distribution = test.distribution, limits = test.limits)  
}

# mainResults()