# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");

# Load libraries
library(ggplot2);
library(extrafont);
library(gridExtra)

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
    		panel.border = element_blank(),
    		panel.background = element_blank()
    	);
    plot <- plot + 	geom_point(aes(x=trueDensity, y=computedDensity), size=0.7, colour=cols);
    plot <- plot + 	geom_line(aes(x=trueDensity, y=trueDensity));

    plot <- plot + 	xlab('true density') +
					ylab('computed density');

    plot <- plot + ggtitle(sprintf("MSE = %.7e", computeMSE(data)));

	  plot <- plot + 	xlim(limits[[1]], limits[[2]]) +
					          ylim(limits[[1]], limits[[2]]);
    print(plot)
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

findResultsAssociatedWithDataSet <- function(file, results){
	idx = sapply(
		results,
		function(x)
			grepl(
				sub("[A-Z-]*", "\\1", basename(x)),
				basename(file)
			),
		USE.NAMES=FALSE);

	list(
		associated=results[idx],
		results=results[!idx]
	);
}

getFiles <- function(){
	dataSetsPaths = getDataSetPaths();
	resultsPaths = getOutputPaths();
	filePairs = list();	i = 1;
	for (file in dataSetsPaths) {
		if(length(resultsPaths) == 0){
			break;
		}
		list[associated, resultsPaths] = findResultsAssociatedWithDataSet(file, resultsPaths);
		if(length(associated) != 0){
			filePairs[[i]] = c(dataFile=file, associatedResults=list(associated));
			i = i + 1;
		}
	}
	filePairs;
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

mainResults <- function(){
	filePairs = getFiles();

	for (filePair in filePairs){
		list[dataPoints , trueDensities, distribution] = readDataSet(filePair$dataFile)
		data <- list();
		outputFiles <- list();
		idx = 1;

		# Read the results
		for(resultPath in filePair$associatedResults){
			computedDensities = readResults(resultPath);
			outputFiles[[idx]] = outputFilePath(resultPath, "results_");
			data[[idx]] = data.frame(points=dataPoints, trueDensities=trueDensities, computedDensities=computedDensities);
			idx <- idx + 1;
		}

		limits = findPlotLimits(data);

		# Plot the results
		for (i in seq(1, length(data))){
			plotResult(data[[i]], outputFiles[[i]], distribution, limits);
		}

	}
}

mainResults();
