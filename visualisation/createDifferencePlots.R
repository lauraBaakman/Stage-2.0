# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");
source("./differencePlot.R")

compareOnGrid <- function(meta_data){
  !is.na(meta_data$gridsize)
}

createTitle<-function(data_set_file, file_values_1, file_values_2){
  estimatorName <- function(meta_data){
    if (is.na(meta_data$sensitivity)){
      sprintf('%s', meta_data$estimator);
    } else {
      sprintf('%s (%s)', meta_data$estimator, meta_data$sensitivity)  
    }
  }
  
  comparisonPart <-function(meta_data_1, meta_data_2){
    sprintf('Compare %s with %s on', estimatorName(meta_data_1), estimatorName(meta_data_2))
  }
  
  dataSetPart <-function(data_set_name, meta_data){
    if(compareOnGrid(meta_data)){
      sprintf('%s on a %s x ... x %s grid', data_set_name, meta_data$gridsize, meta_data$gridsize)
    } else {
      sprintf('%s', data_set_name)
    }
  }
  
  data_set_name = pathToDataSetName(data_set_file);
  
  values_1_meta = splitResultsFileName(file_values_1);
  values_2_meta = splitResultsFileName(file_values_2);

  sprintf('%s %s', comparisonPart(values_1_meta, values_2_meta), dataSetPart(data_set_name, values_1_meta))  
}

createOutputFileName<-function(file_values_1, file_values_2, extension='png'){
  values_1_meta = splitResultsFileName(file_values_1);
  values_2_meta = splitResultsFileName(file_values_2);  
  
  estimatorName <- function(meta_data){
    if (is.na(meta_data$sensitivity)){
      sprintf('%s', meta_data$estimator);
    } else {
      sprintf('%s_%s', meta_data$estimator, meta_data$sensitivity)  
    }
  }

  gridName<-function(meta_data){
    if(compareOnGrid(meta_data)){
      sprintf('grid_%s', meta_data$gridsize)
    } else {
      ''
    }
  }
  
  estimator_1 = estimatorName(values_1_meta)  
  estimator_2 = estimatorName(values_2_meta)  
  data_set = values_1_meta$dataset
  grid = gridName(values_1_meta)
  
  sprintf('difference_%s_%s_vs_%s%s%s.%s', data_set, estimator_1, estimator_2, if(!is.na(values_1_meta$grid)) '_' else '', grid, extension)
}


createOutputFilePath<-function(path, file_name){
  sprintf('%s/%s', path, file_name)
}

createPlots<-function(data_file, result_files){
  positions = readDataSet(data_file)$data
  
  for (result_file_1 in result_files){
    results_1 = readResults(result_file_1);
    
    for (result_file_2 in result_files){
      if (result_file_1 != result_file_2){
        results_2 = readResults(result_file_2);
        title = createTitle(data_file, result_file_1, result_file_2);
        file_path = createOutputFilePath(imagesOutputPath, createOutputFileName(result_file_1, result_file_2));
        createPlot(positions, results_1, results_2, title, file_path)
      }
    }
  }
}

createPlot<-function(positions, result_1, result_2, title, file_path){
  plot = createDifferencePlot(positions, result_1, result_2, title);
  publishDifferencePlotLocally(plot, title, file_path);
}

results = getFiles();

for (result in results){
  createPlots(result$grid_files[1], result$grid_results)  
  createPlots(result$dataFile, result$associatedResults)  
}