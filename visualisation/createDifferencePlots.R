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
  positions = readDataSet(data_file)$data;
  
  printf('Processing data file: %s\n', data_file);
  
  for (idx_1 in 1:length(result_files)){
    result_file_1 = result_files[idx_1];
    results_1 = readResults(result_file_1);
    
    if( (idx_1 + 1) <= length(result_files)){
      for(idx_2 in seq(idx_1 + 1,length(result_files))){
        result_file_2 = result_files[idx_2];
        printf('\tComparing %s with %s\n', basename(result_file_1), basename(result_file_2));
        if (result_file_1 != result_file_2){
          results_2 = readResults(result_file_2);
          title = createTitle(data_file, result_file_1, result_file_2);
          file_path = createOutputFilePath(imagesOutputPath, createOutputFileName(result_file_1, result_file_2));
          createPlot(positions, results_1$computedDensity, results_2$computedDensity, title, file_path)
        }
      }      
    }
  }
}

createPlot<-function(positions, result_1, result_2, title, file_path){
  png(file_path);
  plot = createDifferencePlot3D(positions, result_1, result_2, title);
  dev.off();
}

mainCreateDifferencePlots3D<-function(){
  results = getFiles();
  
  for (result in results){
    if(!is.null(result$grid_files[1])){
        createPlots(result$grid_files[1], result$grid_results)    
    }
    if(!is.null(result$dataFile) && (length(result$associatedResults) != 0)){
      createPlots(result$dataFile, result$associatedResults)    
    }
  }  
}

file.remove('/Users/laura/Desktop/difference_baakman_1_mbe_breiman_vs_parzen_grid_3.png')
mainCreateDifferencePlots3D();