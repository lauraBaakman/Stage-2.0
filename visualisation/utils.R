library(stringr)

generateColours <- function(numberOfPatternsPerSubSet){
  patternColours = list(); 
  for (idx in 1:length(numberOfPatternsPerSubSet)) {
    patternColours <- c(patternColours, 
                        rep(colours[idx], numberOfPatternsPerSubSet[idx]),
                        recursive=TRUE
    );
  }
  patternColours;
}

list <- structure(NA,class="result")
"[<-.result" <- function(x,...,value) {
  args <- as.list(match.call())
  args <- args[-c(1:2,length(args))]
  length(value) <- length(args)
  for(i in seq(along=args)) {
    a <- args[[i]]
    if(!missing(a)) eval.parent(substitute(a <- v,list(a=a,v=value[[i]])))
  }
  x
}

pathToFileName<-function(file){
  tools::file_path_sans_ext(basename(file))
}

datasetNamemapping = NULL
datasetNamemapping$'baakman_1' = 'set 4'
datasetNamemapping$'baakman_2' = 'set 5'
datasetNamemapping$'baakman_3' = 'set 6'
datasetNamemapping$'baakman_4' = 'set 7'
datasetNamemapping$'baakman_5' = 'set 8'
datasetNamemapping$'ferdosi_1' = 'set 1'
datasetNamemapping$'ferdosi_2' = 'set 2'
datasetNamemapping$'ferdosi_3' = 'set 3'
datasetNamemapping$'ferdosi_4' = 'set 9'
datasetNamemapping$'ferdosi_5' = 'set 10'

pathToDataSetName<-function(path){
  filename = pathToFileName(path)
  semantic_set_name = regmatches(filename, regexpr('^[[:alpha:]]+_[[:digit:]]+', filename)); 
  datasetNamemapping[[semantic_set_name]]
}

isGridResultsFile<-function(path){
  filename = pathToFileName(path);  
  grepl("^(ferdosi|baakman)_[[:digit:]]+_[[:digit:]]+_(parzen|mbe|sambe)_?(breiman|silverman)?_grid_[[:digit:]]+$", filename);
}

splitResultsFileName<-function(path){
  filename = pathToFileName(path);
  match = str_match(filename, "([[:alpha:]]+_[[:digit:]]+)_([[:digit:]]+)_(parzen|mbe|sambe)_?(breiman|silverman)?_?(grid)?_?([[:digit:]]+)?")
  list(dataset=match[2], 
        size=match[3],
        estimator=match[4],
        sensitivity=match[5], 
        gridsize=match[7])
}

splitDataFileName<-function(path){
  filename = pathToFileName(path);
  regmatches(filename, regexpr('^[[:alpha:]]+_[[:digit:]]+', filename))
}

isGridFile<-function(path){
  filename = pathToFileName(path);  
  grepl("^(ferdosi|baakman)_[[:digit:]]+_[[:digit:]]+_grid_[[:digit:]]+$", filename)[1];
}

isXisResultFile<-function(path){
  filename = pathToFileName(path);  
  grepl("^.*_xis.*$", filename)[1];
}

splitGridFileName<-function(path){
  filename = pathToFileName(path);
  match = str_match(filename, "([[:alpha:]]+_[[:digit:]]+)_([[:digit:]]+)_grid_([[:digit:]]+)")
  list(dataset=match[2], 
       size=match[3],
       gridsize=match[4]
  )
}

findResultsAssociatedWithDataSet <- function(dataset_file, results){
  isResultAssociatedWithDataSetFile <- function(resultsFile){
    paste(
      regmatches(
        basename(resultsFile), 
        regexpr("([A-Za-z]+)_([0-9]+)_([0-9]+)", basename(resultsFile))
      ), '.txt', sep=''
    ) == basename(dataset_file); 
  }  
  
  idx = sapply(results, isResultAssociatedWithDataSetFile, USE.NAMES = FALSE)
  
  list(
    associated=results[idx],
    results=results[!idx]
  );
}

extractGridFiles<-function(data_set_results){
  grid_files_idx = 1; files_idx = 1;
  for (path in data_set_results$associatedResults){
    if (isGridFile(path)){
      data_set_results$grid_files[[grid_files_idx]] = path;
      grid_files_idx = grid_files_idx + 1;
      data_set_results$associatedResults = data_set_results$associatedResults[-files_idx]
    }
    files_idx = files_idx + 1;
  }
  data_set_results
}

extractGridResultFiles<-function(data_set_results){
  gridResultFilesIdx = sapply(data_set_results$associatedResults, isGridResultsFile, USE.NAMES=FALSE)
  data_set_results$grid_results = data_set_results$associatedResults[gridResultFilesIdx]
  data_set_results$associatedResults = data_set_results$associatedResults[!gridResultFilesIdx]
  
  data_set_results;
}

removeXisResultFiles<-function(files){
  xisFilesIdx = sapply(files, isXisResultFile, USE.NAMES = FALSE)
  files[!xisFilesIdx]
}

getFiles <- function(){
  printf('Reading dataset files from %s\n', dataInputPath);
  printf('Reading result files from %s\n', dataOutputPath);
  
  dataSetsPaths = getDataSetPaths();
  resultsPaths = getOutputPaths();
  resultsPaths = removeXisResultFiles(resultsPaths);
  
  filePairs = list();	i = 1;
  for (file in dataSetsPaths) {
    if(length(resultsPaths) == 0){
      break;
    }
    list[associated, resultsPaths] = findResultsAssociatedWithDataSet(file, resultsPaths);
    if(length(associated) != 0){
      filePairs[[i]] = c(dataFile=file, associatedResults=list(associated));
      filePairs[[i]]= extractGridFiles(filePairs[[i]])
      filePairs[[i]]= extractGridResultFiles(filePairs[[i]])
      i = i + 1;
    }
  }
  filePairs;
}