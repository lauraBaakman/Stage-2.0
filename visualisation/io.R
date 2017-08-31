source('header.r')

outputFilePath <- function(inputFilePath, description, extension='.pdf'){
  inputFile = removeExtension(basename(inputFilePath));
  outputFile = paste(description, inputFile ,extension, sep='');
  file.path(imagesOutputPath, outputFile);
}

overviewFilePath <- function(path){
  outputFile = paste('overview' ,'.csv', sep='');
  file.path(imagesOutputPath, outputFile);
}

parseStringWithInts <- function(string){
  as.numeric(
    unlist(
      regmatches(string, gregexpr("[0-9]+", string))
    )
  );
}

readHeader <- function(filePath){
  line = readLines(con = filePath, n = 2);
  list[rows, cols] = parseStringWithInts(line[1]);
  numberOfPatternsPerSubSet = parseStringWithInts(line[2]);
  list(
    rows=rows, 
    cols=cols, 
    numberOfPatternsPerSubSet=numberOfPatternsPerSubSet
  );
}

readData <- function(filePath, rows){
  data <- read.csv(
    file=filePath, 
    header=FALSE, 
    dec='.', 
    row.names = NULL, 
    skip=2, 
    nrows=rows, 
    sep=' ', 
    col.names = c('x', 'y', 'z')
  )
}

readTrueDensities <- function(filePath, rows){
  densities <- read.csv(
    file=filePath, 
    header=FALSE, 
    dec='.', 
    row.names = NULL, 
    skip=2 + rows, 
    nrows=rows, 
    sep=' ', 
    col.names = c('trueDensity')
  )
}

buildComponentColumn <- function(distribution){
  componentColum <- NULL;
  component = 0;
  for (numberOfPatterns in distribution){
    componentColum <- c(componentColum, rep(component, numberOfPatterns))
    component <- component + 1;
  }
  componentColum;
}

readDataSet <- function(filePath){
  list[rows, cols, numberOfPatternsPerSubSet] <- readHeader(filePath);
  data <- readData(filePath, rows);
  data$component = buildComponentColumn(numberOfPatternsPerSubSet);
  densities <- readTrueDensities(filePath, rows);
  list(
    data = data, 
    densities = densities, 
    numberOfPatternsPerSubSet = numberOfPatternsPerSubSet
  );
}

readResults <- function(filePath){
  read.csv(
    file=filePath, 
    header=FALSE, 
    dec=".", 
    row.names=NULL, 
    sep = " ",
    col.names = c('computedDensity', 'numUsedPatterns')
  )	
}

getDataSetPaths <- function(){
  list.files(path=dataInputPath, pattern="*.txt", full.names=TRUE, no..=TRUE);
}

getOutputPaths <- function() {
  list.files(path=dataOutputPath, pattern="*.txt", full.names=TRUE, no..=TRUE);	
}

removeExtension <- function(file){
  sub("^([^.]*).*", "\\1", file);
}