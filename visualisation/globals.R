# Define paths
dataInputPath = "/Users/laura/Repositories/stage-2.0/data/simulated";
dataOutputPath = "/Users/laura/Repositories/stage-2.0/results/simulated";
imagesOutputPath = "/Users/laura/Desktop";

# Settings for the plots
colours = list(
  blue   	= rgb(73, 119, 177, max = 255),
  green   = rgb(99, 159, 58, max = 255),
  red 	  = rgb(194, 32, 34, max = 255),
  orange  = rgb(228, 126, 29, max = 255),
  purple  = rgb(96, 63, 150, max = 255)
)
font.size = 10;
font.family ="CM Roman";

tex.textwidth = 14.92038; #cm

resetPar <- function() {
  dev.new()
  op <- par(no.readonly = TRUE)
  dev.off()
}

printf <- function(...) {
  cat(sprintf(...))
}

reload <- function(){
  rm(list = ls())
  source('globals.R')
  sapply(list.files(pattern="[.]R$", full.names=TRUE), source);
}