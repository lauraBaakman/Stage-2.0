# Define paths
dataInputPath = "/Users/laura/Repositories/stage/data/simulated/small";
dataOutputPath = "/Users/laura/Repositories/stage/results/simulated/small";
imagesOutputPath = "/Users/laura/Desktop";

# Settings for the plots
colours = list(
  blue   	= rgb(073, 119, 177, max = 255),
  green   = rgb(099, 159, 058, max = 255),
  red 	  = rgb(194, 032, 034, max = 255),
  orange  = rgb(228, 126, 029, max = 255),
  purple  = rgb(096, 063, 150, max = 255),
  black   = rgb(000, 000, 000, max=255),
  yellow  = rgb(255, 255, 157, max=255	)
)
font.size = 10;
font.family ="CM Roman";

tex.textwidth = 14.92038; #cm

glyph.size.min = 0.7
glyph.size.max = 3.0

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