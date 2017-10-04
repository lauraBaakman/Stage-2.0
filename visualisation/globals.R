# Define paths
dataInputPath = "../data/simulated/normal";
dataOutputPath = "../results/normal/";
imagesOutputPath = "/Users/laura/Desktop";

# Settings for the plots
colours = list(
  blue   	= rgb(073, 119, 177, max = 255),
  red 	  = rgb(194, 032, 034, max = 255),
  orange  = rgb(228, 126, 029, max = 255),
  purple  = rgb(096, 063, 150, max = 255),
  black   = rgb(000, 000, 000, max=255),
  yellow  = rgb(255, 255, 157, max=255),	
  green   = rgb(099, 159, 058, max = 255, alpha=128)
)
# Quick hack to get back to bullets for everything
symbols = list(
  filledRectangle       = 16,
  filledTriangle 	      = 16, 
  filledDiamond         = 16, 
  filledEightPointStar  = 16,
  filledTwelvePointStar = 16,
  bullsEye              = 16,
  filledCircle   	      = 16  
)
font.size = 10;
# font.family ="CM Roman";
font.family ="Times-Roman";

tex.textwidth = 14.92038; #cm
tex.textheight = 22.70108; #cm

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