# Remove al existing variables
rm(list = ls())

# Load external methods and variables
source("./header.R");
source("./io.R");

# Load libraries
library(scatterplot3d)
# https://github.com/bhaskarvk/colormap
library(colormap)

createDifferencePlot3D<-function(data, values_1, values_2, plot_title='', alpha=0.2){
  data$differences = computeSquaredError(values_1, values_2);
  data$normalizedDifferences = scaleToUnitRange(data$differences);
  
  sizes = computeSizes(data$normalizedDifferences, maxSize = 5, minSize = 0.5);
  colormap = computeColorMap(numShades = 15);
  colors = computeColors(data$normalizedDifferences, colormap);
  
  #           b,   l, t, r)
  margins = c(2.4, 3, 1, 5)
  glyph=16
  
  plot <- scatterplot3d(
    x = data$x, y = data$y, z = data$z,
    xlab='x', ylab='y', zlab='z',
    pch=glyph,
    color=adjustcolor(colors, alpha.f = alpha),
    grid=FALSE,
    lty.hide=4,
    cex.lab=1.0,
    label.tick.marks=TRUE,
    mar=margins,
    cex.symbols = sizes,
    main=plot_title
  )
  par(mar=margins)
  par(xpd=TRUE)
  legend(plot$xyz.convert(105, 100, 100), 
         pch=glyph,
         col= colormap,
         cex=0.9,
         horiz=FALSE,
         # lty=c(1,1),
         legend = computeLegendValues(data$differences, colormap)
  )
  return(plot);
}

computeSquaredError<-function(values_1, values_2){
  (values_1 - values_2)^2
}

computeSizes<-function(values, minSize=2.0, maxSize=20){
  sizes = minSize + values * (maxSize - minSize);
  return(sizes);
}

computeColorMap<-function(numShades=20){
  colorMap = colormap(colormap=colormaps$portland, nshades=numShades, reverse=F);
  return(colorMap);
}

computeColors<-function(values, colormap){
  colorIdxes = round((values * (length(colormap) - 1)) + 1);
  return(colormap[colorIdxes]);
}

computeLegendValues<-function(values, colormap){
  numShades = length(colormap);
  legendValues = seq(from=min(values), to=max(values), length.out = numShades);
  legendStrings = format(legendValues, scientific = T, digits=3)
  return(legendStrings);
}

scaleToUnitRange<-function(values){
  oldMin = min(values);
  oldMax = max(values);
  oldRange = oldMax - oldMin;
  newValues = (values - oldMin) / oldRange;
  return(newValues);
}

data = readDataSet('/Users/laura/Desktop/small/results/baakman_1_90_grid_3.txt')$data;
values_1 = readResults('/Users/laura/Desktop/small/results/baakman_1_90_mbe_breiman_grid_3.txt')$computedDensity;
values_2 = readResults('/Users/laura/Desktop/small/results/baakman_1_90_parzen_grid_3.txt')$computedDensity;
createDifferencePlot3D(data, values_1, values_2)