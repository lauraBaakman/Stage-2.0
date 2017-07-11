# Load libraries
library(plotly)

createDifferencePlot<-function(data, values_1, values_2, plot_title){
  data$differences = computeSquaredError(values_1$computedDensity, values_2$computedDensity);
  data <- arrange(data, desc(differences));
  
  plot = plot_ly(
    data,     
    x=~x, y=~y, z=~z,
    type = "scatter3d", mode="markers",
    size = ~differences,
    marker = list(
      color=~differences, 
      colorscale = "Blackbody",
      reversescale=TRUE,
      showscale = TRUE,
      symbol='circle',
      sizemode='diameter',
      colorbar = list(
        title='squared error',
        thickness=15,
        titleside='bottom'
      )
    ),
    sizes= c(8, 30)
  ) %>%
  layout(
    title = plot_title
  ) 
  plot;
}

computeSquaredError<-function(values_1, values_2){
  (values_1 - values_2)^2
}

publishDifferencePlotLocally<-function(plot, file_name, outfilepath){
  api_create(plot, filename=file_name, sharing='public')
  plotly_IMAGE(plot, format = "png", out_file = outfilepath)
}

publishDifferencePlotOnline<-function(plot, filename=NULL){
  response = api_create(
    x = plot, 
    filename = filename, fileopt = "overwrite", 
    sharing = "public");
  response$web_url;
}