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