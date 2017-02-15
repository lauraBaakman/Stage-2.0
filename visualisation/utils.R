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