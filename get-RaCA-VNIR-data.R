# load libraries
library(soilDB)
library(plyr)
library(maps)
library(lattice)
library(reshape2)
library(cluster)

seriesNameCSV <- read.csv("~/Documents/RaCA-SOC-a-spectrum-analysis/RaCA-series-names.csv",sep=',')
#seriesNamesCSV <- list('auburn','bbb','christian')
dl <- list()

for(tser in colnames(seriesNameCSV)) {
  tryCatch({
      print("---------------------------\n")
      print(tser)
      print("---------------------------\n")
      
      r.alldat <- fetchRaCA(series=tser,get.vnir=TRUE)
      if(exists("r.alldat")) {
        d <- as.data.frame(r.alldat$spectra)
        d$sample_id <- rownames(d)
        dl <- rbind(dl, merge(d, r.alldat$sample, by='sample_id', all.x=FALSE))
        
        par(mar=c(0,0,0,0))
        matplot(t(r.alldat$spectra), type='l', lty=1, col=rgb(0, 0, 0, alpha=0.25), ylab='', xlab='', axes=FALSE)
        box()
      }
  },
  error = function(cond) {
    message(cond)
  },
  finally={
    if(exists("r.alldat")) {
      remove(r.alldat)
      remove(d)
      next
    }
  })
}

write.table(dl, file="~/Documents/RaCA-SOC-a-spectrum-analysis/RaCA-spectra-raw.txt", row.names=F, sep=",")