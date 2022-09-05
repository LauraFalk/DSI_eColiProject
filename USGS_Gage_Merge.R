#install libraries first!
library(dataRetrieval)
library(lubridate)
library(data.table)
library(measurements)

##EDIT THIS##

setwd("C:/Users/Laura/Desktop/DSI_Project/Data/")
RawFileName <- "EPA_PortalQuery_Appended_SODN.csv"
startDate <- "2010-08-14T12:00"
endDate <- "2022-07-14T12:00"

## End Edits## 

#pulls in Sonde Data
RawFile<- read.csv(RawFileName, skip = 0, header = TRUE, sep = ",", quote = "\"",
                     dec = ".", fill = TRUE, comment.char = "")
USGS <- read.csv("USGS_09481740.csv")

###LAURA YOU NEED TO FIX THE HEADERS NOW THAT YOURE USING EPA FORMAT
#convert raw dateTime to POSIXct format
RawFile$DateTime <- mdy_hms(paste(RawFile$Date,RawFile$Time),tz="America/Phoenix")
RawFile$DateTime <-format(as.POSIXct(RawFile$DateTime),format='%Y-%m-%d %H:%M')

#convert USGS file datetime - delete seconds
#USGS$dateTime1 <- mdy_hms(USGS$dateTime,tz = "America/Phoenix")
USGS$dateTime1 <-format(as.POSIXct(USGS$dateTime), format = '%Y-%m-%d %H:%M')

#Set column name
setnames(USGS,old=c("X_00060_00000", "X_00065_00000"),new=c("Discharge_CFS", "USGS_Depth_ft"))
USGS$USGS_Depth_in <-conv_unit(USGS$USGS_Depth_ft, "ft", "inch")

#create merged file for graphs and tables
Outputtest<-
  merge(x = RawFile[ , c("DateTime", "ExceedsTest1T","Average_MPN", "Count")], 
        y = USGS[ , c("dateTime1", "Discharge_CFS", "USGS_Depth_in")], 
        by.x='DateTime', by.y = 'dateTime1', all.x=TRUE) 
#Export
write.csv(Outputtest,"C:/Users/Laura/Desktop/DSI_Project/ROutputs/Outputtest.csv")

