#
# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#

library(dataRetrieval)
library(ggplot2)
library(janitor)
library(knitr)
library(leaflet)
library(lubridate)
library(mapview)
library(sf)
library(shiny)
library(shinyTime)
library(sp)
library(tidyverse)
library(xgboost)
library(climateAnalyzeR)

#### Define Functions ##########################################################

sysDate1 <- Sys.time()

# T Min
get.Tmin <- function(sysDate) {
  formattedEndYear <- as.numeric(format(sysDate, "%Y"))
  TMin <- climateAnalyzeR::import_data("daily_wx"
                                       , station_id = 'KA7WSB-1'
                                       , start_year = formattedEndYear-1
                                       , end_year = formattedEndYear
                                       , station_type = 'RAWS')
  
  Var_TMin <- as.numeric(unlist(TMin %>%
                                  mutate(DateasDate = as.POSIXct(TMin$date, format = "%m/%d/%Y")) %>%
                                  subset(DateasDate == as.Date(sysDate) - 2) %>%
                                  select(tmin_f)))
}

Var_TMin <- get.Tmin(sysDate1)

# Discharge
get.DischargeCFS <- function(sysDate) {
  startDate <- as.Date(format(sysDate1,'%Y-%m-%d')) - 31
  endDate <- as.Date(format(sysDate1,'%Y-%m-%d')) - 1
  USGSRaw <- readNWISuv(siteNumbers = '09481740', c('00060','00065'), startDate,endDate, tz = 'America/Phoenix')
  
  tail(USGSRaw$X_00060_00000, n=1)
}

Var_Discharge_CFS <- get.DischargeCFS(sysDate1)

# Stage
get.stage <- function(sysDate) {
  startDate <- as.Date(format(sysDate1,'%Y-%m-%d')) - 31
  endDate <- as.Date(format(sysDate1,'%Y-%m-%d')) - 1
  USGSRaw <- readNWISuv(siteNumbers = '09481740', c('00060','00065'), startDate,endDate, tz = 'America/Phoenix')
  
  # Create quantiles for categorization
  CFS_Quantiles<- quantile(USGSRaw$X_00060_00000, na.rm = TRUE)
  
  # Determine the difference between prior reading and current.
  USGSRaw <- USGSRaw %>% 
    mutate(DisDif = X_00060_00000 - lag(X_00060_00000))
  
  # This will create a binary variable or either rise of fall. Rise = 1, fall = 0. It will allow me to more easily create summary statistics.
  USGSRaw$DisDif2 <- ifelse(USGSRaw$DisDif>0,1,0)
  
  # Create a numeric classifier. 
  # 1 = Low Flow, 2 = Base flow, 3 = High and Rising Flow 4 = High and Falling Flow
  USGSRaw$Stage <- ifelse(USGSRaw$X_00060_00000 <=CFS_Quantiles[2], 1, 
                          ifelse(USGSRaw$X_00060_00000 > CFS_Quantiles[2] & USGSRaw$X_00060_00000 <= CFS_Quantiles[4],2,
                                 ifelse(USGSRaw$X_00060_00000 > CFS_Quantiles[4] & USGSRaw$DisDif2 == 1,3,
                                        ifelse(USGSRaw$X_00060_00000 > CFS_Quantiles[4] & USGSRaw$DisDif2 == 0,4, NA))))
  
  
  # Create the stage variable.
  tail(USGSRaw$Stage, n=1)
}

Var_Stage <- get.stage(sysDate1)

# El Nino
get.NinXTS <- function(sysDate) {
  formattedEndYear <- as.numeric(format(sysDate, "%Y"))
  formattedMonth <- as.numeric(format(sysDate,"%m"))
  
  # Bring in the website data
  url <- "https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php"
  
  NinXTS <- url %>%
    rvest::read_html()
  
  # Grab the data table
  NinText <- rvest::html_table(rvest::html_nodes(NinXTS, xpath = './/table[4]//table[2]'))
  
  # Convert ONI index to dataframe
  NinTable <- as.data.frame(NinText[1]) %>%
    row_to_names(row_number = 1) %>%
    mutate(Year = as.numeric(Year)) %>%
    drop_na(Year)
  
  # I need to do either month-of or last non-na value to account for delays. 
  formattedMonth <-12
  
  NinVal <- NinTable %>%
    subset(2022 == Year) %>%
    select(case_when(formattedMonth == 1 ~ "NDJ",
                     formattedMonth == 2 ~ "DJF",
                     formattedMonth == 3 ~ "JFM",
                     formattedMonth == 4 ~ "FMA",
                     formattedMonth == 5 ~ "MAM",
                     formattedMonth == 6 ~ "AMJ",
                     formattedMonth == 7 ~ "MJJ",
                     formattedMonth == 8 ~ "JJA",
                     formattedMonth == 9 ~ "JAS",
                     formattedMonth == 10 ~ "ASO",
                     formattedMonth == 11 ~ "SON",
                     formattedMonth == 12 ~ "OND")) %>%
    unlist()
  
  PrevVal <- NinTable %>%
    subset(2022 == Year) %>%
    select(case_when(formattedMonth == 2 ~ "NDJ",
                     formattedMonth == 3 ~ "DJF",
                     formattedMonth == 4 ~ "JFM",
                     formattedMonth == 5 ~ "FMA",
                     formattedMonth == 6 ~ "MAM",
                     formattedMonth == 7 ~ "AMJ",
                     formattedMonth == 8 ~ "MJJ",
                     formattedMonth == 9 ~ "JJA",
                     formattedMonth == 10 ~ "JAS",
                     formattedMonth == 11 ~ "ASO",
                     formattedMonth == 12 ~ "SON",
                     formattedMonth == 1 ~ "OND")) %>%
    unlist()
  
  if_else(!is.na(NinVal), NinVal, PrevVal)                 
}

Var_NinXTS <- get.NinXTS(sysDate1)


# Time of Day
#modified from https://stackoverflow.com/questions/49370387/convert-time-object-to-categorical-morning-afternoon-evening-night-variable

get.TOD <- function(sysTime) {
  
  # Create categorical variables
  currenttime <- as.POSIXct(sysTime, format = "%H:%M") %>% format("%H:%M:%S")
  
  currenttime <- cut(chron::times(currenttime) , breaks = (1/24) * c(0,5,11,16,19,24))
  Var_TOD <- c(4, 1, 2, 3, 4)[as.numeric(currenttime)]
}

Var_TOD <- get.TOD(sysDate1)

# Dist from Sonoita is within the mapping layer

spatiallocs <- read_sf("Data/Processed/Attributed_Location/ecoli_UniqueLocs.shp",stringsAsFactors = TRUE)
spatiallocs <- spatiallocs %>%
  arrange(DistCatego)

# Retrieve all variables using the functions
predictionDF <- as.data.frame(spatiallocs)
predictionDF$PreviousTmin <- c(Var_TMin)
predictionDF$Discharge_CFS	<- c(Var_Discharge_CFS)
predictionDF$Stage	<- c(Var_Stage)
predictionDF$NinXTS	<- c(Var_NinXTS)
predictionDF$TOD <- c(Var_TOD)

predictionDF <- predictionDF %>%
  rename(DistFromSonoita = DistCatego) %>%
  select(PreviousTmin, Discharge_CFS, Stage, NinXTS, TOD, DistFromSonoita)

DisplayDF <- predictionDF %>%
  select(-DistFromSonoita) %>%
  distinct()


# Run the model for 235
XGBModel <- xgb.load('Data/Processed/XGBmodel235')
predictionDM <- data.matrix(predictionDF)
pred <- predict(XGBModel,predictionDM)
pred <-  as.numeric(pred > 0.4)
spatiallocs$pred235 <- c(pred)
spatiallocs$pred35 <- ifelse(spatiallocs$pred235 > 0, "Bacteria Level >235  Likely", "High Bacteria levels > 235 not predicted")


# Run the model for 575
XGBModel <- xgb.load('Data/Processed/XGBmodel575')
pred <- predict(XGBModel,predictionDM)
pred <-  as.numeric(pred > 0.4)
spatiallocs$pred575 <- c(pred)
spatiallocs$pred575 <- ifelse(spatiallocs$pred575 > 0, "Bacteria Level >575 Likely", "High Bacteria levels > 575 not predicted")


### Shiny UI ##################################################################

# Define UI for application that draws a histogram
ui <- fluidPage(
  

    # Application title
    titlePanel("E. coli in the Upper Santa Cruz"),
    
    # Time Selection straight from shiny
    h4(
      "The time is ",
      # We give textOutput a span container to make it appear
      # right in the h4, without starting a new line.
      textOutput("currentTime", container = span)
    ),
    selectInput("interval", "Update every:", c(
      "5 seconds" = "5000",
      "1 second" = "1000",
      "0.5 second" = "500"
    ), selected = 1000, selectize = FALSE),
    
    # display table
    fluidRow(
      column(12,
             tableOutput('table')
      )
    )
    


    # Sidebar with a slider input for number of bins 
    #sidebarLayout(
        #sidebarPanel(
            #sliderInput("bins",
                       # "Number of bins:",
                       # min = 1,
                       # max = 50,
                       # value = 30)
    #    ),

        # Show a plot of the generated distribution
       # mainPanel(
          # plotOutput("distPlot")
      #  )
  #  )
)

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  
  output$table <- renderTable(DisplayDF)
  output$currentTime <- renderText({
    # invalidateLater causes this output to automatically
    # become invalidated when input$interval milliseconds
    # have elapsed
    invalidateLater(as.integer(input$interval), session)
    
    format(Sys.time())
  })
    #output$distPlot <- renderPlot({
        # generate bins based on input$bins from ui.R
        #x    <- faithful[, 2]
       # bins <- seq(min(x), max(x), length.out = input$bins + 1)

        # draw the histogram with the specified number of bins
        #hist(x, breaks = bins, col = 'darkgray', border = 'white',
             #xlab = 'Waiting time to next eruption (in mins)',
             #main = 'Histogram of waiting times')
   # })
}


# Run the application 
shinyApp(ui = ui, server = server)
