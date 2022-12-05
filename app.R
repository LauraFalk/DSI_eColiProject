library(dataRetrieval)
library(knitr)
library(leaflet)
library(mapview)
library(sf)
library(shiny)
library(shinyTime)
library(sp)
library(tidyverse)
library(xgboost)
library(climateAnalyzeR)

###### Background info

# Map color palette
ColorPal <- colorFactor(palette = c('Green', 'Red'), levels = c(0,1))


###### Model inputs


###### User Interface
ui <- fluidPage(
  titlePanel("Upper Santa Cruz River - bacterial level predictions"),
  leafletOutput("mymap"),
  dateInput("CurrentDate", "Current Date:", 
                 max    = Sys.Date(),
                 format = "mm/dd/yyyy"),
  timeInput("Time", "Time:", value = Sys.time()),
)


####### App 

server <- function(input,output, session) {
  formatyear <- as.POSIXct.Date(CurrentDate)
  
  output$USGSDataframe <- reactive({
    climateAnalyzeR::import_data("daily_wx", station_id = 'KA7WSB-1', start_year = date1, end_year = date1, station_type = 'RAWS')
    TMin$Date1 <- as.POSIXct(TMin$date1, format = "%m/%d/%Y", tz = tz)
    
    
    Var_TMin <- TMin %>%
      subset(date1 == date1 - 2) %>%
      select(tmin_f)
    Var_TMin <- as.numeric(unlist(Var_TMin))
  })
  
  
  
  
  output$mymap <- renderLeaflet({
    leaflet() %>%
      addTiles() %>%  # Add default OpenStreetMap map tiles
      addCircleMarkers(
      data = spatiallocs
        , popup = paste0( "Sampling Organization(s):"
                          ,spatiallocs$Samplers
                          , "<br>"
                          ,spatiallocs$Prediction)
        , color = ColorPal(spatiallocs$pred2)
      )

  })
  
}

shinyApp(ui, server)
