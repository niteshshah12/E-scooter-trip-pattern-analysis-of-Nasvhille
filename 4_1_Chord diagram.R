# Libraries
library(tidyverse)
library(viridis)
library(patchwork)
library(hrbrthemes)
library(circlize)
library(chords)  #devtools::install_github("mattflor/chorddiag")

#Reference: https://www.data-to-viz.com/graph/chord.html

# Load dataset
data <- read.csv("G:/My Drive/1. Project & Research/E-scooter/chord plot/all.csv", header=TRUE)
# short names
row.names(data) <- data$X
data <- data[2:6]

# I need a long format
data_long <- data %>%
  rownames_to_column %>%
  gather(key = 'key', value = 'value', -rowname)

colnames(data_long) <- c("Origins", "Destinations","value")


data_long[,3] = data_long[,3]/1000


# parameters
circos.clear()

circos.par(start.degree = 90,gap.degree = 4, track.margin = c(-0.1, 0.1), points.overflow.warning = FALSE)

par(mar = rep(0, 4))

# color palette
#mycolor <- viridis(10, alpha = 1, begin = 0, end = 1, option = "D")
#mycolor <- mycolor[sample(1:5)]

mycolor = c(commercial = "palevioletred2",downtown  = "skyblue3",others  = "peachpuff1",park  = "palegreen3",vanderbilt  = "plum3")

#commercial=#DB7093, downtown= #8db6cd, other=#ffdab9, park=#7ccd7c , vanderbilt= #cd96cd

# Base plot
chordDiagram(
  x = data_long,
  grid.col = mycolor,
  transparency = 0.25,
  directional = 1,
  direction.type = c("arrows", "diffHeight"), 
  diffHeight  = -0.05,
  annotationTrack = "grid", 
  annotationTrackHeight = c(0.05, 0.1),
  link.arr.type = "big.arrow", 
  link.sort = TRUE, 
  link.largest.ontop = TRUE)

# Add text and axis
circos.trackPlotRegion(
  track.index = 1, 
  bg.border = NA, 
  panel.fun = function(x, y) {
    
    xlim = get.cell.meta.data("xlim")
    sector.index = get.cell.meta.data("sector.index")
    
    # Add names to the sector. 
    circos.text(
      x = mean(xlim), 
      y = 3.8, 
      labels = sector.index,
      facing = "bending",
      cex = 1.3
    )
    
    # Add graduation on axis
    circos.axis(
      h = "top", 
      major.at = NULL,
      minor.ticks = 4, 
      major.tick.percentage = 0.1,
      labels.niceFacing = FALSE)
  }
)

