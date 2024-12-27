# Arbeitsverzeichniss
setwd("C:/srv/R/UE08")

# Pakete installieren wenn nicht schon vorhanden
install_and_load <- function(package) {
  if (!require(package, character.only = TRUE)) {
    install.packages(package, dependencies = TRUE)
    library(package, character.only = TRUE)
  }
}
packages <- c("sf", "tmap", "dplyr", "classInt")
lapply(packages, install_and_load)

# INKAR_2024 laden wenn es existiert
if (file.exists("INKAR_2024.RDS")) {
  tryCatch({
    INKAR <- readRDS("INKAR_2024.RDS")
    head(INKAR)
  }, error = function(e) {
    message("Error loading RDS file: ", e)
  })
} else {
  message("File INKAR_2024.RDS does not exist.")
}

# Shapefile laden
Gemeinde_sf <- sf::st_read("shp/VZ250_GEM.shp")

# Group and summarize data
Kreis_sf <- Gemeinde_sf %>%
  dplyr::group_by(ARS_K, GEN_K) %>%
  dplyr::summarize(Gemeinden = n())

Land_sf <- Gemeinde_sf %>%
  dplyr::mutate(Bundesland = substr(ARS_K, 1, 2)) %>%
  dplyr::group_by(Bundesland) %>%
  dplyr::summarize(Gemeinden = n())


# Merge data
Kreisdata_sf <- merge(x = Kreis_sf, y = INKAR, by.x = "ARS_K", by.y = "ID")

# Lineare Regression
model <- lm(AL2016_21 ~ Bev2016_21, data = Kreisdata_sf)

# Residuen berechnen
Kreisdata_sf$residuals <- residuals(model)

# Set tmap mode to plot
tmap_mode("plot")

# Create class intervals for residuals
residuals_class <- classInt::classIntervals(
  Kreisdata_sf$residuals,
  n = nclass.Sturges(Kreisdata_sf$residuals),
  style = 'pretty'
)

# Labels
breaks <- residuals_class$brks
labels <- c(paste0("<", breaks[2]),
            paste0(breaks[-c(1, length(breaks))], " bis <", breaks[-c(1, 2)]),
            paste0(">", breaks[length(breaks) - 1]))

main_title <- "Residuen der linearen Regression der Veraenderung der Arbeitslosenzahlen"
legend_title <- "Residuen"

# Plot the map
map <- tm_shape(Kreisdata_sf) +
  tm_fill("residuals",
          breaks = residuals_class$brks,
          title = legend_title,
          labels = labels,
          midpoint = NA) +
  tm_shape(Land_sf) +
  tm_borders(col = 'grey') +
  tm_shape(sf::st_centroid(Land_sf)) +
  tm_text("Bundesland", remove.overlap = TRUE, size = 0.8) +
  tm_scale_bar(breaks = c(0, 50), position = c("RIGHT", "BOTTOM")) +
  tm_layout(frame = FALSE,
            main.title = main_title,
            main.title.size = 1,
            main.title.position = c("CENTER", "TOP"),
            legend.title.size = 0.8)

tmap_save(map, "Karte_Residuen.png")