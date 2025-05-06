# Env management
install.packages("renv")

# renv::init(restart = TRUE) # initialize renv
# renv::install("rstudio/renv", prompt = FALSE) # installing dev version
renv::consent(provided = TRUE) # consent to write to filesystem
# renv::restore(packages = "renv", prompt = FALSE)

# renv::settings$snapshot.type("implicit") # capture all dependencies

renv::restore(prompt = FALSE) # install all packages for project from renv.lock file
# END Env management

# Manual Env management
# Installing and loading some packages
install.packages("pak", repos = sprintf("https://r-lib.github.io/p/pak/stable/%s/%s/%s", .Platform$pkgType, R.Version()$os, R.Version()$arch))

package_list <- c(
  "here", # use for paths creation
  "maps",
  "shinydashboardPlus",
  "shinyjs",
  "shinyscreenshot",
  "shinyWidgets",
  "sparkline",
  "tidygraph", # for creating networks
  "ggraph", # plotting networks
  "wordcloud2",
  "visNetwork",
  "openxlsx",
  "sparkline",
  "svglite"
)

pak::pkg_install("massimoaria/bibliometrix@master", dependencies = TRUE)
pak::pkg_install("dkahle/ggmap", dependencies = TRUE)

for (p in package_list) {
  if (p %in% installed.packages() == FALSE) {
    pak::pkg_install(p, dependencies = TRUE)
  }
  library(p, character.only = TRUE)
}

# END Manual Env management

library(bibliometrix)

biblioshiny()

# devtools::install_github("uname/repo", subdir="pkg/folder", force = TRUE)

# Combine from multiple sources
# ieee = convert2df(file.choose(), dbsource = "scopus", format = "bibtex")

wos <- convert2df(file.choose(), dbsource = "wos", format = "plaintext")

scopus <- convert2df(file.choose(), dbsource = "scopus", format = "csv")

mergedSourcesWithDup <- mergeDbSources(wos, scopus, remove.duplicated = FALSE)
write.csv(mergedSourcesWithDup, "merge-dedup/XAI-M-merge-WoS-Scopus-WithDup.csv")

mergedSources <- mergeDbSources(wos, scopus, remove.duplicated = TRUE)
write.csv(mergedSources, "merge-dedup/XAI-M-merge-WoS-Scopus-dedup.csv")

# scopus = convert2df(file.choose(), dbsource = "scopus", format = "bibtex")

# mergedSources = mergeDbSources(wos, scopus, ieee, remove.duplicated = TRUE)
# mergedSourcesWithDup = mergeDbSources(wos, scopus, ieee, remove.duplicated = FALSE)
# write.csv(mergedSourcesWithDup, "merge-dedup/merge-IEEE-WoS-Scopus-WithDup.csv")

# mergedSources = mergeDbSources(wos, scopus, ieee, remove.duplicated = TRUE)
# write.csv(mergedSources, "merge-dedup/merge-IEEE-WoS-Scopus-dedup.csv")

# mergedSources = mergeDbSources(wos, scopus, remove.duplicated = TRUE)
# write.csv(mergedSources, "merge-dedup/merge-dedup.csv")

# NB
# If merge is .CSV load data as Scopus database

## KEEP AT END OF FILE
renv::snapshot(prompt = FALSE) # note newly installed packages
# renv::upgrade(prompt = FALSE) # upgrade renv version
# renv::update(prompt = FALSE) # update packages

# To check where packages are installed
.libPaths()

# To clean all R packages
# remove.packages(pkgs=row.names(x=installed.packages()))
