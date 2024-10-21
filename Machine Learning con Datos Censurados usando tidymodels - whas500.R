################################################################################
################################################################################
## TALLER: Machine Learning con Datos Censurados usando tidymodels
##
## LUGAR:  III Congreso & XIV Jornadas de Usuarios de R 
##         Sevilla 6, 7 y 8 de Noviembre de 2024
##
## Autor:  Jesús Herranz Valera
##                                           
################################################################################
################################################################################
 
rm(list=ls(all=TRUE))  

################################################################################
## Librerías
################################################################################

library(survival)
library(tidymodels)
library(tidyverse)
library(censored)
library(smoothHR)    ## Fichero de datos
library(survminer)   ## KM Plot 
tidymodels_prefer()


################################################################################
################################################################################
## 1.- Tratamiento de Datos
################################################################################
################################################################################

################################################################################
## Lectura del fichero, que está en la librería "smoothHR"

## Selección de columnas
xx_all <- whas500 %>% select(age, gender, hr, sysbp, diasbp, bmi, cvd, afb,
                             sho, chf, av3, miord, mitype, lenfol, fstat)
xx_all = as_tibble(xx_all)
dim(xx_all)

## Tiempo de días a meses
xx_all$lenfol = xx_all$lenfol / 30.4375

## Variable respuesta en Análisis de Supervivencia se construye con Surv()
head( Surv( xx_all$lenfol, xx_all$fstat ))


################################################################################
## Definición de la Muestra de Training y Muestra de Testing

## Se crea las especificaciones de la partición
set.seed(47) ## Se fija una semilla, para reproducir la partición
sp_split <- initial_split(xx_all, prop = 0.70, strata = fstat)
sp_split

## Se crean los dataframe de training y testing
xx_train <- training(sp_split)
xx_test <- testing(sp_split)
dim(xx_train)
dim(xx_test)

## Comprobamos la estratificación
prop.table(table(xx_all$fstat))
prop.table(table(xx_train$fstat))
prop.table(table(xx_test$fstat))

## Se borra el dataframe original
rm(xx_all)

## Se extraen todos los tiempos de supervivencia ordenados, de la muestra de training
all_time_survival <- 
  xx_train %>% 
  filter(fstat == 1) %>% 
  pull(lenfol) %>% unique %>% sort        

all_time_survival


#########################################
## Creación de la variable respuesta, para el Análisis de Supervivencia

xx_train <- xx_train %>%
  mutate(surv_var = Surv(lenfol , fstat), .keep = "unused")
xx_test <- xx_test %>%
  mutate(surv_var = Surv(lenfol , fstat), .keep = "unused")


################################################################################
## Especificaciones de la técnica de remuestreo

set.seed(52) ## Se fija una semilla, para reproducir la partición
cv_split <- vfold_cv(xx_train, v = 10, repeats=2)
cv_split


################################################################################
## KM estimator 
surv_all <- survfit( surv_var ~ 1 , data = xx_train )
surv_all 
dev.new()
ggsurvplot(surv_all, xlab="Time(months)", conf.int = FALSE,  censor = FALSE, 
           size = 1.2, palette="black", surv.median.line = "hv" ) 
        
 
################################################################################
################################################################################
## 3.- Regresión penalizada con glmnet con tidymodels
################################################################################
################################################################################

################################################################################
## Paralelización
################################################################################

## S.O. ---- Windows
R.Version()$platform

cores <- parallel::detectCores()
cores
if (!grepl("mingw32", R.Version()$platform)) {
   ## Linux
   library(doMC)
   registerDoMC(cores = cores - 1)
} else {
   ## Windows
   library(doParallel)
   cl <- makePSOCKcluster(cores - 1)
   registerDoParallel(cl)
}


################################################################################
## 3.1.- Especificaciones del modelo
################################################################################

## 1.- Se crea la receta con los pasos del pre-procesamiento
obj_rec <-
  recipe( surv_var ~ . , data=xx_train ) %>%
  step_normalize(all_numeric_predictors())
obj_rec

## Se prepara la receta
obj_prep <- prep(obj_rec, training = xx_train)

## Se aplica la receta a los dos data frame
xx_train_proc <- bake(obj_prep, xx_train)
xx_test_proc <- bake(obj_prep, xx_test)
dim(xx_train)
dim(xx_train_proc)


## 2.- Especificaciones del modelos: elastic net con datos censurados
enet_spec <- 
    proportional_hazards(penalty = tune(), mixture = tune()) %>%
    set_engine("glmnet") %>% 
    set_mode("censored regression") 
enet_spec

## Información de los parámetros del modelo elastic net
penalty()
mixture()
               
## 3.- Se crea el workflow   
wflow <- workflow() %>%
  add_model(enet_spec) %>% 
  add_recipe(obj_rec)
  
## 4.- Se crea un grid con los parámetros a explorar 
enet_grid <- grid_regular(penalty(), mixture(),
                          levels = list(penalty = 50, mixture = 6) )   
enet_grid %>% print(n=5)


################################################################################   
## 3.2.- Optimización de parámetros
################################################################################

## Se ejecuta la optimización de parámetros
keep_pred <- control_grid(save_pred = TRUE) ## salva las predicciones

tune_result_enet <- wflow %>% 
  tune_grid( resamples = cv_split, 
             grid = enet_grid, 
             control = keep_pred,
             metrics = metric_set(concordance_survival, brier_survival, 
                                  roc_auc_survival), 
             eval_time = 12 ) 


################################################################################ 
## Explorando los Resultados

## Plot
dev.new()
autoplot(tune_result_enet) +
  scale_color_viridis_d(direction = -1) +
  theme(legend.position = "top")

## Se analizan los resultados
tune_result_enet %>% 
  collect_metrics()

## Los mejores modelos, con mayores c-index
show_best(tune_result_enet, metric="concordance_survival")
## Modelo con máximo c-index
tune_best_enet <- tune_result_enet %>% select_best(metric = "concordance_survival")
tune_best_enet$penalty  
tune_best_enet$mixture

## Los mejores modelos, con mayores AUC en t=12
show_best(tune_result_enet, metric="roc_auc_survival", eval_time = 12 )


################################################################################
## 3.3.- Modelo final
################################################################################

## Se crea al workflow final
final_wflow <-
  wflow %>% 
  finalize_workflow( select_best(tune_result_enet, metric="concordance_survival") )

## Se crea al model final
enet_fit <-
  final_wflow %>%
  fit(xx_train)
enet_fit
  
## Coeficientes del modelo final  
tidy(enet_fit) %>% print(n=4)  
tidy(enet_fit) %>% filter( estimate != 0 ) %>% print(n=4)    
  
## Modelo final "glmnet"
out_glmnet <- extract_fit_engine(enet_fit)
class(out_glmnet)


################################################################################
## 3.3.- Modelo final ----- OTRA OPCIÓN
################################################################################

## Especificaciones de Elastic Net con parámetros óptimos
enet_spec_final <- 
    proportional_hazards( penalty = tune_best_enet$penalty,
                          mixture = tune_best_enet$mixture) %>%
    set_engine("glmnet")  %>% 
    set_mode("censored regression") 

## 5.- Se crea el workflow   
wflow2 <- workflow() %>%
  add_model(enet_spec_final) %>% 
  add_recipe(obj_rec)

final_fit_2 <- 
    fit (wflow2, xx_train )
tidy(final_fit_2) %>% filter ( estimate != 0 )  


################################################################################
## 3.4.- Evaluación de la Capacidad Predictiva en Testing
################################################################################

################################################################################
## Predicciones en Testing
pred_enet_lin_pred <- predict(enet_fit, xx_test, type = "linear_pred")
pred_enet_time <- predict(enet_fit, xx_test, type = "time")
pred_enet_df <- bind_cols(xx_test %>% select(surv_var), 
                          pred_enet_lin_pred, pred_enet_time ) 
pred_enet_df %>% print(n=4)

## Predicciones directas del modelo final proporcionado por "glmnet"
head( predict( out_glmnet, newx = as.matrix( xx_test_proc %>% select( - surv_var )),
               type="link", s = tune_best_enet$penalty )[ , 1])
  
################################################################################
## c-index
concordance_survival(pred_enet_df, truth = surv_var, estimate = .pred_linear_pred ) 
concordance_survival(pred_enet_df, truth = surv_var, estimate = .pred_time )

## Relación entre el predictor lineal y el tiempo de supervivencia
dev.new()
pred_enet_df %>%
  ggplot(aes(.pred_linear_pred, .pred_time)) +
  geom_point()


################################################################################
## Medidas dinámicas: Brier Scores(t) y AUC(t)

## Tiempos de evaluación a explorar, hasta 5 años, con saltos de 6 meses 
time_points <- seq( 6, 60, by=6 )

## Predicciones en Testing
pred_enet_time_df <- augment( enet_fit, xx_test, eval_time = time_points)
pred_enet_time_df %>% print(n=4)
pred_enet_time_df$.pred[[1]]   ## 10 supervivencias predichas para la observación 1

#############################
## Dynamic Brier Score
brier_scores <-
  brier_survival( pred_enet_time_df, truth = surv_var, .pred )
brier_scores

## Plot
dev.new()
brier_scores %>%
  ggplot(aes(.eval_time, .estimate)) +
  geom_hline(yintercept = 0.25, col="red", lwd = 1) +
  geom_line() +
  geom_point() +
  labs(x = "Time", y="Brier score") +
  theme_bw()

## Integrated Brier Scores
brier_survival_integrated(pred_enet_time_df, truth = surv_var, .pred )


#############################
## Dynamic ROC curves
roc_scores <-
  roc_auc_survival( pred_enet_time_df, truth = surv_var, .pred )
roc_scores

## Plot
dev.new()
roc_scores %>%
  ggplot(aes(.eval_time, .estimate)) +
  geom_hline(yintercept = 0.5, col="red", lwd = 1) +
  geom_line() +
  geom_point() +
  labs(x = "Time", y="ROC AUC") +
  theme_bw()

## Todas las curvas ROC
all_roc_curve <-
  roc_curve_survival(pred_enet_time_df, truth = surv_var, .pred )
dev.new()
autoplot(all_roc_curve)

## Curva ROC en el tiempo t = 60
pred_enet_time_60 <- augment( enet_fit, xx_test, eval_time = 60)
roc_curve_60 <-
  roc_curve_survival(pred_enet_time_60, truth = surv_var, .pred )
dev.new()
autoplot(roc_curve_60)


################################################################################
## Grupos de High / Low Risks

## Creación de los grupos en Testing con la mediana de Training
pred_enet_lp_train <- predict(enet_fit, xx_train, type = "linear_pred")
cutoff_lp <- median(pred_enet_lp_train$.pred_linear_pred)

pred_enet_lin_pred_group <- as.integer( pred_enet_lin_pred$.pred_linear_pred >= cutoff_lp )
pred_enet_lin_pred_group <- factor(pred_enet_lin_pred_group, levels=0:1,
                                   labels=c("High Risk","Low Risk"))
table(pred_enet_lin_pred_group)

## KM Plot y Test log-rank 
surv_group <- survfit( surv_var ~ pred_enet_lin_pred_group, data=xx_test )
survdiff( surv_var ~ pred_enet_lin_pred_group, data=xx_test)

dev.new()
ggsurvplot(surv_group, xlab="Time(months)", conf.int = FALSE,  censor = FALSE, 
           size=1.2, legend.labs = levels(pred_enet_lin_pred_group))


################################################################################
## Elastic Net. ALTERNATIVA: Suma de las Supervivencias en todo el rango del tiempo
################################################################################

## Predicciones en todos los tiempos de supervivencia, donde se han producido eventos
pred_enet_time_df_all <- augment( enet_fit, xx_test, eval_time = all_time_survival)

## Dataframe, con cada tiempo de cada observación en una fila, para el KM plot
pred_enet_all_survival <- pred_enet_time_df_all %>% 
  mutate(id = factor(1:nrow(xx_test))) %>% 
  unnest(cols = .pred)

## KM Plot
dev.new()  
pred_enet_all_survival %>% 
  ggplot(aes(x = .eval_time, y = .pred_survival, col = id)) +
  geom_step() +
  theme(legend.position = "none")


################################
## Se añade la suma de todos los tiempos de supervivencia
pred_enet_time_df_all <- pred_enet_time_df_all %>% 
  rowwise() %>% 
  mutate(sum_survival = sum(.pred %>% select(.pred_survival))) %>% 
  ungroup()

## Se comprueban los resultados    
pred_enet_time_df_all %>% select(sum_survival)
pred_enet_time_df_all$.pred[[1]]$.pred_survival %>% sum
pred_enet_time_df_all$.pred[[2]]$.pred_survival %>% sum
pred_enet_time_df_all$.pred[[3]]$.pred_survival %>% sum

## c-index con la predicción de "time" y con la suma de supervivencia
concordance_survival(pred_enet_df, truth = surv_var, estimate = .pred_linear_pred ) 
concordance_survival(pred_enet_time_df_all, truth = surv_var, estimate = sum_survival )

## c-index en tiempos concretos
pred_enet_surv <- predict(enet_fit, xx_test, type = "survival", eval_time = c(12,60) ) 

pred_enet_surv_12 <- pred_enet_surv %>% 
                     tidyr::unnest(col = .pred) %>% filter( .eval_time == 12 )
pred_enet_12_df <- bind_cols(xx_test %>% select(surv_var), pred_enet_surv_12 ) 
concordance_survival(pred_enet_12_df, truth = surv_var, estimate = .pred_survival ) 

pred_enet_surv_60 <- pred_enet_surv %>% 
                     tidyr::unnest(col = .pred) %>% filter( .eval_time == 60 )
pred_enet_60_df <- bind_cols(xx_test %>% select(surv_var), pred_enet_surv_60 ) 
concordance_survival(pred_enet_60_df, truth = surv_var, estimate = .pred_survival ) 


################################################################################
## Cerramos los clusters
################################################################################

stopCluster(cl)    



################################################################################
################################################################################
## 4.- Survival Trees
################################################################################
################################################################################

################################################################################
## Paralelización
################################################################################

## S.O. ---- Windows
R.Version()$platform

cores <- parallel::detectCores()
cores
if (!grepl("mingw32", R.Version()$platform)) {
   ## Linux
   library(doMC)
   registerDoMC(cores = cores - 1)
} else {
   ## Windows
   library(doParallel)
   cl <- makePSOCKcluster(cores - 1)
   registerDoParallel(cl)
}


################################################################################
## 4.1.- Especificaciones del modelo
################################################################################

## 1.- Se crea la receta con los pasos del pre-procesamiento
obj_rec_tree <-
  recipe( surv_var ~ . , data=xx_train ) 

## 2.- Especificaciones del modelos: survival tree con datos censurados
tree_spec <- 
    decision_tree( tree_depth = tune(), min_n = tune()) %>%
    set_engine("partykit") %>% 
    set_mode("censored regression") 
tree_spec

## Información de los parámetros del modelo
tree_depth()
min_n()
               
## 3.- Se crea el workflow   
wflow_tree <- workflow() %>%
  add_model(tree_spec) %>% 
  add_recipe(obj_rec_tree)
  
## 4.- Se crea un grid con los parámetros a explorar   
tree_grid <- grid_regular(tree_depth(), min_n(), levels = 4 )   
tree_grid %>% print(n=5)


################################################################################   
## 4.2.- Optimización de parámetros
################################################################################

## Se ejecuta la optimización de parámetros
tune_result_tree <- wflow_tree %>% 
  tune_grid( resamples = cv_split, 
             grid = tree_grid, 
             metrics = metric_set(concordance_survival, brier_survival, 
                                  roc_auc_survival), 
             eval_time = 12 ) 


################################################################################ 
## Explorando los Resultados

## Plot
dev.new()
autoplot(tune_result_tree) +
  scale_color_viridis_d(direction = -1) +
  theme(legend.position = "top")

## Se analizan los resultados
tune_result_tree %>% 
  collect_metrics()

## Los mejores modelos, con mayores c-index
show_best(tune_result_tree, metric="concordance_survival")
## Modelo con máximo c-index
tune_best <- tune_result_tree %>% select_best(metric = "concordance_survival")
#tune_best$cost_complexity 
tune_best$tree_depth
tune_best$min_n

## Los mejores modelos, con mayores AUC en t=12
show_best(tune_result_tree, metric="roc_auc_survival", eval_time = 12 )


################################################################################
## 4.3.- Modelo final
################################################################################

## Se crea al workflow final
final_wflow_tree <-
  wflow_tree %>% 
  finalize_workflow( select_best(tune_result_tree, metric="concordance_survival") )

## Se crea al model final
tree_fit <-
  final_wflow_tree %>%
  fit(xx_train)
tree_fit
    
## Modelo final "tree"
out_tree <- extract_fit_engine(tree_fit)
class(out_tree)

out_tree
dev.new()
plot(out_tree)


################################################################################
## 4.4.- Evaluación de la Capacidad Predictiva en Testing
################################################################################

################################################################################
## Predicciones en Testing
pred_tree_time <- predict(tree_fit, xx_test, type = "time")
pred_tree_df <- bind_cols(xx_test %>% select(surv_var), pred_tree_time ) 
pred_tree_df %>% print(n=4)
pred_tree_df %>% count(.pred_time)
  
################################################################################
## c-index
concordance_survival(pred_tree_df, truth = surv_var, estimate = .pred_time ) 


################################################################################
## Medidas dinámicas: Brier Scores(t) y AUC(t)

## Tiempos de evaluación a explorar, hasta 5 años, con saltos de 6 meses 
time_points <- seq( 6, 60, by=6 )

## Predicciones en Testing
pred_tree_time_df <- augment( tree_fit, xx_test, eval_time = time_points)
pred_tree_time_df %>% print(n=4)
pred_tree_time_df$.pred[[1]]   ## 10 supervivencias predichas para la observación 1

#############################
## Dynamic Brier Score
brier_scores <-
  brier_survival( pred_tree_time_df, truth = surv_var, .pred )
brier_scores

## Plot
dev.new()
brier_scores %>%
  ggplot(aes(.eval_time, .estimate)) +
  geom_hline(yintercept = 0.25, col="red", lwd = 1) +
  geom_line() +
  geom_point() +
  labs(x = "Time", y="Brier score") +
  theme_bw()

## Integrated Brier Scores
brier_survival_integrated(pred_tree_time_df, truth = surv_var, .pred )


#############################
## Dynamic ROC curves
roc_scores <-
  roc_auc_survival( pred_tree_time_df, truth = surv_var, .pred )
roc_scores

## Plot
dev.new()
roc_scores %>%
  ggplot(aes(.eval_time, .estimate)) +
  geom_hline(yintercept = 0.5, col="red", lwd = 1) +
  geom_line() +
  geom_point() +
  labs(x = "Time", y="ROC AUC") +
  theme_bw()

## Todas las curvas ROC
all_roc_curve <-
  roc_curve_survival(pred_tree_time_df, truth = surv_var, .pred )
dev.new()
autoplot(all_roc_curve)

## Curva ROC en el tiempo t = 60
pred_tree_time_60 <- augment( tree_fit, xx_test, eval_time = 60)
roc_curve_60 <-
  roc_curve_survival(pred_tree_time_60, truth = surv_var, .pred )
dev.new()
autoplot(roc_curve_60)


################################################################################
## Survival Trees. ALTERNATIVA: Suma de las Supervivencias en todo el rango del tiempo
################################################################################

## Predicciones en todos los tiempos de supervivencia, donde se han producido eventos
pred_tree_time_df_all <- augment( tree_fit, xx_test, eval_time = all_time_survival)

## Dataframe, con cada tiempo de cada observación en una fila, para el KM plot
pred_tree_all_survival <- pred_tree_time_df_all %>% 
  mutate(id = factor(1:nrow(xx_test))) %>% 
  unnest(cols = .pred)
dim(pred_tree_all_survival)
length(all_time_survival)
nrow(xx_test)
length(all_time_survival) * nrow(xx_test)

## KM Plot
dev.new()  
pred_tree_all_survival %>% 
  ggplot(aes(x = .eval_time, y = .pred_survival, col = id)) +
  geom_step() +
  theme(legend.position = "none")


################################
## Se añade la suma de todos los tiempos de supervivencia
pred_tree_time_df_all <- pred_tree_time_df_all %>% 
  rowwise() %>% 
  mutate(sum_survival = sum(.pred %>% select(.pred_survival))) %>% 
  ungroup()

## Se comprueban los resultados  
pred_tree_time_df_all %>% select(sum_survival)
pred_tree_time_df_all$.pred[[1]]$.pred_survival %>% sum
pred_tree_time_df_all$.pred[[4]]$.pred_survival %>% sum
pred_tree_time_df_all$.pred[[6]]$.pred_survival %>% sum
pred_tree_time_df_all$.pred[[9]]$.pred_survival %>% sum

## c-index con la predicción de "time" y con la suma de supervivencia
concordance_survival(pred_tree_time_df_all, truth = surv_var, estimate = .pred_time ) 
concordance_survival(pred_tree_time_df_all, truth = surv_var, estimate = sum_survival )

pred_tree_time_df_all %>% select(.pred_time) %>% table
pred_tree_time_df_all %>% select(sum_survival) %>% table


################################################################################
## Cerramos los clusters
################################################################################

stopCluster(cl)    



################################################################################
################################################################################
## 5.- Random Forest
################################################################################
################################################################################

################################################################################
## Paralelización
################################################################################

## S.O.
R.Version()$platform

cores <- parallel::detectCores()
cores
if (!grepl("mingw32", R.Version()$platform)) {
   ## Linux
   library(doMC)
   registerDoMC(cores = cores - 1)
} else {
   ## Windows
   library(doParallel)
   cl <- makePSOCKcluster(cores - 1)
   registerDoParallel(cl)
}


################################################################################
## 5.1.- Especificaciones del modelo
################################################################################

## 1.- Se crea la receta con los pasos del pre-procesamiento
obj_rec_rf <-
  recipe( surv_var ~ . , data=xx_train ) 
  
## 2.- Especificaciones del modelos: random forest con datos censurados
rf_spec <- 
    rand_forest(trees = 200, mtry = tune(), min_n = tune()) %>%
    set_engine("partykit") %>% 
    set_mode("censored regression") 
rf_spec

## Parámetros del random forest
trees()
min_n()
mtry()
extract_parameter_set_dials(rf_spec)
extract_parameter_set_dials(rf_spec) %>%
  finalize(xx_train)

## 3.- Se crea el workflow   
wflow_rf <- workflow() %>%
  add_model(rf_spec) %>% 
  add_recipe(obj_rec_rf)

## 4.- Se crea un grid con los parámetros a explorar 
rf_grid <- crossing( mtry = c(4, 6, 8), min_n = c(10, 20, 30, 40))


################################################################################   
## 5.2.- Optimización de parámetros
################################################################################

## Se ejecuta la optimización de parámetros
tune_result_rf <- wflow_rf %>% 
  tune_grid( resamples = cv_split, 
             grid = rf_grid, 
             metrics = metric_set(concordance_survival, brier_survival, 
                                  roc_auc_survival), 
             eval_time = 12 ) 


################################################################################ 
## Explorando los Resultados

## Plot
autoplot(tune_result_rf) +
  scale_color_viridis_d(direction = -1) +
  theme(legend.position = "top")

## Los mejores modelos, con máximo c-index
show_best(tune_result_rf, metric="concordance_survival")
tune_best_rf <- tune_result_rf %>% select_best(metric = "concordance_survival")
tune_best_rf$mtry  
tune_best_rf$min_n


################################################################################
## 5.3.- Modelo final
################################################################################

## Se crea al workflow final
final_wflow_rf <-
  wflow_rf %>% 
  finalize_workflow( select_best(tune_result_rf, metric="concordance_survival") )

## Se crea al modelo final
rf_fit <-
  final_wflow_rf %>%
  fit(xx_train)
rf_fit


################################################################################
## 5.4.- Evaluación de la Capacidad Predictiva en Testing
################################################################################

################################################################################
## Predicciones en Testing
pred_rf_time <- predict(rf_fit, xx_test, type = "time")
pred_rf_df <- bind_cols(xx_test %>% select(surv_var), pred_rf_time) 
pred_rf_df %>% print(n=4)
  
################################################################################
## c-index
concordance_survival(pred_rf_df, truth = surv_var, estimate = .pred_time ) 


################################################################################
## Medidas dinámicas: Brier Scores(t) y AUC(t)

## Tiempos de evaluación a explorar, hasta 5 años, con saltos de 6 meses 
time_points <- seq( 6, 60, by=6 )

## Predicciones en Testing
pred_rf_time_df <- augment( rf_fit, xx_test, eval_time = time_points)

#############################
## Dynamic Brier Score
brier_scores <-
  brier_survival( pred_rf_time_df, truth = surv_var, .pred )
brier_scores

## Plot
dev.new()
brier_scores %>%
  ggplot(aes(.eval_time, .estimate)) +
  geom_hline(yintercept = 0.25, col="red", lwd = 1) +
  geom_line() +
  geom_point() +
  labs(x = "Time", y="Brier score") +
  theme_bw()

## Integrated Brier Scores
brier_survival_integrated(pred_rf_time_df, truth = surv_var, .pred )


#############################
## Dynamic ROC curves
roc_scores <-
  roc_auc_survival( pred_rf_time_df, truth = surv_var, .pred )
roc_scores

## Plot
dev.new()
roc_scores %>%
  ggplot(aes(.eval_time, .estimate)) +
  geom_hline(yintercept = 0.5, col="red", lwd = 1) +
  geom_line() +
  geom_point() +
  labs(x = "Time", y="ROC AUC") +
  theme_bw()


################################################################################
## Random Forest. ALTERNATIVA: Suma de las Supervivencias en todo el rango del tiempo
################################################################################

## Predicciones en todos los tiempos de supervivencia, donde se han producido eventos
pred_rf_time_df_all <- augment( rf_fit, xx_test, eval_time = all_time_survival)

## Dataframe, con cada tiempo de cada observación en una fila, para el KM plot
pred_rf_all_survival <- pred_rf_time_df_all %>% 
  mutate(id = factor(1:nrow(xx_test))) %>% 
  unnest(cols = .pred)

## KM Plot
dev.new()  
pred_rf_all_survival %>% 
  ggplot(aes(x = .eval_time, y = .pred_survival, col = id)) +
  geom_step() +
  theme(legend.position = "none")

## KM Plot de las que tienen mejor supervivencia
dev.new()  
pred_rf_all_survival %>% filter ( .pred_time > 77 ) %>% 
  ggplot(aes(x = .eval_time, y = .pred_survival, col = id)) +
  geom_step() +
  theme(legend.position = "none")


################################
## Se añade la suma de todos los tiempos de supervivencia
pred_rf_time_df_all <- pred_rf_time_df_all %>% 
  rowwise() %>% 
  mutate(sum_survival = sum(.pred %>% select(.pred_survival))) %>% 
  ungroup()

## Se comprueban los resultados    
pred_rf_time_df_all %>% select(sum_survival)
pred_rf_time_df_all$.pred[[1]]$.pred_survival %>% sum
pred_rf_time_df_all$.pred[[2]]$.pred_survival %>% sum
pred_rf_time_df_all$.pred[[3]]$.pred_survival %>% sum

## c-index con la predicción de "time" y con la suma de supervivencia
concordance_survival(pred_rf_time_df_all, truth = surv_var, estimate = .pred_time ) 
concordance_survival(pred_rf_time_df_all, truth = surv_var, estimate = sum_survival )

pred_rf_time_df_all %>% select(.pred_time)%>% round(1) %>% table %>% tail
pred_rf_time_df_all %>% select(sum_survival)%>% round(1) %>% table %>% tail


## c-index en tiempos concretos
pred_rf_surv <- predict(rf_fit, xx_test, type = "survival", eval_time = c(12,60) ) 

pred_rf_surv_12 <- pred_rf_surv %>% 
                   tidyr::unnest(col = .pred) %>% filter( .eval_time == 12 )
pred_rf_12_df <- bind_cols(xx_test %>% select(surv_var), pred_rf_surv_12 ) 
concordance_survival(pred_rf_12_df, truth = surv_var, estimate = .pred_survival ) 

pred_rf_surv_60 <- pred_rf_surv %>% 
                   tidyr::unnest(col = .pred) %>% filter( .eval_time == 60 )
pred_rf_60_df <- bind_cols(xx_test %>% select(surv_var), pred_rf_surv_60 ) 
concordance_survival(pred_rf_60_df, truth = surv_var, estimate = .pred_survival ) 


################################################################################
## Cerramos los clusters
################################################################################

stopCluster(cl) 



################################################################################
################################################################################
## ANEXO 1: Regresión de Cox con tidymodels
################################################################################
################################################################################ 
 
################################################################################
## 1- Regresión de Cox con survival
################################################################################

out_cox <- coxph( surv_var ~ . , data = xx_train )
summary(out_cox)                


################################################################################
## 2.- Regresión de Cox con tidymodels
################################################################################

## Se crean las especificaciones del modelo
ph_spec <- 
  proportional_hazards() %>%
  set_engine("survival") %>% 
  set_mode("censored regression") 
ph_spec

## Se ajusta el model
ph_fit <- ph_spec %>% 
  fit( surv_var ~ ., data = xx_train)
ph_fit
tidy(ph_fit)


################################################################################
## 3.- Prediciones de Regresión de Cox con survival
################################################################################ 

## Linear Predictor con survival
#pred_cox_lin_pred <- predict(out_cox, xx_test, type = "lp" )
pred_cox_lin_pred <- predict(out_cox, xx_test, type = "lp", reference = "zero" )
head(pred_cox_lin_pred)

## Función de supervivencia predicha para la Observacion 5 (buen pronóstico, azul)
dev.new()
surv_obs_5 <- survfit( out_cox, newdata = xx_test %>% slice(5), se=F ) 
plot( surv_obs_5, mark.time=F, conf.int=F, col="blue", lwd=2 )   

## Función de supervivencia predicha para la Observacion 6 (mal pronóstico, rojo)
surv_obs_6 <- survfit( out_cox, newdata = xx_test %>% slice(6), se=F ) 
lines( surv_obs_6, mark.time=F, conf.int=F, col="red", lwd=2 )   


################################################################################
## 4.- Prediciones de Regresión de Cox con tidymodels
################################################################################ 

## predict tipo "survival" de tidymodels
pred_ph_surv <- predict(ph_fit, xx_test, type = "survival", 
                        eval_time=seq(6, 60, by=6)) 
pred_ph_surv %>% print(n=5)

## predict tipo "survival" en tiempos concretos (1 y 5 años)
pred_ph_surv_12 <- pred_ph_surv %>% 
                   tidyr::unnest(col = .pred) %>% filter( .eval_time == 12 )
pred_ph_surv_60 <- pred_ph_surv %>% 
                   tidyr::unnest(col = .pred) %>% filter( .eval_time == 60 )
pred_ph_surv_12 %>% print(n=6)
pred_ph_surv_60 %>% print(n=6)
## Añadimos los valores de las observaciones 5 y 6 al gráfico de supervivencia
points( pred_ph_surv_12 %>% slice(5,6), pch=16, cex=1.2 )
points( pred_ph_surv_60 %>% slice(5,6), pch=16, cex=1.2 )
                         
## predict tipo "linear_pred" y "time" de tidymodels  
pred_ph_lin_pred <- predict(ph_fit, xx_test, type = "linear_pred")
pred_ph_time <- predict(ph_fit, xx_test, type = "time")
pred_ph_df <- bind_cols(xx_test %>% select(surv_var), 
                        pred_ph_lin_pred, pred_ph_time ) 
pred_ph_df %>% print(n=5)
head(pred_cox_lin_pred)


################################################################################
## c-index en Testing
concordance_survival(pred_ph_df, truth = surv_var, estimate = .pred_linear_pred ) 
concordance_survival(pred_ph_df, truth = surv_var, estimate = .pred_time )

## Relación entre el predictor lineal y el tiempo de supervivencia
dev.new()
pred_ph_df %>%
  ggplot(aes(.pred_linear_pred, .pred_time)) +
  geom_point()


################################################################################
## 5.- Capacidad Predictiva con Técnicas de Remuestreo
################################################################################ 

#####################################
## Paralelización

## S.O.
R.Version()$platform

cores <- parallel::detectCores()
cores
if (!grepl("mingw32", R.Version()$platform)) {
   ## Linux
   library(doMC)
   registerDoMC(cores = cores - 1)
} else {
   ## Windows
   library(doParallel)
   cl <- makePSOCKcluster(cores - 1)
   registerDoParallel(cl)
}

## Workflow
ph_wflow <- workflow() %>%
  add_model(ph_spec) %>% 
  add_formula(surv_var ~ .) 

## Remuestreo
ph_res <- ph_wflow %>% 
  fit_resamples( resamples = cv_split, 
                 metrics = metric_set(concordance_survival) )

## Resultados
collect_metrics(ph_res)

#####################################
## Cerramos los clusters

stopCluster(cl) 



################################################################################
################################################################################
## ANEXO 3: Cross-Validated Predictions
################################################################################
################################################################################

################################################################################
## Predicciones del proceso de tune. Cross-Validated Predictions
assess_res <- collect_predictions(tune_result_enet) 
assess_res %>% print(n=4)
349 * 300 * 2    ## 349 obs. en training, 300 parámetros, 2 repeticiones de CV


## Predicciones con summarize ( 1 valor por observación )
assess_res_summ <- collect_predictions(tune_result_enet, summarize=TRUE) 
assess_res_summ %>% print(n=4)

## Predicciones del mejor modelo
assess_res_summ_best <-
  assess_res_summ %>% 
  filter( penalty == tune_best_enet$penalty, 
          mixture == tune_best_enet$mixture )
assess_res_summ_best %>% print(n=4)

## Se calcula el c-index
concordance_survival( assess_res_summ_best, truth = surv_var, .pred_time )


## Alternativa para crear las CV predictions
## Se seleccionan las predicciones con los parámetros óptimos
#assess_res_best <-
#  assess_res %>% 
#  filter( penalty == tune_best_enet$penalty, 
#          mixture == tune_best_enet$mixture )
#assess_res_best %>% print(n=4)
#
#cv_pred_df <-
#  assess_res_best %>%
#  group_by(.row) %>%
#  summarize(.pred_time = mean(.pred_time) )
#cv_pred_df <- bind_cols(cv_pred_df, xx_train %>% select(surv_var)) 


################################################################################
## Cross-Validated Predictions - Grupos de High / Low Risks

## Creación de los grupos con la mediana de Training
pred_enet_time_train <- predict(enet_fit, xx_train, type = "time")
cutoff_time <- median(pred_enet_time_train$.pred_time)

pred_enet_time_group <- as.integer( assess_res_summ_best$.pred_time >= cutoff_time )
pred_enet_time_group <- factor(pred_enet_time_group, levels=0:1,
                                   labels=c("High Risk","Low Risk"))
table(pred_enet_time_group)

## Cross-Validated KM Plot y Test log-rank, en xx_train 
surv_group <- survfit( surv_var ~ pred_enet_time_group, data=xx_train )
survdiff( surv_var ~ pred_enet_time_group, data=xx_train)

dev.new()
ggsurvplot(surv_group, xlab="Time(months)", conf.int = FALSE,  censor = FALSE, 
           size=1.2, legend.labs = levels(pred_enet_time_group))


################################################################################
## Cross-Validated Predictions - AUC

## AUC
roc_scores <-
  roc_auc_survival(assess_res_summ_best, truth = surv_var, .pred )
roc_scores

## Todas las curvas ROC
all_roc_curve <-
  roc_curve_survival(assess_res_summ_best, truth = surv_var, .pred )
dev.new()
autoplot(all_roc_curve)


################################################################################
################################################################################
################################################################################
## OTROS COMPLEMENTOS
################################################################################
################################################################################
################################################################################

################################################################################
## Cálculo del c-index con Remuestreo, del modelo con los parámetros óptimos
## -- Suele ser menor que el obtenido en el proceso de la CV de optimización
################################################################################

#set.seed(82) ## Se fija una semilla, para reproducir la partición
#cv_split_2 <- vfold_cv(xx_train, v = 10, repeats=5)
#cv_split_2

## Remuestreo
#keep_pred <- control_grid(save_pred = TRUE) ## salva las predicciones

#resamp_enet_final <- final_wflow %>% 
#  fit_resamples( resamples = cv_split_2, 
#                control = keep_pred,
#                  metrics = metric_set(concordance_survival, brier_survival, 
#                                       roc_auc_survival), 
#                 eval_time = time_points ) 
#collect_metrics(resamp_enet_final)
#collect_metrics(resamp_enet_final) %>% filter( .metric=="concordance_survival" )



