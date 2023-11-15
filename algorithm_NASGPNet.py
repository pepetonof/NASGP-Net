# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:06:05 2023

@author: josef
"""

from deap import tools
import pickle
import random
# import numpy as np
from datetime import datetime, timedelta
# import matplotlib.pyplot as plt
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#%%Variation Operators
def MateMutation(population, toolbox, lambda_, cxpb, mutpb):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")
    mut_options = ['shrink','replace','uniform']
    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = list(map(toolbox.clone, random.sample(population, 2)))
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            mut_op = random.choice(mut_options)
            ind = toolbox.clone(random.choice(population))
            # if mut_op=='ephem':
            #     ind, = toolbox.mutate_eph(ind)
            if mut_op=='shrink':
                ind, = toolbox.mutate_shrink(ind)
            elif mut_op=='replace':
                ind, = toolbox.mutate_replace(ind)
            elif mut_op=='uniform':
                # print('uniform')
                ind, = toolbox.mutate_uniform(ind)
            # ind, = toolbox.mutate(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction?
            ind = toolbox.clone(random.choice(population))
            ind, = toolbox.mutate_eph(ind)
            del ind.fitness.values
            offspring.append(ind)
            # offspring.append(random.choice(population))

    return offspring

#%%Checkpoint
def checkpoint(generation, population, offspring,
               invalid_ind, idx, elitism_inds,
               no_evs, delta_t, cache,
               halloffame, logbook,
               rndstate, ruta):
    
    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(generation=generation, population=population, offspring=offspring,
               invalid_ind=invalid_ind, idx=idx, elitism_inds=elitism_inds,
               no_evs=no_evs, delta_t=delta_t, cache=cache, #archive=archive,
               halloffame=halloffame, logbook=logbook,
               rndstate=rndstate)
    with open(ruta + '/'+ "checkpoint_evo.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)
    del cp_file
    
    return

#%%Set attributes of individuals using cache and update it
def assign_attributes(_ind, key, cache, toolbox, surrogate=None):
    # _ind=toolbox.clone(ind)
    if key in cache:
        #Assign attributes from cache
        _ind.fitness.values = cache[key].fitness.values
        _ind.dice = cache[key].dice
        _ind.iou = cache[key].iou
        _ind.hd95 = cache[key].hd
        _ind.hd = cache[key].hd95
        # _ind.nsd = cache[key].nsd
        _ind.params = cache[key].params
                
        print('Syntax tree:\t', str(_ind), round(_ind.fitness.values[0],3), round(_ind.dice,3), _ind.params, "\t in cache")
        
    else:
        # #Assign attributes from the original objective function
        # if surrogate == None:
        fit = toolbox.evaluate(_ind)
        _ind.fitness.values = fit[0],
        _ind.dice = fit[1]
        _ind.iou = fit[2]
        _ind.hd = fit[3]
        _ind.hd95 = fit[4]
        # _ind.nsd = fit[5]
        _ind.params = fit[5]
        
        print('Syntax tree:\t', str(_ind), round(_ind.fitness.values[0], 3), round(_ind.dice,3), _ind.params, "\t in original")
            
        # #Assign attributes from the surrogate model
        # else:
        #     fit = toolbox.evaluate_surrogate(_ind, surrogate)
        #     _ind.fitness.values = fit[0],
        #     _ind.dice = fit[1]
        #     _ind.iou = fit[2]
        #     _ind.hd = fit[3]
        #     _ind.hd95 = fit[4]
        #     # _ind.nds = fit[5]
        #     _ind.params = fit[5]
            
        #     print('Syntax tree:\t', str(_ind), round(_ind.fitness.values[0],3), round(_ind.dice,3), _ind.params, "\t in surrogate")    
    return _ind

#%%NASGP-Net Algorithm
def eaNASGPNet(pop_size, toolbox, cxpb, mutpb, ngen, nelit,
                 # gen_update, p, m,
                 ruta, checkpoint_name,
                 stats=None, halloffame=None, verbose_evo=__debug__,
                 ):
    
    ####Take time. Keep delta_t, no matters loose t
    init_time=datetime.now()
    delta_t=timedelta(seconds=0)
    
    ####If checkpoint then load data from the file "/checkpoint"
    if checkpoint_name:
        print('recovering...', ruta)
        with open(ruta+'/'+checkpoint_name, "rb") as cp_file:
            cp = pickle.load(cp_file)
        # print(cp.keys())
        start_gen    = cp["generation"]
        population   = cp["population"]
        offspring    = cp["offspring"]
        invalid_ind  = cp["invalid_ind"]
        idx          = cp["idx"]
        elitism_inds = cp["elitism_inds"]
        
        no_evs       = cp["no_evs"]
        delta_t      = cp["delta_t"]
        cache        = cp["cache"]
        
        halloffame   = cp["halloffame"]
        logbook      = cp["logbook"]
        random.setstate(cp["rndstate"])
        
    else:
        ####Else start a new evolution
        population = toolbox.population(n=pop_size)
        start_gen  = 0
        halloffame = tools.HallOfFame(maxsize=nelit)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'time', 'best', 
                          'best_dice',
                          'best_iou',
                          'best_hd',
                          'best_hd95',
                          # 'best_nds',
                          'best_params'] + (stats.fields if stats else [])
        offspring = []
        elitism_inds = []
        
        ###Count the number of evaluations and evaluated individuals
        idx=0
        no_evs=0
        cache={}
        
        ###Individuals to evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        
    #%%%%Start option 1
    #Si no ha terminado con invalid ind y sigue en la generacion 0
    if idx<len(invalid_ind): #and start_gen == 0:
        #Evaluate using cache or surrogate model;
        while idx < len(invalid_ind):
            ind = invalid_ind[idx]
            key = toolbox.identifier(ind)
            
            #Predict using original fitness function
            # ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
            
            if key in cache:
                #Assign attributes from cache
                ind.fitness.values = cache[key].fitness.values
                ind.dice = cache[key].dice
                ind.iou = cache[key].iou
                ind.hd95 = cache[key].hd
                ind.hd = cache[key].hd95
                # _ind.nsd = cache[key].nsd
                ind.params = cache[key].params
                        
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.dice,3), ind.params, "\t in cache")
                
            else:
                # #Assign attributes from the original objective function
                # if surrogate == None:
                fit = toolbox.evaluate(ind)
                ind.fitness.values = fit[0],
                ind.dice = fit[1]
                ind.iou = fit[2]
                ind.hd = fit[3]
                ind.hd95 = fit[4]
                # _ind.nsd = fit[5]
                ind.params = fit[5]
                
                #Add to cache
                cache[key]=ind
                
                ####Increment the number of evaluations when original objective function is used
                ####and key is not in cache
                no_evs+=1
                
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.dice,3), ind.params, "\t in original")
            
            # #Add to cache when original objective function is used
            # if key not in cache:
            #     #Add to cache
            #     cache[key]=ind
                
            #     ####Increment the number of evaluations when original objective function is used
            #     ####and key is not in cache
            #     no_evs+=1
                
            ####Increment the number of evaluated individuals from invalid ind
            idx+=1
            
            ####Take time every evaluation
            t = datetime.now()
            delta_t += (t - init_time)
            init_time = t #Keep delta_t, no matters loose t
            
            ####Checkpoint every evaluation and every generation
            checkpoint(generation=start_gen, population=population, offspring=offspring,
                       invalid_ind=invalid_ind, idx=idx, elitism_inds=elitism_inds,
                       no_evs=no_evs, delta_t=delta_t, cache=cache,
                       halloffame=halloffame, logbook=logbook,
                       rndstate=random.getstate(), ruta=ruta)
            
            print(f"{idx}/{len(invalid_ind)}", start_gen, ruta.split("/").pop(), delta_t)
         
        best_ind = tools.selBest(population, 1)[0]
        print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.dice,3),best_ind.params, start_gen)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        
        logbook.record(gen=start_gen, nevals=no_evs, time=delta_t,
                       best = str(best_ind), 
                       best_dice=best_ind.dice,
                       best_iou=best_ind.iou,
                       best_hd=best_ind.hd,
                       best_hd95=best_ind.hd95,
                       # best_nds=best_ind.nds,
                       best_params=best_ind.params,
                       **record)
        
        #For print
        if verbose_evo:
            print(logbook.stream)
        
    # #%%%Start option 2
    # #Si no ha terminado con 'invalid_ind' en la sesión anterior pero ya ha
    # #finalizado con generación 0
    # if idx<len(invalid_ind) and start_gen > 0:
    #     #Evaluate using cache or surrogate model;
    #     while idx < len(invalid_ind):
    #         ind = invalid_ind[idx]
    #         key = toolbox.identifier(ind)
            
    #         #Precird the fitness valies of new inds via surrogate model
    #         ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
            
    #         #Add to cache when original objective function is used
    #         if key not in cache:
    #             #Add to cache
    #             cache[key]=ind
                
    #             ##Increment the number of evaluations when original objective function is used
    #             ##and key is not in cache
    #             no_evs+=1
                
    #         ####Increment the number of evaluated individuals from invalid ind
    #         idx+=1
            
    #         ####Take time every evaluation
    #         t = datetime.now()
    #         delta_t += (t - init_time)
    #         init_time = t #Keep delta_t, no matters loose t
            
    #         ####Checkpoint every evaluation and every generation
    #         checkpoint(generation=start_gen, population=population, offspring=offspring,
    #                    invalid_ind=invalid_ind, idx=idx, elitism_inds=elitism_inds,
    #                    no_evs=no_evs, delta_t=delta_t, cache=cache,
    #                    halloffame=halloffame, logbook=logbook,
    #                    rndstate=random.getstate(), ruta=ruta)
            
    #         print(f"{idx}/{len(invalid_ind)}", start_gen, ruta.split("/").pop())
        
    #     best_ind = tools.selBest(population, 1)[0]
    #     print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.dice,3),best_ind.params, start_gen)
        
    #     # Append the current generation statistics to the logbook
    #     record = stats.compile(population) if stats else {}
    #     logbook.record(gen=start_gen, nevals=no_evs, time=delta_t,
    #                    best = str(best_ind),
    #                    best_dice=best_ind.dice,
    #                    best_iou=best_ind.iou,
    #                    best_hd=best_ind.hd,
    #                    best_hd95=best_ind.hd95,
    #                    best_nds=best_ind.nds,
    #                    best_params=best_ind.params,
    #                    **record)
    
    #%%%Start option 3
    #Si ya ha terminado con "invalid_ind" en la sesión anterior o si no checkpoint_name, 
    #entonces continúa normalmente (pasa a la siguiente generación)
    # print("Continue Here\n\n")
    if idx==len(invalid_ind) or checkpoint_name==False:
        start_gen+=1
        
        for gen in range(start_gen, ngen+1):
            print('\nGen:\t', gen)
            
            population_for_eli=[toolbox.clone(ind) for ind in population]
            elitism_inds = toolbox.selectElitism(population_for_eli, k=nelit)
            offspring = toolbox.select(population, len(population) - nelit)
            offspring = MateMutation(offspring, toolbox, len(offspring), cxpb, mutpb)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            idx = 0
            
            #Evaluate using cache or original model;
            while idx < len(invalid_ind):
                ind = invalid_ind[idx]
                key = toolbox.identifier(ind)
                
                # ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
                if key in cache:
                    #Assign attributes from cache
                    ind.fitness.values = cache[key].fitness.values
                    ind.dice = cache[key].dice
                    ind.iou = cache[key].iou
                    ind.hd95 = cache[key].hd
                    ind.hd = cache[key].hd95
                    # _ind.nsd = cache[key].nsd
                    ind.params = cache[key].params
                            
                    print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.dice,3), ind.params, "\t in cache")
                    
                else:
                    # #Assign attributes from the original objective function
                    # if surrogate == None:
                    fit = toolbox.evaluate(ind)
                    ind.fitness.values = fit[0],
                    ind.dice = fit[1]
                    ind.iou = fit[2]
                    ind.hd = fit[3]
                    ind.hd95 = fit[4]
                    # _ind.nsd = fit[5]
                    ind.params = fit[5]
                    
                    #Add to cache
                    cache[key]=ind
                    
                    ####Increment the number of evaluations when original objective function is used
                    ####and key is not in cache
                    no_evs+=1
                    
                    print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.dice,3), ind.params, "\t in original")
                
                
                # #Add to cache when original objective function is used
                # if key not in cache:
                #     #Add to cache
                #     cache[key]=ind
                    
                #     ####Increment the number of evaluations when original objective function is used
                #     ####and key is not in cache
                #     no_evs+=1
                
                ####Increment the number of evaluated individuals from invalid ind
                idx+=1
                
                ####Take time every evaluation
                t = datetime.now()
                delta_t += (t - init_time)
                init_time = t #Keep delta_t, no matters loose t
                
                ####Checkpoint every evaluation and every generation
                checkpoint(generation=gen, population=population, offspring=offspring,
                           invalid_ind=invalid_ind, idx=idx, elitism_inds=elitism_inds,
                           no_evs=no_evs, delta_t=delta_t, cache=cache,
                           halloffame=halloffame, logbook=logbook,
                           rndstate=random.getstate(), ruta=ruta)
                
                print(f"{idx}/{len(invalid_ind)}", gen, ruta.split("/").pop(), delta_t)
                
            #Back the elitism individuals to population
            offspring.extend(elitism_inds)
            
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
            
            #Replace the curren population by the offspring
            population[:] = offspring
            
            ####Take time every evaluation
            t = datetime.now()
            delta_t += (t - init_time)
            init_time = t #Keep delta_t, no matters loose t
            
            ####Checkpoint every evaluation and every generation
            checkpoint(generation=gen, population=population, offspring=offspring,
                       invalid_ind=invalid_ind, idx=idx, elitism_inds=elitism_inds,
                       no_evs=no_evs, delta_t=delta_t, cache=cache,
                       halloffame=halloffame, logbook=logbook,
                       rndstate=random.getstate(), ruta=ruta)
            
            #Take the best individual after finishes each generation
            #and print their attributes
            best_ind = tools.selBest(population, 1)[0]
            print(str(best_ind), best_ind.fitness.values, best_ind.dice, best_ind.params)
            print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.dice,3), best_ind.params, gen, delta_t)
            
            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=no_evs, time=delta_t,
                           best = str(best_ind),
                           best_dice=best_ind.dice,
                           best_iou=best_ind.iou,
                           best_hd=best_ind.hd,
                            best_hd95=best_ind.hd95,
                           # best_nds=best_ind.nds,
                           best_params=best_ind.params,
                           **record)
            
            #For print
            if verbose_evo:
                print(logbook.stream)
        
        #Save logbook as .pkl
        # with open(ruta + "/logbook.pkl", "wb") as log_file:
        #     pickle.dump(logbook, log_file)
            
        print("Time", delta_t)
        
        return population, logbook, cache# archive, cache