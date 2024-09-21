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

# from surrogate import get_features

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
def checkpoint(generation, population, 
               # offspring,
               invalid_ind, 
               idx, elitism_inds,
               no_evs, delta_t, cache,
               halloffame, logbook,
               rndstate, ruta):
    
    # Fill the dictionary using the dict(key=value[, ...]) constructor
    cp = dict(generation=generation, population=population, 
               # offspring=offspring,
               invalid_ind=invalid_ind, idx=idx, elitism_inds=elitism_inds,
               no_evs=no_evs, delta_t=delta_t, cache=cache, #archive=archive,
               halloffame=halloffame, logbook=logbook,
               rndstate=rndstate)
    with open(ruta + '/'+ "checkpoint_evo.pkl", "wb") as cp_file:
        pickle.dump(cp, cp_file)
    del cp_file
    
    return

# #%%Set attributes of individuals using cache and update it
# def assign_attributes(_ind, key, cache, toolbox, surrogate=None):
#     # _ind=toolbox.clone(ind)
#     if key in cache:
#         #Assign attributes from cache
#         _ind.fitness.values = cache[key].fitness.values
#         _ind.dice = cache[key].dice
#         _ind.iou = cache[key].iou
#         _ind.hd95 = cache[key].hd
#         _ind.hd = cache[key].hd95
#         # _ind.nsd = cache[key].nsd
#         _ind.params = cache[key].params
                
#         print('Syntax tree:\t', str(_ind), round(_ind.fitness.values[0],3), round(_ind.dice,3), _ind.params, "\t in cache")
        
#     else:
#         # #Assign attributes from the original objective function
#         # if surrogate == None:
#         fit = toolbox.evaluate(_ind)
#         _ind.fitness.values = fit[0],
        
#         _ind.dice_t= fit[1]
#         _ind.dice = fit [2]
        
#         _ind.iou = fit[3]
#         _ind.hd = fit[4]
#         _ind.hd95 = fit[5]
        
#         _ind.params = fit[6]
        
#         print('Syntax tree hola:\t', str(_ind), round(_ind.fitness.values[0], 3), round(_ind.dice,3), _ind.params, "\t in original")
            
#         # #Assign attributes from the surrogate model
#         # else:
#         #     fit = toolbox.evaluate_surrogate(_ind, surrogate)
#         #     _ind.fitness.values = fit[0],
#         #     _ind.dice = fit[1]
#         #     _ind.iou = fit[2]
#         #     _ind.hd = fit[3]
#         #     _ind.hd95 = fit[4]
#         #     # _ind.nds = fit[5]
#         #     _ind.params = fit[5]
            
#         #     print('Syntax tree:\t', str(_ind), round(_ind.fitness.values[0],3), round(_ind.dice,3), _ind.params, "\t in surrogate")    
#     return _ind

def baseline_eaNASGPNet(pop_size, toolbox, 
             # pset,
             cxpb, mutpb, ngen, nelit,
             ruta, checkpoint_name, 
             stats = None, halloffame=None, verbose_evo=__debug__):
    
    ####Take time. Keep delta_t, no matters loose t
    init_time=datetime.now()
    delta_t=timedelta(seconds=0)
    metrics = toolbox.evaluate.keywords["metrics"]
    
    ####If checkpoint then load data from the file "/checkpoint"
    if checkpoint_name:
        print('recovering...', ruta)
        with open(ruta+'/'+checkpoint_name, "rb") as cp_file:
            cp = pickle.load(cp_file)
        # print(cp.keys())
        start_gen    = cp["generation"]
        population   = cp["population"]
        invalid_ind  = cp["invalid_ind"]
        idx          = cp["idx"]
        elitism_inds = cp["elitism_inds"]
        no_evs       = cp["no_evs"]
        delta_t      = cp["delta_t"]
        cache        = cp["cache"]
        halloffame   = cp["halloffame"]
        logbook      = cp["logbook"]
        random.setstate(cp["rndstate"])
        
        print('Time:\t', delta_t)
        print('Start Get: \t', start_gen)
        print('Idx Pop: \t', idx)
        print('Best', str(tools.selBest(population, 1)[0]), 
              round(tools.selBest(population, 1)[0].fitness.values[0],3), 
              tools.selBest(population, 1)[0].params)
        
    else:
        ####Else start a new evolution
        population = toolbox.population(n=pop_size)
        start_gen  = 0
        halloffame = tools.HallOfFame(maxsize=nelit)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'time', 'best', 
                          'best_dice',
                          'best_params'] + (stats.fields if stats else [])
        offspring = []
        elitism_inds = []
        
        ###Count the number of evaluations and evaluated individuals
        idx=0
        no_evs=0
        cache={}
        
        # ###Individuals to evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    total_inds = len(invalid_ind)
    while len(invalid_ind)>0:#si no ha terminado, continua con evs
        
        ind = invalid_ind[0]#se toma el individuo de la primer posicion
        key = toolbox.identifier(ind)
        if key in cache:
            ind.fitness.values = cache[key].fitness.values
            ind.params = cache[key].params
            for metric in metrics:
                setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
            print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
            
        else:
            fit, params, *out_metrics = toolbox.evaluate(ind)
            ind.fitness.values = fit,
            ind.params = params
            for metric, value in zip(metrics, out_metrics): #Segmentation metrics
                setattr(ind, metric._get_name(), value)
            print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")
            
            cache[key]=ind
            no_evs += 1
            
        ####Take time every evaluation
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t
        
        #update len of invalid ind
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        idx += 1
        
        #checkpoint
        checkpoint(generation=start_gen, 
                   population=population, 
                   invalid_ind=invalid_ind, 
                   idx=idx, elitism_inds=elitism_inds,
                   no_evs=no_evs, delta_t=delta_t, cache=cache,
                   halloffame=halloffame, logbook=logbook,
                   rndstate=random.getstate(), ruta=ruta)
        
        print(f"{idx}/{total_inds}", start_gen, ruta.split("/").pop(), delta_t)
        
    #best individual to store in logbook
    best_ind = tools.selBest(population, 1)[0]
    print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3),best_ind.params, start_gen)
    
    #append the current generation statistics to the logbook
    record = stats.compile(population) if stats else {}
    dict_log = {"gen":start_gen,
                "nevals":no_evs,
                "time":delta_t,
                "best":str(best_ind)}
    
    ##metrics of the best individual
    for m in metrics:
        key = str(type(m)).strip('>').strip("'").split('.')[-1]
        dict_log["best_"+key]=getattr(best_ind, key)
        
    logbook.record(**dict_log, **record)
    
    if verbose_evo:
        print(logbook.stream)
    
    #time
    #checkpoint
    #best_ind
    #log_record
    
    start_gen+=1
    # print('population',len(population), [str(ind) for ind in population])
    # print('invalid_ind', len(invalid_ind), [str(ind) for ind in invalid_ind])
    for gen in range(start_gen, ngen+1):
        print('\nGen:\t', gen)
        
        #elitism
        elitism_inds = toolbox.selectElitism(toolbox.clone(population), k=nelit)
        # print('elitism_inds', len(elitism_inds), [str(ind) for ind in elitism_inds])#1
        
        #tournament
        offspring = toolbox.select(population, len(population) - nelit)
        # print('parents', len(offspring), [str(ind) for ind in offspring])#4
        
        #crossover and mutation
        offspring = MateMutation(offspring, toolbox, len(offspring), cxpb, mutpb) #4
        
        # print('Offsprring', len(offspring), [str(ind) for ind in offspring])
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        idx = 0
        total_inds = len(invalid_ind)
        
        while len(invalid_ind)>0:
            ind = invalid_ind[0]
            key = toolbox.identifier(ind)
            if key in cache:
                ind.fitness.values = cache[key].fitness.values
                ind.params = cache[key].params
                for metric in metrics:
                    setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
            else:
                fit, params, *out_metrics = toolbox.evaluate(ind)
                ind.fitness.values = fit,
                ind.params = params
                for metric, value in zip(metrics,out_metrics):
                    setattr(ind, metric._get_name(), value)
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
                
                cache[key] = ind
                no_evs +=1
            
            #time
            t = datetime.now()
            delta_t += (t - init_time)
            init_time = t #Keep delta_t, no matters loose t
            
            #update the population every evaluation in order to keep it in the checkpoint
            population = elitism_inds+offspring
            
            #update len of invalid ind
            invalid_ind = [ind for ind in population if not ind.fitness.valid]
            idx += 1
            
            #checkpoint
            checkpoint(generation=start_gen, 
                       population=population, 
                       invalid_ind=invalid_ind, 
                       idx=idx, elitism_inds=elitism_inds,
                       no_evs=no_evs, delta_t=delta_t, cache=cache,
                       halloffame=halloffame, logbook=logbook,
                       rndstate=random.getstate(), ruta=ruta)

            print(f"{idx}/{total_inds}", gen, ruta.split("/").pop(), delta_t)
            
        if halloffame is not None:
            halloffame.update(offspring)
            
        #time
        t = datetime.now()
        delta_t += (t - init_time)
        init_time = t #Keep delta_t, no matters loose t
        
        #checkpoint
        checkpoint(generation=start_gen, 
                   population=population, 
                   invalid_ind=invalid_ind, 
                   idx=idx, elitism_inds=elitism_inds,
                   no_evs=no_evs, delta_t=delta_t, cache=cache,
                   halloffame=halloffame, logbook=logbook,
                   rndstate=random.getstate(), ruta=ruta)
        #best ind
        best_ind = tools.selBest(population, 1)[0]
        print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3), best_ind.params, gen)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        dict_log = {"gen":start_gen,
                    "nevals":no_evs,
                    "time":delta_t,
                    "best":str(best_ind)}
        for m in metrics:
            key = key = str(type(m)).strip('>').strip("'").split('.')[-1]
            dict_log["best_"+key]=getattr(best_ind, key)
        
        logbook.record(**dict_log,
                        **record)
        
        #For print
        if verbose_evo:
            print(logbook.stream)
    
    print("Time", delta_t)
        
    return population, logbook, cache


def baseline_surrogate(pop_size, toolbox,
                       pset,
                       cxpb, mutpb, ngen, nelit,
                       ruta, checkpoint_name,
                       stats = None, halloffame=None, verbose_evo = __debug__):
    ####Take time. Keep delta_t, no matters loose t
    init_time=datetime.now()
    delta_t=timedelta(seconds=0)
    metrics = toolbox.evaluate.keywords["metrics"]
    
    ####If checkpoint then load data from the file "/checkpoint"
    if checkpoint_name:
        print('recovering...', ruta)
        with open(ruta+'/'+checkpoint_name, "rb") as cp_file:
            cp = pickle.load(cp_file)
        # print(cp.keys())
        start_gen    = cp["generation"]
        population   = cp["population"]
        invalid_ind  = cp["invalid_ind"]
        idx          = cp["idx"]
        elitism_inds = cp["elitism_inds"]
        no_evs       = cp["no_evs"]
        delta_t      = cp["delta_t"]
        cache        = cp["cache"]
        halloffame   = cp["halloffame"]
        logbook      = cp["logbook"]
        random.setstate(cp["rndstate"])
        
        archive = cp["archive"]###!!!
        
        print('Time:\t', delta_t)
        print('Start Get: \t', start_gen)
        print('Idx Pop: \t', idx)
        print('Best', str(tools.selBest(population, 1)[0]), 
              round(tools.selBest(population, 1)[0].fitness.values[0],3), 
              tools.selBest(population, 1)[0].params)
        
    else:
        ####Else start a new evolution
        population = toolbox.population(n=pop_size)
        start_gen  = 0
        halloffame = tools.HallOfFame(maxsize=nelit)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'time', 'best', 
                          'best_dice',
                          'best_params'] + (stats.fields if stats else [])
        offspring = []
        elitism_inds = []
        
        ###Count the number of evaluations and evaluated individuals
        idx=0
        no_evs=0
        cache={}
        
        # ###Individuals to evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    
    
    return
      

#%%NASGP-Net Algorithm
def eaNASGPNet(pop_size, toolbox, cxpb, mutpb, ngen, nelit,
                 ruta, checkpoint_name,
                 stats=None, halloffame=None, verbose_evo=__debug__,
                 ):
    
    ####Take time. Keep delta_t, no matters loose t
    init_time=datetime.now()
    delta_t=timedelta(seconds=0)
    metrics = toolbox.evaluate.keywords["metrics"]
    
    ####If checkpoint then load data from the file "/checkpoint"
    if checkpoint_name:
        print('recovering...', ruta)
        with open(ruta+'/'+checkpoint_name, "rb") as cp_file:
            cp = pickle.load(cp_file)
        # print(cp.keys())
        start_gen    = cp["generation"]
        population   = cp["population"]
        invalid_ind  = cp["invalid_ind"]
        idx          = cp["idx"]
        elitism_inds = cp["elitism_inds"]
        no_evs       = cp["no_evs"]
        delta_t      = cp["delta_t"]
        cache        = cp["cache"]
        halloffame   = cp["halloffame"]
        logbook      = cp["logbook"]
        random.setstate(cp["rndstate"])
        
        print('Time:\t', delta_t)
        print('Start Get: \t', start_gen)
        print('Idx Pop: \t', idx)
        print('Best', str(tools.selBest(population, 1)[0]), 
              round(tools.selBest(population, 1)[0].fitness.values[0],3), 
              tools.selBest(population, 1)[0].params)
        
    else:
        ####Else start a new evolution
        # random.seed(42)#!!!
        population = toolbox.population(n=pop_size)
        start_gen  = 0
        halloffame = tools.HallOfFame(maxsize=nelit)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'time', 'best', 
                          'best_dice',
                          'best_iou',
                          'best_hd',
                          'best_hd95',
                          'best_nsd',
                          'best_params'] + (stats.fields if stats else [])
        offspring = []
        elitism_inds = []
        
        ###Count the number of evaluations and evaluated individuals
        idx=0
        no_evs=0
        cache={}
        
        # ###Individuals to evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        
    #%%%%Start option 1
    #Si no ha terminado con invalid ind y sigue en la generacion 0
    ###Individuals to evaluate
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    
    if idx<len(invalid_ind): #and start_gen == 0:
        while idx < len(invalid_ind):
            ind = invalid_ind[idx]
            key = toolbox.identifier(ind)
            
            #Predict using original fitness function
            # ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
            
            if key in cache:
                #Assign attributes from cache
                ind.fitness.values = cache[key].fitness.values
                ind.params = cache[key].params

                for metric in metrics: #Segmentation metrics
                    setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
     
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
                
            else:
                
                if start_gen==0 and idx==0:
                    #Esto de guardarlos datos, para qué era? llevar un seguimiento 
                    #de los datos de entrenamiento y prueba por fold
                    fit, params, *out_metrics = toolbox.evaluate(ind, save_data=True)
                else:
                    fit, params, *out_metrics  = toolbox.evaluate(ind)
                
                ind.fitness.values = fit,
                ind.params = params
                for metric, value in zip(metrics, out_metrics): #Segmentation metrics
                    setattr(ind, metric._get_name(), value)
                
                #Add to cache
                cache[key]=ind
                
                ####Increment the number of evaluations when original objective function is used
                ####and key is not in cache
                no_evs+=1
                
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")

            ####Increment the number of evaluated individuals from invalid ind
            idx+=1
            
            ####Take time every evaluation
            t = datetime.now()
            delta_t += (t - init_time)
            init_time = t #Keep delta_t, no matters loose t
            
            ####Checkpoint every evaluation and every generation
            checkpoint(generation=start_gen, 
                       population=population, 
                       invalid_ind=invalid_ind, 
                       idx=idx, elitism_inds=elitism_inds,
                       no_evs=no_evs, delta_t=delta_t, cache=cache,
                       halloffame=halloffame, logbook=logbook,
                       rndstate=random.getstate(), ruta=ruta)
            
            print(f"{idx}/{len(invalid_ind)}", start_gen, ruta.split("/").pop(), delta_t)
         
        best_ind = tools.selBest(population, 1)[0]
        print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3),best_ind.params, start_gen)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        dict_log = {"gen":start_gen,
                    "nevals":no_evs,
                    "time":delta_t,
                    "best":str(best_ind)}
        for m in metrics:
            key = str(type(m)).strip('>').strip("'").split('.')[-1]
            dict_log["best_"+key]=getattr(best_ind, key)
            
        logbook.record(**dict_log,
                       **record)
        
        #For print
        if verbose_evo:
            print(logbook.stream)
        
    
    #%%%Start option 2
    #Si ya ha terminado con "invalid_ind" en la sesión anterior o si no checkpoint_name, 
    #entonces continúa normalmente (pasa a la siguiente generación)
    # print("Continue Here\n\n")
    if idx==len(invalid_ind) or checkpoint_name==False:
        start_gen+=1
        
        for gen in range(start_gen, ngen+1):
            print('\nGen:\t', gen)
            
            #Elitism
            population_for_eli=[toolbox.clone(ind) for ind in population]
            elitism_inds = toolbox.selectElitism(population_for_eli, k=nelit)
            
            #Tournament selection
            offspring = toolbox.select(population, len(population) - nelit)
            
            #Crossover and Mutation
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
                    ind.params = cache[key].params
                    
                    for metric in metrics: #Segmentation metrics
                        setattr(ind, metric._get_name(), getattr(cache[key], metric._get_name()))
                            
                    print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.DiceMetric,3), ind.params, "\t in cache")
                    
                else:
                    # #Assign attributes from the original objective function                  
                    fit, params, *out_metrics  = toolbox.evaluate(ind)
                
                    ind.fitness.values = fit,
                    ind.params = params
                    for metric, value in zip(toolbox.evaluate.keywords["metrics"], out_metrics): #Segmentation metrics
                        setattr(ind, metric._get_name(), value)
                    
                    #Add to cache
                    cache[key]=ind
                    
                    ####Increment the number of evaluations when original objective function is used
                    ####and key is not in cache
                    no_evs+=1
                    
                    print('Syntax tree:\t', str(ind), round(ind.fitness.values[0], 3), round(ind.DiceMetric,3), ind.params, "\t in original")
                
                ####Increment the number of evaluated individuals from invalid ind
                idx+=1
                
                ####Take time every evaluation
                t = datetime.now()
                delta_t += (t - init_time)
                init_time = t #Keep delta_t, no matters loose t
                
                population = elitism_inds+offspring
                
                ####Checkpoint every evaluation and every generation
                checkpoint(generation=gen, 
                           population=population, 
                           # offspring=offspring,
                           invalid_ind=invalid_ind, 
                           idx=idx, elitism_inds=elitism_inds,
                           no_evs=no_evs, delta_t=delta_t, cache=cache,
                           halloffame=halloffame, logbook=logbook,
                           rndstate=random.getstate(), ruta=ruta)
                
                print(f"{idx}/{len(invalid_ind)}", gen, ruta.split("/").pop(), delta_t)
                
                # #Save test, train, split list
                # if idx == len(invalid_ind) and gen == ngen:
                    
                
            #Back the elitism individuals to population
            # offspring.extend(elitism_inds)
            
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
            
            #Replace the curren population by the offspring
            # population[:] = offspring
            
            ####Take time every evaluation
            t = datetime.now()
            delta_t += (t - init_time)
            init_time = t #Keep delta_t, no matters loose t
            
            ####Checkpoint every evaluation and every generation
            checkpoint(generation=gen, population=population, 
                       # offspring=offspring,
                       invalid_ind=invalid_ind, 
                       idx=idx, elitism_inds=elitism_inds,
                       no_evs=no_evs, delta_t=delta_t, cache=cache,
                       halloffame=halloffame, logbook=logbook,
                       rndstate=random.getstate(), ruta=ruta)
            
            #Take the best individual after finishes each generation
            #and print their attributes
            best_ind = tools.selBest(population, 1)[0]
            # print(str(best_ind), best_ind.fitness.values, best_ind.dice, best_ind.params)
            print('Best:', str(best_ind), round(best_ind.fitness.values[0],3), round(best_ind.DiceMetric,3), best_ind.params, gen)
            
            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            dict_log = {"gen":start_gen,
                        "nevals":no_evs,
                        "time":delta_t,
                        "best":str(best_ind)}
            for m in metrics:
                key = key = str(type(m)).strip('>').strip("'").split('.')[-1]
                dict_log["best_"+key]=getattr(best_ind, key)
            
            logbook.record(**dict_log,
                            **record)
            
            #For print
            if verbose_evo:
                print(logbook.stream)
            
        print("Time", delta_t)
        
        return population, logbook, cache# archive, cache