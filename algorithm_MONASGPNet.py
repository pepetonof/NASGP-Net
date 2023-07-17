# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 20:06:05 2023

@author: josef
"""

from deap import tools
import pickle
import random
import numpy as np
from datetime import datetime, timedelta
import imageio

import matplotlib.pyplot as plt
from tqdm import tqdm
# from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


#%%Population Memory
#Split the population in non replaceable and replaceable memory
def population_memory(pop, MNR_p):
  #Non-Replaceable Solutions Size
  sz=int(len(pop)*MNR_p)
  
  idxs=np.arange(len(pop))
  nr_idx = np.random.choice(idxs, sz, replace=False)
  r_idx = np.array([i for i in idxs if i not in nr_idx])
  
  pop_non_rep = [pop[i] for i in nr_idx]
  pop_rep = [pop[i] for i in r_idx]

  M={"nrep":pop_non_rep, "rep":pop_rep}
  return M

#%%Initial population of the uGA cycle
def u_population(M, size):#, pb_nrep):
  # rng = np.random.default_rng()
  # p = np.array(len(M["nrep"])*[pb_nrep/len(M["nrep"])] + len(M["rep"])*[(1-pb_nrep)/len(M["rep"])])
  full_mem=M["nrep"]+M["rep"]
  upop=random.sample(full_mem, size)
  return upop

#%%Dominance: ind1 dominates ind2?
def dominate(ind1, ind2):
    y1=np.array(ind1.fitness.values)
    y2=np.array(ind2.fitness.values)
    if(np.any(y1 < y2) and np.all(y1 <= y2)):
        return True
    else:
        return False

#%%Pareto-Dominace Relation
def dominate_relation(idx1,idx2):
    if dominate(idx1,idx2):
        return 0
    elif dominate(idx2,idx1):
        return 1
    else:
        return 2
    
#%%Pareto Front
def ParetoFront(P):
    #Paso 1
    i=0
    j=0
    Pn=[]
    flag=True #Stop Flag
    N=len(P)
    while i<N and flag==True:
        while j<N and flag==True:
            #Paso 2
            if i==j:
                j=j+1
            if i!=j and i<N and j<N:
                if dominate(P[j],P[i]):
                    #print('Dominate',i,j,P[i],P[j],Pn)
                    #Paso 4
                    i=i+1
                    j=0
                    if i==N:
                        flag=False
                else:
                    #print('No-Dominate',i,j,P[i],P[j],Pn)
                    #Paso 3
                    j=j+1   
                    if j==N or i==N-1 and i==N-1:
                        Pn.append(P[i])
                        #print('Appended',i,P[i])
                        j=0
                        #Paso 4
                        i=i+1
                        #print('i aumented',i)
                        if i==N:
                            flag=False
            else:
                flag=False
    return Pn

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
               no_evs, delta_t, cache, #archive,
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
        _ind.params = cache[key].params
        # print('Syntax tree:\t', str(_ind), round(_ind.fitness.values[0],3), round(_ind.dice,3), _ind.params, "\t in cache")
        print('Syntax tree:\t', str(_ind), round(_ind.fitness.values[0], 3), _ind.fitness.values[1], "\t in cache")
        
    else:
        #Assign attributes from the original objective function
        if surrogate == None:
            # print(str(_ind))
            # model = toolbox.make_model(_ind, 1)
            # params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print(params)
            fit = toolbox.evaluate(_ind)
            # print("From original", fit, type(fit))
            _ind.fitness.values = fit#[0],
            _ind.dice = 1-fit[0]
            _ind.params = fit[1]
            # print('Syntax tree:\t', str(_ind), round(fit[0],3), round(fit[1],3), fit[2], "\t in original")
            print('Syntax tree:\t', str(_ind), round(fit[0],3), fit[1], "\t in original")
            
        #Assign attributes from the surrogate model
        else:
            fit = toolbox.evaluate_surrogate(_ind, surrogate)
            # print("From surrogate", fit, type(fit))
            _ind.fitness.values = fit#[0],
            _ind.dice = 1-fit[0]
            _ind.params = fit[1]
            
            # print('Syntax tree:\t', str(_ind), round(fit[0],3), round(fit[1],3), fit[2], "\t in surrogate") 
            print('Syntax tree:\t', str(_ind), round(fit[0],3), fit[1], "\t in surrogate")    
    return _ind

#%%MONASGPNetCycle
def MONASGPNetCycle(pop_memory, pop_size, toolbox, cxpb, mutpb, ngen, nelit,
                    ruta, checkpoint_name, cache,
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
        population = u_population(pop_memory, pop_size) #toolbox.population(n=pop_size)
        start_gen  = 0
        halloffame = tools.HallOfFame(maxsize=nelit)
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals', 'time', #'best', 'best_dice', 'best_params',
                          (stats.fields if stats else [])]
        offspring = []
        elitism_inds = []
        
        ###Count the number of evaluations and evaluated individuals
        idx=0
        no_evs=0
        
        ###Individuals to evaluate
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        
    #%%%%Start option 1
    #Si no ha terminado con invalid ind y sigue en la generacion 0
    if idx<len(invalid_ind) and start_gen == 0:
        print('\nGen:\t', start_gen)
        #Evaluate using cache or original;
        while idx < len(invalid_ind):
            ind = invalid_ind[idx]
            key = toolbox.identifier(ind)
            
            ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
            
            #Add to cache when original objective function is used
            if key not in cache:
                #Add to cache
                cache[key]=ind
                
                ####Increment the number of evaluation when original objective function is used
                ####and key is not in cache
                no_evs+=1
            
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
            
            print(f"{idx}/{len(invalid_ind)}", start_gen, ruta.split("/").pop())
         
        best_ind = tools.selBest(population, 1)[0]
        # print('Best:\t', str(best_ind), best_ind.fitness.values, best_ind.dice, best_ind.params)
        print('Best:\t', str(best_ind), round(best_ind.fitness.values[0],3), best_ind.fitness.values[1], start_gen)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=start_gen, nevals=no_evs, time=delta_t,
                       best = str(best_ind), best_dice=best_ind.dice, best_params=best_ind.params,
                       # r_2=None, mse=None, 
                       # rmse=None, mae=None,
                       **record)
        # print(logbook.stream)
    
    #%%%Start option 2
    #Si no ha terminado con 'invalid_ind' en la sesión anterior pero ya ha
    #finalizado con generación 0
    if idx<len(invalid_ind) and start_gen > 0:
        #Evaluate using cache or surrogate model;
        while idx < len(invalid_ind):
            ind = invalid_ind[idx]
            key = toolbox.identifier(ind)
            
            #Precird the fitness valies of new inds via original objective function
            ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
            
            #Add to cache when original objective function is used
            if key not in cache:
                cache[key]=ind
                
                ####Increment the number of evaluation when original objective function is used
                no_evs+=1
            
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
            
            print(f"{idx}/{len(invalid_ind)}", start_gen, ruta.split("/").pop())
        
        best_ind = tools.selBest(population, 1)[0]
        print('Best:\t', str(best_ind), round(best_ind.fitness.values[0],3), best_ind.fitness.values[1], start_gen)
        
        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=start_gen, nevals=no_evs, time=delta_t,
                       best = str(best_ind), best_dice=best_ind.dice, best_params=best_ind.params,
                       **record)
        # print(logbook.stream)
    
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
            # offspring = MateMutationParentFitness(offspring, toolbox, len(offspring), cxpb, mutpb)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            idx = 0
            
            #Evaluate using cache or original objective function;
            while idx < len(invalid_ind):
                ind = invalid_ind[idx]
                key = toolbox.identifier(ind)
                    
                # ind = assign_attributes(ind, key, cache, toolbox, surrogate=rf)
                ind = assign_attributes(ind, key, cache, toolbox, surrogate=None)
                
                #Add to cache when original objective function is used
                if key not in cache:
                    cache[key]=ind
                
                ####Increment the number of evaluated individuals from invalid ind
                idx+=1
                
                ####Increment the number of evaluation when original objective function is used
                no_evs+=1
                
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
                
                print(f"{idx}/{len(invalid_ind)}", gen, ruta.split("/").pop())
                
            #Back the elitism individuals to population
            offspring.extend(elitism_inds)
            
            # Update the hall of fame with the generated individuals
            if halloffame is not None:
                halloffame.update(offspring)
            
            #Replace the curren population by the offspring
            population[:] = offspring
            
            #For print
            if verbose_evo:
                print(logbook.stream)
                
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
            
            # print(str(best_ind), best_ind.fitness.values, best_ind.dice, best_ind.params)
            print('Best:\t', str(best_ind), round(best_ind.fitness.values[0],3), best_ind.fitness.values[1], gen)
            
            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=no_evs, time=delta_t,
                           best = str(best_ind), best_dice=best_ind.dice, best_params=best_ind.params,
                           **record)
            # print(logbook.stream)
        print("Time", delta_t, '\n')
        return population, logbook, cache# archive, cache
    

#%%Adaptive grid
def adaptive_grid(sol,E,num_div,
                  it, ruta, t_frame):

    #Maxs and mins for each grid
    fits=np.array([ind.fitness.values for ind in E]+[sol.fitness.values])
    _maxs=np.max(fits, axis=0)
    _mins=np.min(fits, axis=0)
    
    #number of functions
    num_f = len(E[0].fitness.values)
    
    #grids
    grids = [np.linspace(_mins[i], _maxs[i], num_div+1) for i in range(num_f)]
    
    #Assign coordinates to each solution on E and the sol to introduce
    #Moreover, add a count for each solution
    crowed_grid=np.zeros((num_div,num_div))
    for ind in E:
       # if ind.location==None:
       #     loc = coord(ind, grids)
       #     ind.location = loc
       # else:
       #     loc = ind.location
       loc = coord(ind, grids)
       ind.location = loc
       
       crowed_grid[loc[0], loc[1]]+=1
       
    loc_sol = coord(ind, grids)
    
    #Coordinates of the locations with most crowed population
    result = np.where(crowed_grid == np.amax(crowed_grid))
    listOfCordinates = list(zip(result[0], result[1]))
    
    # print("Try to allocate {sol} with {fit} in E with len {leng}".format(sol=sol, fit=sol.fitness.values, leng=len(E)))
    # print(crowed_grid, listOfCordinates)
    # print("Sol location", loc_sol)
    
    if loc_sol not in listOfCordinates:
      # t_frame=plot_adaptive(num_div, crowed_grid, E, grids,
      #               it, ruta, t_frame)  
      sol.location = loc_sol
      # print("Added:\t ", sol, " to ", E, len(E))
      
      #if sol is in less crowded region of the E
      E.append(sol)
      
      #and delete a sol form the E from the most crowed region
      idx_to_delete=[i for i in range(len(E)) if E[i].location in listOfCordinates]
      
      # print(idx_to_delete)
      idx_selected=random.choice(idx_to_delete)
      
      crowed_grid[E[idx_selected].location[0], E[idx_selected].location[1]]-=1
      # print("Sol to eliminate", E[idx_selected].location[0], E[idx_selected].location[1])
      # print("Idx of element to delete {idx} from E with len {leng}".format(idx=idx_selected, leng=len(E)))
      E.pop(idx_selected)
      crowed_grid[sol.location[0], sol.location[1]]+=1
      # print(crowed_grid)
      
      #To plot meshgrid and solutions in M
      t_frame=plot_adaptive(num_div, crowed_grid, E, grids,
                    it, ruta, t_frame)
      
    return t_frame

#%%%Assign coordinates to an individual
#For homework we have always 2 objective functions
def coord(individual, grids):
    coord = []
    for g in range(len(grids)):
        for j in range(1, len(grids[g])):
            if individual.fitness.values[g]>=grids[g][j-1] and individual.fitness.values[g]<=grids[g][j]:
                coord.append(j)
                
    #Transform from cartesian to ij matrix indexing    
    i=len(grids[1])-coord[1]-1
    j=coord[0]-1
    return i, j

#%%%Plot the External Memory
def plot_adaptive(num_div, crowed_grid, E, grids,
                  it, ruta, t_frame):

    #To plot meshgrid and solutions in M
    xv, yv = np.meshgrid(grids[0], grids[1])
    fig, ax = plt.subplots()
    ax.plot(xv, yv, marker='+', color='k', linestyle='none')
    
    nds_test = np.array([x.fitness.values for x in E])
    ax.scatter(nds_test[:,0],nds_test[:,1],s=20,c="red")
    
    #Moreover, the number of solutions on each hypercube is shown
    dots_name=crowed_grid.flatten()
    
    #Position of each label?
    eps = 1/(num_div*3)
    
    lins=np.linspace(eps,1-eps,num_div)
    
    nx,ny=np.meshgrid(lins[::-1], lins, indexing='ij')
    nx,ny=nx.flatten("C"),ny.flatten("C")
    dots_loc=np.array(list(zip(nx,ny)))
    
    for i, txt in enumerate(dots_name):
        if dots_name[i]!=0:
            plt.annotate(str(int(txt)), (dots_loc[i,1], dots_loc[i,0]), 
                        xycoords='axes fraction')
            
    #Set title
    ax.set_title('External Memory:'+ str(it))
    
    plt.savefig(ruta+"/frame_"+str(t_frame)+".png", 
                transparent = False,  
                facecolor = 'white'
               )
    plt.close()
    
    t_frame+=1
    return t_frame


def eaMONASGPNet(pop, upop_size, percmem_nr, 
                 extmem_size, its, its_u,
                 div_agrid, rep_cycle,
                 cxpb, mutpb, 
                 toolbox, stats, ruta):
    
    M = population_memory(pop, percmem_nr)
    E = []
    t_frame = 0
    
    cache = {}
    no_evs = 0
    time = timedelta(seconds=0)
    
    g_log = tools.Logbook()
    g_log.header = ['it', 'nevals', 'time',
                      (stats.fields if stats else [])]
    
    for it in range(its):
        #uGA Cycle
        epop, log, cache = MONASGPNetCycle(pop_memory=M, pop_size=upop_size, toolbox=toolbox,
                                    cxpb = cxpb, mutpb = mutpb, ngen = its_u, nelit=1,
                                    ruta = ruta, checkpoint_name=False, cache=cache,
                                    stats=None, halloffame=None, verbose_evo=False)
        
        time += log.select("time")[-1]#delta_t
        no_evs += log.select("nevals")[-1]
        
        #Update logbook
        g_log.record(it=it, nevals=no_evs, time=time)
        
        """Filtering that use 3 type of elitism"""
        # We choose two nondominated vectors from the final population
        # assuming that we have two or more nondominated vectors and compare them 
        # with the contents of the external memory (this memory is initially empty).
        # If there is only one, then this vector is the only one selected
        nds = ParetoFront(epop)
        if len(nds)>1:
            selected=random.sample(nds,2)
        else:
            selected=nds
        
        """Compare selected with E"""
        # First type of elitism
        # If either of them (or both) remains as nondominated after
        # comparing against the vectors in this external memory, then they are included
        # there. All the dominated vectors are eliminated from the external memory
        if len(E)==0:
            E.extend(selected)
        else:
            for sol in selected:
                flag=True
                idx=0
                while idx<len(E):
                    sole=E[idx]
                    r=dominate_relation(sol, sole)
                    if r==1: #Si la nueva solucion es no dominada
                        #Descartar sol
                        flag=False
                        break
                    elif r==0:#Si la nueva solución domina a sole
                        #Eliminar sole de E
                        E.pop(idx)
                    else:
                        idx+=1
                        
                if flag==True:
                    if len(E)<extmem_size:
                        E.append(sol)
                    else:
                        #If E es full when trying o insert sol
                        #then adaptive grid
                        t_frame=adaptive_grid(sol, E, div_agrid, 
                                      it, ruta, t_frame)
        
        """Compare selected with M[rep]"""
        #Secondt type of elitism
        #Compare selected solutions from Pi (selected)
        #against 2 solutions from population memory replaceable
        selected_pmr_idx=random.sample(list(range(len(M["rep"]))),2)
        for i,j in zip(range(len(selected)), selected_pmr_idx):
            
          #Assign a fitness to individuals with invalid ind in the population memory
          #in order to compare with dominate_relation
          if not M["rep"][j].fitness.valid:
              M["rep"][j].fitness.values = toolbox.evaluate(M["rep"][j])
              
          if dominate_relation(selected[i], M["rep"][j])==0:
            #print(selected_nds[i], " dominates ", M["rep"][j])
            M["rep"][j]=selected[i]
        
        """Replace M["rep"] with elements from E every rep_cycle"""
        #Third type of elitism
        if it%rep_cycle==0:
          #If E has more elements than M then select len(M) elements from E
          if len(E)>len(M["rep"]):
            rep=random.sample(E, len(M["rep"]))
            M["rep"]=rep
            
          #If E has less elements than M then, select n elements that exceeds len(M)
          elif len(E)<len(M["rep"]):
            # dif_el=len(M["rep"])-len(E)
            # idx_to_del=random.sample(list(range(len(M["rep"]))), dif_el)
            idx_selected = random.sample(list(range(len(M["rep"]))), len(E))
            # for idx,del_idx in enumerate(idx_to_del):
            for idx, idxs in enumerate(idx_selected):
              M["rep"][idxs]=E[idx]
        
          else:
            M["rep"]=E
    
    if t_frame>0:
        frames = []
        for t in range(t_frame):
            image = imageio.v2.imread(ruta+"/frame_"+str(t)+".png")
            frames.append(image)
        
        imageio.mimsave(ruta+'/example.gif', # output gif
                        frames,         # array of input frames
                        fps = 100,
                        loop=1,
                        )         # optional: frames per second
    
    return M, E, g_log

#%%MONASGPNet with Checkpoint
def eaMONASGPNet_CHKP(pop, upop_size, percmem_nr, 
                 extmem_size, its, its_u,
                 div_agrid, rep_cycle,
                 cxpb, mutpb, 
                 toolbox, stats, ruta,
                 checkpoint):
    #If checkpoint, recover data
    if checkpoint:
        print('recovering micro...', ruta)
        with open(ruta+'/'+checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        
        it_start = cp["it"]
        E        = cp["E"]
        M        = cp["M"]
        cache    = cp["cache"]
        g_log    = cp["g_log"]
        
        no_evs   = cp["no_evs"]
        time     = cp["time"]
        t_frame  = cp["t_frame"]
        random.setstate(cp["rndstate"])
        
        checkpoint_name = checkpoint[:-4]+"_evo.pkl"
        
        print("It_start:\t", it_start)
    
    #else, start new iteration
    else:
        it_start = 0
        E = []
        M = population_memory(pop, percmem_nr)
        cache = {}
        g_log = tools.Logbook()
        g_log.header = ['it', 'nevals', 'time',
                          (stats.fields if stats else [])]
        
        no_evs = 0
        time = timedelta(seconds=0)
        t_frame = 0
        
        checkpoint_name = False
        
    for it in range(it_start, its):
        #uGA Cycle
        print("IT:\t", it)
        epop, log, cache = MONASGPNetCycle(pop_memory=M, pop_size=upop_size, toolbox=toolbox,
                                    cxpb = cxpb, mutpb = mutpb, ngen = its_u, nelit=1,
                                    ruta = ruta, checkpoint_name=checkpoint_name, cache=cache,
                                    stats=None, halloffame=None, verbose_evo=False)
        
        time += log.select("time")[-1]#delta_t
        no_evs += log.select("nevals")[-1]
        
        #Update logbook
        g_log.record(it=it, nevals=no_evs, time=time)
        
        """Filtering that use 3 type of elitism"""
        # We choose two nondominated vectors from the final population
        # assuming that we have two or more nondominated vectors and compare them 
        # with the contents of the external memory (this memory is initially empty).
        # If there is only one, then this vector is the only one selected
        nds = ParetoFront(epop)
        if len(nds)>1:
            selected=random.sample(nds,2)
        else:
            selected=nds
        
        """Compare selected with E"""
        # First type of elitism
        # If either of them (or both) remains as nondominated after
        # comparing against the vectors in this external memory, then they are included
        # there. All the dominated vectors are eliminated from the external memory
        if len(E)==0:
            E.extend(selected)
        else:
            for sol in selected:
                flag=True
                idx=0
                while idx<len(E):
                    sole=E[idx]
                    r=dominate_relation(sol, sole)
                    if r==1: #Si la nueva solucion es no dominada
                        #Descartar sol
                        flag=False
                        break
                    elif r==0:#Si la nueva solución domina a sole
                        #Eliminar sole de E
                        E.pop(idx)
                    else:
                        idx+=1
                        
                if flag==True:
                    if len(E)<extmem_size:
                        E.append(sol)
                    else:
                        #If E es full when trying o insert sol
                        #then adaptive grid
                        t_frame=adaptive_grid(sol, E, div_agrid, 
                                      it, ruta, t_frame)
        
        """Compare selected with M[rep]"""
        #Secondt type of elitism
        #Compare selected solutions from Pi (selected)
        #against 2 solutions from population memory replaceable
        selected_pmr_idx=random.sample(list(range(len(M["rep"]))),2)
        for i,j in zip(range(len(selected)), selected_pmr_idx):
            
          #Assign a fitness to individuals with invalid ind in the population memory
          #in order to compare with dominate_relation
          if not M["rep"][j].fitness.valid:
              M["rep"][j].fitness.values = toolbox.evaluate(M["rep"][j])
              key=toolbox.identifier(M["rep"][j])
              if key not in cache:
                  cache[key]=M["rep"][j]
              
          if dominate_relation(selected[i], M["rep"][j])==0:
            #print(selected_nds[i], " dominates ", M["rep"][j])
            M["rep"][j]=selected[i]
        
        """Replace M["rep"] with elements from E every rep_cycle"""
        #Third type of elitism
        if it%rep_cycle==0:
          #If E has more elements than M then select len(M) elements from E
          if len(E)>len(M["rep"]):
            rep=random.sample(E, len(M["rep"]))
            M["rep"]=rep
            
          #If E has less elements than M then, select n elements that exceeds len(M)
          elif len(E)<len(M["rep"]):
            # dif_el=len(M["rep"])-len(E)
            # idx_to_del=random.sample(list(range(len(M["rep"]))), dif_el)
            idx_selected = random.sample(list(range(len(M["rep"]))), len(E))
            # for idx,del_idx in enumerate(idx_to_del):
            for idx, idxs in enumerate(idx_selected):
              M["rep"][idxs]=E[idx]
        
          else:
            M["rep"]=E
        
        #Reinitilized checkpoint for future calls
        checkpoint_name = False
        
        """Save progress of uGA"""
        cp = dict(it=it, E=E, M=M, cache=cache, g_log=g_log,
                  no_evs=no_evs, time=time, t_frame=t_frame,
                  rndstate=random.getstate())
        with open(ruta + '/'+ "checkpoint.pkl", "wb") as cp_file:
            pickle.dump(cp, cp_file)
    
    if t_frame>0:
        frames = []
        for t in range(t_frame):
            image = imageio.v2.imread(ruta+"/frame_"+str(t)+".png")
            frames.append(image)
        
        imageio.mimsave(ruta+'/example.gif', # output gif
                        frames,         # array of input frames
                        fps = 100,
                        loop=1,
                        )         # optional: frames per second
    
    return M, E, g_log