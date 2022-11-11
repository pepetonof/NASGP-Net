# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 01:33:18 2022

@author: josef
"""
import torch.multiprocessing as mp
import torch

from deap import tools
import pickle
import random
import numpy as np

from datetime import datetime, timedelta

from utils.save_utils import save_progress
from toolbox_functions import evaluationMP

#%%Variation Operators
def MyVar(population, toolbox, lambda_, cxpb, mutpb):
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

#%%NASGP-Net
class NASGP_Net:
    def __init__(self, evolutionary_parameters, training_parameters,
                 # population_size, num_gen, cxpb, mutpb, 
                 # num_elit, tournament_size,
                 # max_params, w,
                 
                 # num_epochs, loss_fn, learning_rate,
                 
                 toolbox, pset, loaders,
                 stats=None, halloffame=None, verbose_evo=__debug__, verbose_train=False,
                 checkpoint=False, ruta=None, foldername=None):
        
        #Variables para iniciar objeto
        #Para evolución
        # self.pz = population_size
        # self.ngen = num_gen
        # self.cxpb = cxpb
        # self.mutpb = mutpb
        # self.num_elit = num_elit
        # self.tz = tournament_size
        # self.max_params = max_params
        # self.w = w
        self.pz = evolutionary_parameters["population_size"]
        self.ngen = evolutionary_parameters["n_gen"]
        self.cxpb = evolutionary_parameters["cxpb"]
        self.mutpb = evolutionary_parameters["mutpb"]
        self.num_elit = evolutionary_parameters["n_elit"]
        self.tz = evolutionary_parameters["tournament_size"]
        self.max_params = evolutionary_parameters["max_params"]
        self.w = evolutionary_parameters["w"]
        
        #Para entrenamiento
        # self.nepochs = num_epochs
        # self.loss_fn = loss_fn
        # self.lr = learning_rate
        self.nepochs = training_parameters["num_epochs"]
        self.loss_fn = training_parameters["loss_f"]
        self.lr = training_parameters["learning_rate"]
        
        ###Estadísticas e impresión
        self.stats=stats
        self.halloffame=halloffame
        self.verbose=verbose_evo
        self.verbose_train=verbose_train
        
        #Toolbox, pset and loaders
        self.toolbox=toolbox
        self.pset = pset
        self.loaders = loaders
        
        #Variables para checkpoint y datos
        self.checkpoint = checkpoint
        self.ruta = ruta
        self.foldername = foldername
        
        #Cache para ejecución más rápida
        self.cache={}
        
        # pset=primitive_set()
        # creator=create(pse
        # string='apool(sub(sconv(mod, 8, 3, 5, 2), 0.34, se(dCon(sconv(mod, 8, 5, 3, 1), 0.6)), 0.39))'
        # pt=gp_tree.PrimitiveTree.from_string(string,pset)
        # ind=creator.Individual(pt)
        # print(ind, '\n')
        # apt_dice=toolbox.evaluationMP(ind, 2, combo_loss, 0.0001,
        #                               31000000, 0.3,
        #                               '1', 'cuda:0', [])
        # print(apt_dice, ind.dice, ind.params)
    
    #%%Recover data
    def recover(self, ruta):
        print('recovering...', ruta)
        with open(ruta+'/'+self.checkpoint, "rb") as cp_file:
            cp = pickle.load(cp_file)
        
        self.population = cp["population"]
        self.best       = cp["best"]
        self.gen        = cp["generation"]
        self.halloffame = cp["halloffame"]
        self.logbook    = cp["logbook"]
        self.offspring  = cp["offspring"]
        self.elitism_inds = cp["elitism_inds"]
                    
        self.invalid_ind= cp["invalid_ind"]
        self.idx        = cp["idx"]
        self.no_evs     = cp["no_evs"]
        self.delta_t    = cp["delta_t"]
        
        self.cache      = cp["cache"]
        random.setstate(cp["rndstate"])
        
        print('StartGen:', self.gen, 'Idx:',self.idx, 'NoEvs:', self.no_evs, '\n',
              'LenPop:',len(self.population),'LenOff:', len(self.offspring),
              'LenInv:',len(self.invalid_ind),'\n',
              'Best:', str(self.best), 'BestFit:', self.best.fitness.values, self.best.dice, self.best.params, '\n')
        
        return 
    
    #%%Assign attributes from cache
    def assign_attributes_cache(self, ind, key, cache):
        ind.fitness.values = cache[key].fitness.values
        ind.dice = cache[key].dice
        ind.params = cache[key].params
        ind.train_loss = cache[key].train_loss
        ind.valid_loss = cache[key].valid_loss
        ind.train_dice = cache[key].train_dice
        ind.valid_dice = cache[key].valid_dice
        ind.dices = cache[key].dices
        ind.ious = cache[key].ious
        ind.hds = cache[key].hds
        
        return
    
    #%% Assign attributes from evaluation   
    def assign_attributes_evaluation(self, ind, fit, dice, params, metrics, graphs):
        ind.fitness.values = fit,
        ind.dice = dice
        ind.params = params
        ind.dices = metrics["dices"]
        ind.ious = metrics["ious"]
        ind.hds = metrics["hds"]
        ind.train_loss = graphs["train_loss"]
        ind.valid_loss = graphs["valid_loss"]
        ind.train_dice = graphs["train_dice"]
        ind.valid_dice = graphs["valid_dice"]
        
        return
    
    #%%Evaluate individuals
    def evaluate_individuals(self, idx):
        idx_mp=[]
        devices = torch.cuda.device_count()
        
        while idx < len(self.invalid_ind):
            #individual and key
            ind = self.invalid_ind[idx]
            key = self.toolbox.identifier(ind)
            
            #If self.cache
            if key in self.cache:
                self.assign_attributes_cache(ind, key, self.cache)
                
                idx = idx + 1
                self.idx = idx
                # self.idx += 1
                
                print('Syntax tree:\t', str(ind), round(ind.fitness.values[0],3), round(ind.dice,3), ind.params, '\t in cache')
                
                #Best individual
                self.best = tools.selBest(self.population, 1)[0]
                print(f"{self.idx}/{len(self.invalid_ind)}", self.gen, self.foldername)
                # print('Best:', str(self.best), round(self.best.fitness.values[0],3), round(self.best.dice,3), self.best.params)
                
                #Take time and save it
                t = datetime.now()
                self.delta_t += (t - self.init_time)
                self.init_time = t #no importa perder t, conservar delta_t
                
                #Save progress every evaluation
                save_progress(self.ruta, 'checkpoint_evo.pkl',
                              self.population, self.best,
                              self.gen,
                              self.halloffame, self.logbook,
                              self.offspring,
                              self.elitism_inds,
                              self.invalid_ind, self.idx, 
                              self.no_evs, self.delta_t,
                              
                              self.cache)
                
            #else, multiprocessing
            else:
                idx_mp.append(idx)
                idx = idx + 1
                # idx_mp.append(self.idx)
                # self.idx += 1
                
            if len(idx_mp) == devices or (idx == len(self.invalid_ind) and len(idx_mp)>0):
                pool=mp.Pool(processes=4)
                for i in range(len(idx_mp)):
                    ind = self.invalid_ind[idx_mp[i]]
                    fit, dice, params, metrics, graphs = pool.apply(evaluationMP, args = (ind, self.nepochs, self.loss_fn, self.lr, 
                                                    self.max_params, self.w, self.loaders, self.pset, 
                                                    'cuda:'+str(i), self.ruta, self.verbose_train))
                    self.assign_attributes_evaluation(ind, fit, dice, params, metrics, graphs)
                    
                    #Add to self.cache
                    key = self.toolbox.identifier(ind)
                    self.cache[key]=ind
                    
                    #Update no_evaluations
                    self.no_evs += 1
                    
                pool.close()
                pool.join()
                self.idx = idx

                #Best individual
                self.best = tools.selBest(self.population, 1)[0]
                print('Index', idx_mp, self.gen, self.foldername)
                print('Best:', str(self.best), round(self.best.fitness.values[0],3), round(self.best.dice,3), self.best.params)
                
                #Take time and save it
                t = datetime.now()
                self.delta_t += (t - self.init_time)
                self.init_time = t #no importa perder t, conservar delta_t
                
                #Save progress every evaluation
                save_progress(self.ruta, 'checkpoint_evo.pkl',
                              self.population, self.best,
                              self.gen, 
                              self.halloffame, self.logbook,
                              
                              self.offspring,
                              self.elitism_inds,
                              self.invalid_ind, self.idx, 
                              self.no_evs, self.delta_t,
                              self.cache)
                idx_mp=[]
        return
    
    #%% Generational progress
    def generational_progress_checkpoint(self):#, start_gen, newg, invalid_ind, self.idx):
        # Si NO ha terminado con 'invalid ind' en la sesión anterior, 
        # mantiene generación.
        if self.checkpoint and self.ruta and self.idx<(len(self.invalid_ind)):
            
            # idx = self.idx
            self.evaluate_individuals(self.idx)
            
        #Si YA terminó con 'invalid_ind' en la sesión anterior, 
        #pasa a la siguiente generación O
        #Si No CHECKPOINT continua normalmente        
        elif (self.checkpoint and self.ruta and self.idx==(len(self.invalid_ind))) or (self.checkpoint == False or self.checkpoint==None or self.ruta==None):
            
            population_for_eli=[self.toolbox.clone(ind) for ind in self.population]
            self.elitism_inds = self.toolbox.selectElitism(population_for_eli, k=self.num_elit)
            self.offspring = self.toolbox.select(self.population, len(self.population) - self.num_elit)
            self.offspring = MyVar(self.offspring, self.toolbox, len(self.offspring), 
                                              self.cxpb, self.mutpb)
            self.invalid_ind = [ind for ind in self.offspring if not ind.fitness.valid]
            self.idx = 0
            # idx = 0
            
            # Evaluate individuals
            self.evaluate_individuals(self.idx)
        
            # add the best back to self.population:
            self.offspring.extend(self.elitism_inds)
    
            # Update the hall of fame with the generated individuals
            if self.halloffame is not None:
                self.halloffame.update(self.offspring)
            
            # Replace the current self.population by the offspring
            self.population[:] = self.offspring
        
        #Take the self.best individual
        self.best = tools.selBest(self.population, 1)[0]
        print('Best:', str(self.best), round(self.best.fitness.values[0],3), round(self.best.dice,3), self.best.params, self.gen)
        
        #Save tree and train and validation loss images as .png
        self.toolbox.save_ind(self.best, ruta=self.ruta, filename='best_tree'+self.foldername + str(self.gen))
        self.toolbox.save_graphtv(self.best, ruta=self.ruta, filename='TrainValidationLoss.png')
        self.toolbox.save_graphtvd(self.best, ruta=self.ruta, filename='TrainValidationDice.png')
        
        # Append the current generation statistics to the logbook
        record = self.stats.compile(self.population) if self.stats else {}
        self.logbook.record(gen=self.gen, nevals=self.no_evs, time=self.delta_t,
                            params=self.best.params,
                            dice=np.mean(self.best.dices), iou=np.mean(self.best.ious), hd=np.mean(self.best.hds),
                            **record)
        if self.verbose:
            print(self.logbook.stream)
        
        #Take time and save it
        t = datetime.now()
        self.delta_t += (t - self.init_time)
        self.init_time = t #no importa perder t, conservar delta_t
        
        #Also, save progress every generation
        save_progress(self.ruta, 'checkpoint_evo.pkl',
                          self.population, self.best,
                          self.gen, 
                          self.halloffame, self.logbook,
                          self.offspring,
                          self.elitism_inds,
                          self.invalid_ind, self.idx, 
                          self.no_evs, self.delta_t,
                          self.cache)
        return
    
        
    #%%Run 
    def run(self):
        self.init_time = datetime.now() #no importa perder t, conservar delta_t
        self.delta_t = timedelta(seconds=0)
        
        if self.checkpoint and self.ruta:
            #Recover information from file
            self.recover(self.ruta)
        else:
            # Generacion 0
            self.population = self.toolbox.population(n=self.pz)#self.make_population(self.pz)
            self.gen = 0
            
            self.halloffame = tools.HallOfFame(maxsize=self.num_elit)
            self.logbook    = tools.Logbook()
            
            self.logbook.header = ['gen', 'nevals', 'time', 'params', 'dice', 'iou', 'hd'] +(self.stats.fields if self.stats else [])
            
            self.invalid_ind = [ind for ind in self.population if not ind.fitness.valid]
            self.offspring = []
            self.elitism_inds = []
            
            #Count the number of evaluated individuals and no_evaluations
            self.idx=0
            self.no_evs=0
            # # idx=0
            
            #Evaluate individuals
            self.evaluate_individuals(self.idx)

            #Hall of fame
            if self.halloffame is not None:
                self.halloffame.update(self.population)
            
            #Select best and save their graph
            self.best=tools.selBest(self.population,1)[0]
            print('Best:', str(self.best), round(self.best.fitness.values[0],3), round(self.best.dice,3), self.best.params, self.gen)
            
            #Save tree and train and validation loss images as .png
            self.toolbox.save_ind(self.best, ruta=self.ruta ,filename='best_tree'+self.foldername+str(self.gen))
            self.toolbox.save_graphtv(self.best, ruta=self.ruta, filename='TrainValidationLoss.png')
            self.toolbox.save_graphtvd(self.best, ruta=self.ruta, filename='TrainValidationDice.png')
            
            #Take time and save it
            t = datetime.now()
            self.delta_t += (t - self.init_time)
            self.init_time = t #no importa perder t, conservar delta_t
            
            # Append the current generation statistics to the logbook
            record = self.stats.compile(self.population) if self.stats else {}
            self.logbook.record(gen=self.gen, nevals=self.no_evs, time=self.delta_t,
                                params=self.best.params,
                                dice=np.mean(self.best.dices), iou=np.mean(self.best.ious), hd=np.mean(self.best.hds),
                                **record)
            if self.verbose:
                print(self.logbook.stream)
            
            #Also, save progress every generation
            save_progress(self.ruta, 'checkpoint_evo.pkl',
                          self.population, self.best,
                          self.gen, 
                          self.halloffame, self.logbook,
                          
                          self.offspring,
                          self.elitism_inds,
                          self.invalid_ind, self.idx, 
                          self.no_evs, self.delta_t,
                          self.cache)

        """Tres posibles opciones para asignar generación de comienzo"""
        # Si NO ha terminado con 'invalid ind' en la sesión anterior, 
        # mantiene generación.
        if self.checkpoint and self.ruta and self.idx<(len(self.invalid_ind)):
            start_gen_aux=self.gen
              
        #Si YA terminó con 'invalid_inds' en la sesión anterior, 
        #pasa a la siguiente generación O
        #Si No CHECKPOINT continua normalmente
        elif (self.checkpoint and self.ruta and self.idx==(len(self.invalid_ind))) or (self.checkpoint == False or self.checkpoint==None or self.ruta==None):
            start_gen_aux=self.gen+1

        for gen in range(start_gen_aux, self.ngen+1):
            self.gen=gen
            print('\nGen:\t', self.gen)
            self.generational_progress_checkpoint()
        
        print('Time', self.delta_t)
        
        return self.population, self.logbook