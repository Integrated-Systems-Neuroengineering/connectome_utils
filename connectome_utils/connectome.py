#!/usr/bin/env python3

import copy
import logging
from jaal import Jaal
import pandas as pd

class synapse:
    def __init__(self, presynapticNeuron, postsynapticNeuron, weight):
        self.presynapticNeuron = presynapticNeuron
        self.postsynapticNeuron = postsynapticNeuron
        self.weight = weight
        self.synapseType = None #is the synapse within core (homogeneous) or between core (heterogeneous)
        self.colIndex = None
        self.rowIndex = None

    def get_presynapticNeuron(self):
        return self.presynapticNeuron

    def get_postsynapticNeuron(self):
        return self.postsynapticNeuron

    def set_synapseType(self):
        #check if the synapse is between or withing core
        if self.postsynapticNeuron and self.presynapticNeuron:
            if self.postsynapticNeuron.get_core() != self.presynapticNeuron.get_core():
                self.synapseType = "hetero"
                #check if there is a relay axon to the post synaptic neuron
                relayAxon = self.postsynapticNeuron.get_relay_axon()
                #if it exists: it's the postsynaptic neuron
                if relayAxon:
                    self.postsynapticNeuron = relayAxon
                #if it doesn't exist: create it and add it to the connectome
                else:
                    self.postsynapticNeuron.set_relay_axon()
                    self.postsynapticNeuron = self.postsynapticNeuron.get_relay_axon()
            else:
                self.synapseType = "homo"
        else:
            raise Exception("core is None")
        return self.synapseType

    def get_weight(self):
        #get the synapse weight
        return self.weight

    def set_weight(self, newWeight):
        self.weight = newWeight

    def set_index(self, newRowIndex, newColIndex):
        self.rowIndex = newRowIndex
        self.colIndex = newColIndex

    def get_index(self):
        return self.rowIndex, self.colIndex

    def __repr__(self):
        return self.obj2string()

    def __str__(self):
        return self.obj2string()

    def obj2string(self):
        return (
            "("
            + str(self.postsynapticNeuron.get_user_key())
            + ", "
            + str(self.weight)
            + ", "
            + str(self.synapseType)
            + ")"
        )



class neuron:
    #dummy neuron/axons just exist for padding regions in HBM
    globalAxonCount = 0  # STATIC total number of axons
    globalNeuronCount = 0  # STATIC total number of nonAxon neurons
    globalCount = 0  # STATIC total number of both axons and neurons

    def __init__(self, userKey, neuronType, neuronModel=None, output=False, dummy=False, axonType = None):
        self.userKey = userKey #name user gave to neuron
        self.neuronType = neuronType #axon or neuron
        self.connectome = None #connectome that this neuron is part of
        self.alignment = 'homo' #track if the neuron has offcore synapses
        if neuronType == "axon":
            self.output = False #axons aren't output neurons
            if not dummy:
                self.globalTypeIdx = copy.deepcopy(neuron.globalAxonCount) #get current axon count
                neuron.globalAxonCount += 1 #increment it
                if axonType == 'Raxon' or axonType == 'Uaxon': #tag if this axon is a user created axon (U) or a routing axon (R)
                    self.axonType = axonType
                else:
                    raise ValueError('incorrect axonType supplied')

        if neuronType == "neuron":
            self.neuronModel = neuronModel
            self.relayAxon = None #if the neuron needs an axon to route incomping spikes from offcore synapses
            self.output = output #is the neuron an output neuron
            if not dummy:
                self.globalTypeIdx = copy.deepcopy(neuron.globalNeuronCount) #get count of neurons in network
                neuron.globalNeuronCount += 1
            self.output = output
        if not dummy:
            self.globalIdx = copy.deepcopy(neuron.globalCount) #get count of all neurons and axons
            neuron.globalCount += 1
            self.core = 0 #set core neuron belongs to
            self.coreTypeIdx = self.globalTypeIdx #get count of axon/neuron
            self.coreIdx = self.globalIdx # get count of all axon/neurons
        self.synapses = [] #initiate synapse list

    @classmethod
    def reset_count(cls):
        # resets counting variables for neuron class
        cls.globalAxonCount = 0
        cls.globalNeuronCount = 0
        cls.globalCount = 0


    def __lt__(self, other):
         return self.__class__.__name__ < other.__class__.__name__


    def addSynapse(self, postsynapticNeuron, weight):
        #append synapse to network
        newSynapse = synapse(self, postsynapticNeuron, weight) #create synapse object
        self.synapses.append(newSynapse) #add to synapse list

    def get_output(self): #check if output neuron
        return self.output

    def get_user_key(self): #get user neuron name
        return self.userKey

    def get_synapses(self): #get all synapses
        return self.synapses

    def get_neuron_type(self): #get if axon/neuron
        return self.neuronType

    def get_relay_axon(self): #get relay axon if neuron has one
        return self.relayAxon

    def get_alignment(self): #check if neunon has synapses off core
        try:
            return self.alignment
        except:
            return None #we don't currently know the alignment

    def get_axon_type(self): #get if axon is a user provided axon or a routing/relay axon
        try:
            return self.axonType
        except:
            return None

    def set_relay_axon(self):
        #assign a relay axon to the neuronu
        name = str(self.get_user_key())+('RAx') #get a name for axon
        relayAxon = neuron(userKey = name, neuronType = "axon", axonType = 'Raxon') #create the neuron
        relayAxon.set_core(self.core) #set to be same core as neuron
        connectome = self.get_connectome() #get the connectome object the neuron belongs to
        connectome.addNeuron(relayAxon) #add relay axon to connectome
        relayAxon.addSynapse(self, 1) #point relay axon to current neuron
        self.relayAxon = relayAxon
        return relayAxon


    # setter getter for core
    def set_core(self, core):
        self.core = core

    def get_core(self):
        return self.core

    #setter getter for the neuron/axon's core specific index
    def set_coreIdx(self, coreIdx):
        self.coreIdx = coreIdx

    def get_coreIdx(self):
        return self.coreIdx

    # setter getter for index specific to the neuron type for a specific core
    def set_coreTypeIdx(self, coreTypeIdx):
        self.coreTypeIdx = coreTypeIdx

    def get_coreTypeIdx(self):
        return self.coreTypeIdx

    # getter setter for unique index across all cores
    def set_globalIdx(self, globalIdx):
        self.globalIdx = globalIdx

    def get_globalIdx(self):
        return self.globalIdx


    #getter for specific synapse
    def get_synapse(self, postsynapticKey): #user supplies key of postsynaptic neuron
        canidateSynapses = [
            currSynapse
            for currSynapse in self.synapses
            if currSynapse.get_postsynapticNeuron().get_user_key() == postsynapticKey
        ] #get synapses by list comprehension
        if len(canidateSynapses) == 0:
            logging.error("Neuron has no synapse to specified postsynaptic neuron")
        else:
            if len(canidateSynapses) != 1:
                logging.error("Neuron has two synapses to the same postsynaptic neuron")
            else:
                return canidateSynapses[0]

    def set_synapseTypes(self): #check every neuron for if it's a within or between core synapse

        for synapse in self.synapses:
            synapseType = synapse.set_synapseType()
            print(self.neuronType)
            if synapseType == 'hetero' and self.alignment == 'homo': #if the neuron has >= 1 offcore synapse tag it as heterogenous alignment neuron
                try:
                    self.alignment == 'hetero'
                except:
                    breakpoint()


    def get_neuronModel(self):
        return self.neuronModel

    def __repr__(self):
        return self.obj2string()

    def __str__(self):
        return self.obj2string()

    def obj2string(self):
        string = (
            "userID: "
            + str(self.userKey)
            + ", GlobalID: "
            + str(self.globalIdx)
            + ", Core Assignment: "
            + str(self.core)
            + ", Synapses: "
        )
        for currSynapse in self.synapses:
            string = string + str(currSynapse)
        return string

    #return if the neuron is tagged as an output neuron
    def get_output(self):
        return self.output

    #set the connectome the neuron belongs to
    def set_connectome(self, connectome):
        self.connectome = connectome

    #get the connectome
    def get_connectome(self):
        return self.connectome


class connectome:
    # connectomeDict = None
    def __init__(self):
        self.connectomeDict = {}
        self.axons = {}
        self.neurons = {}
        self.mergedNeurons = {} #contain  both axons and neurons
        self.cutoffs = []

    #add neuron to connectome
    def addNeuron(self, neuron):
        self.connectomeDict[neuron.get_user_key()] = neuron
        neuron.set_connectome(self) #set the neurons connectome

    def __repr__(self):
        return self.obj2string()

    def __str__(self):
        return self.obj2string()

    def get_neuron_by_key(self, neuronKey):
        return self.connectomeDict[neuronKey]

    def get_neuron_by_idx(self, idx): #get neuron by coreTypeIdx
        for key in self.connectomeDict:
            # axons and
            if (
                self.connectomeDict[key].get_neuron_type() == "neuron"
                and self.connectomeDict[key].get_coreTypeIdx() == idx
            ):
                return self.connectomeDict[key]

    def get_axon_by_idx(self, idx): #get axon by coreTypeIdx
        for key in self.connectomeDict:
            if (
                self.connectomeDict[key].get_neuron_type() == "axon"
                and self.connectomeDict[key].get_coreTypeIdx() == idx
            ):
                return self.connectomeDict[key]

    def obj2string(self):
        string = ""
        for key in self.connectomeDict:
            string = string + str(self.connectomeDict[key]) + "\n"
        return string

    def get_axons(self): #searches throgh all neurons/axons and adds axons to axons dictionary and returns dictionary
        # Update Axons dictionary
        for key in self.connectomeDict:
            if self.connectomeDict[key].get_neuron_type() == "axon": #if axon add to axons dict
                self.axons[key] = self.connectomeDict[key]
            elif key in self.axons: #else remove key if in axons dict
                self.axons.pop(key)
        return self.axons

    def get_neurons(self): #same as get axons but for neurons
        # Update Neurons dictionary
        for key in self.connectomeDict:
            if self.connectomeDict[key].get_neuron_type() == "neuron":
                self.neurons[key] = self.connectomeDict[key]
            elif key in self.neurons:
                self.neurons.pop(key)
        return self.neurons

    def get_merged_neurons(self): #update and get dictionary off all axons/neurons indexed by gloabal index
        # Update get_merge_neurons dictionary
        for key in self.connectomeDict:
            self.mergedNeurons[
                self.connectomeDict[key].get_globalIdx()
            ] = self.connectomeDict[key]
        return self.mergedNeurons

    def update_class_ordered_coreIdx(self):# return a list of neurons by order of their neuron model
        self.get_neurons() #update neurons dictionary
        dict_list=list(self.neurons.items())
        dict_list.sort(key=lambda x: x[1].neuronModel) #sort by neuron class
        breakpoint()
        for idx, elem in enumerate(dict_list):
            elem[1].coreTypeIdx = idx

    def get_class_ordered_list(self):# return a list of neurons by order of their neuron model
        self.get_neurons() #update neurons dictionary
        dict_list=list(self.neurons.items())
        dict_list.sort(key=lambda x: x[1].neuronModel) #sort by neuron class
        return dict_list


    def get_part_format(self): #return the connectome in a format the partitioning algorithm expects
        mergedNeurons = self.get_merged_neurons()
        networkConnectivity = []
        for (
            key
        ) in (
            mergedNeurons.keys()
        ):  # Nishant's partitioning code expects a specific format so we convert to that format
            networkConnectivity.append(
                [
                    copy.deepcopy(synapse.get_postsynapticNeuron().get_globalIdx())
                    for synapse in mergedNeurons[key].get_synapses()
                ]
            )
        return networkConnectivity

    def get_core_outputs_idx(self, core): #get output neurons for a specific core
        outputs = []
        for key in self.connectomeDict:
            currNeuron = self.connectomeDict[key]
            if (
                currNeuron.get_output() == True
                and currNeuron.get_neuron_type() == "neuron"
                and currNeuron.get_core() == core
            ):
                outputs.append(currNeuron.get_coreTypeIdx())
        return outputs

    def get_models(self): # get sorted list of neuron models
        self.get_neurons() #update neurons dictionary
        dict_list=list(self.neurons.items())
        #model_list = [elem.get_neuronModel() for elem in dict_list] #sort by neuron class
        breakpoint()
        model_set = set()
        for elem in dict_list:
            model_set.add(elem[1].get_neuronModel())
        model_list = list(model_set)
        model_list.sort()
        return model_list

    def get_neuron_by_model(self, model): #get all the neurons of a certain model
        self.get_neurons() #update neurons dictionary
        dict_list=list(self.neurons.items())
        return [ elem[1] for elem in dict_list if elem[1].get_neuronModel() == model ]

    def pad_models(self): #add 'dummy' neurons to the connectome so that definitions of neurons for a model line up in HBM correctly
        breakpoint()
        pad_idx = 0
        model_list = self.get_models()
        cutoffs = []
        cutoff = 0
        for model in model_list:
            currList = self.get_neuron_by_model(model)
            if len(currList)%32 != 0:
                remainder = 32-(len(currList)%32)
                for i in range(remainder):
                    padNeuron = neuron('pad'+str(pad_idx), neuronType="neuron", neuronModel=model, output=False, dummy=False)
                    pad_idx = pad_idx + 1
                    self.addNeuron(padNeuron)
                cutoff += len(currList)+remainder
            else:
                cutoff += len(currList)
            cutoffs.append(cutoff)
        self.update_class_ordered_coreIdx()
        self.cutoffs = cutoffs



    def get_outputs_idx(self): #get all output neurons
        outputs = []
        for key in self.connectomeDict:
            currNeuron = self.connectomeDict[key]
            if (
                currNeuron.get_output() == True
                and currNeuron.get_neuron_type() == "neuron"
            ):
                outputs.append(currNeuron.get_coreTypeIdx())
        return outputs

    def apply_partition(self, membership): #apply a partition to the network
        mergedNeurons = self.get_merged_neurons()
        for key in membership.keys():
            mergedNeurons[key].set_core(membership[key]) #set core designation
        dictkeys = list(self.connectomeDict.keys()) #cast to list since we'll be modifying the dict
        for neuronKey in dictkeys:
            self.connectomeDict[neuronKey].set_synapseTypes() #update synapses and instantiate any relay neurons if needed

    def gen_neuron_df(self): #create Node dictionary for graph viz
        df = pd.DataFrame(columns=('id', 'type', 'alignment', 'core', 'axonType'))
         # Update Neurons dictionary
        for key in self.connectomeDict:
            neuron = self.connectomeDict[key]
            row = [ key, neuron.get_neuron_type(), neuron.get_alignment(), str(neuron.get_core()), neuron.get_axon_type()
            ]
            df.loc[ len(df) ] = row

        return df


    def gen_synapse_df(self): #generate an edge dictionary for graph viz
        breakpoint()
        df = pd.DataFrame(columns=('to', 'from'))
        for key in self.connectomeDict:
            neuron = self.connectomeDict[key]
            synapses = neuron.get_synapses()
            for synapse in synapses:
                toVal = synapse.get_postsynapticNeuron().get_user_key()
                fromVal = synapse.get_presynapticNeuron().get_user_key()
                row = [ toVal, fromVal]
                df.loc[ len(df) ] = row

        breakpoint()
        return df


    def graph_viz(self): #make a graph visualizaiton
       node_df =  self.gen_neuron_df()
       edge_df = self.gen_synapse_df()
       breakpoint()
       Jaal(edge_df, node_df).plot(directed=True, host = '0.0.0.0', port = '8050') #host is set to allow access from remote client
