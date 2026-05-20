#!/usr/bin/env python3
"""Connectome modeling module for neural network representation.

This module provides classes to represent and manipulate neural network
connectomes, including neurons, axons, and synapses. It supports partitioning
networks across multiple cores and tracking connectivity relationships.
"""

import copy
import logging


class synapse:
    """Represents a synaptic connection between two neurons.

    A synapse connects a presynaptic neuron (source) to a postsynaptic neuron
    (target) with a given weight. Synapses can be classified as homogeneous
    (within the same core) or heterogeneous (between different cores).

    Attributes
    ----------
    presynapticNeuron : neuron
        The source neuron of the synapse.
    postsynapticNeuron : neuron
        The target neuron of the synapse.
    weight : float
        The synaptic weight.
    synapseType : str or None
        Either 'homo' (within core) or 'hetero' (between cores).
    colIndex : int or None
        Column index in the connectivity matrix.
    rowIndex : int or None
        Row index in the connectivity matrix.
    """

    def __init__(self, presynapticNeuron, postsynapticNeuron, weight, delayed=False):
        """Initialize a synapse.

        Parameters
        ----------
        presynapticNeuron : neuron
            The source neuron of the synapse.
        postsynapticNeuron : neuron
            The target neuron of the synapse.
        weight : float
            The synaptic weight.
        """
        self.presynapticNeuron = presynapticNeuron
        self.postsynapticNeuron = postsynapticNeuron
        self.weight = weight
        self.delayed = delayed
    def is_delayed(self):
        return self.delayed
        self.synapseType = None  # is the synapse within core (homogeneous) or between core (heterogeneous)
        self.colIndex = None
        self.rowIndex = None

    def get_presynapticNeuron(self):
        """Return the presynaptic (source) neuron."""
        return self.presynapticNeuron

    def get_postsynapticNeuron(self):
        """Return the postsynaptic (target) neuron."""
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
            self.hbmIdx = None
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


    def addSynapse(self, postsynapticNeuron, weight, delayed=False):
        #append synapse to network
        newSynapse = synapse(self, postsynapticNeuron, weight, delayed=delayed) #create synapse object
        self.synapses.append(newSynapse) #add to synapse list

    def set_hbmIdx(self,idx):
        self.hbmIdx = idx

    def get_hbmIdx(self):
        return self.hbmIdx

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

    def set_synapseTypes(self):
        """Check every synapse to determine if it's within or between core."""
        for synapse in self.synapses:
            synapseType = synapse.set_synapseType()
            if synapseType == 'hetero' and self.alignment == 'homo':
                self.alignment = 'hetero'


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

    def set_connectome(self, connectome):
        self.connectome = connectome

    #get the connectome
    def get_connectome(self):
        return self.connectome


class connectome:
    def __init__(self):
        self.neuronArr = []
        self.axonArr = []
        self.pureNeuronArr = []
        self.outputs = []
        self.neuronModelIdxs = {}  # hash(neuronModel) -> index in neuronModels
        self.neuronModels = []  # list of lists: neuronModels[modelIdx] = [neuronArr indices]
        self.coreArrHbm = []  # coreArrHbm[coreIdx][hbmIdx] = neuronArr index
        self.connectomeDict = {}  # userKey -> neuronArr index
        self.mergedNeurons = {}  # globalIdx -> neuron object
        self.cutoffs = []
        self.pureNeuronIdxLookup = {}  # userKey -> pureNeuronArr index

    def addNeuron(self, neuron):
        """Add a neuron or axon to the connectome."""
        index = len(self.neuronArr)
        self.neuronArr.append(neuron)
        self.connectomeDict[neuron.get_user_key()] = index

        if neuron.get_neuron_type() == 'neuron':
            position = len(self.pureNeuronArr)
            self.pureNeuronArr.append(index)
            self.pureNeuronIdxLookup[neuron.get_user_key()] = position
            if neuron.get_output():
                self.outputs.append(position)
            # Track neuron by model type
            modelKey = hash(neuron.get_neuronModel())
            if modelKey not in self.neuronModelIdxs:
                self.neuronModelIdxs[modelKey] = len(self.neuronModels)
                self.neuronModels.append([])
            self.neuronModels[self.neuronModelIdxs[modelKey]].append(index)

        if neuron.get_neuron_type() == 'axon':
            self.axonArr.append(index)

        neuron.set_connectome(self)

    def __repr__(self):
        return self.obj2string()

    def __str__(self):
        return self.obj2string()

    def get_neuron_by_key(self, neuronKey):
        return self.neuronArr[self.connectomeDict[neuronKey]]

    def get_pureNeuron_idx(self, neuronKey):
        return self.pureNeuronIdxLookup[neuronKey]

    def get_neuron_by_idx(self, idx):
        return self.neuronArr[self.pureNeuronArr[idx]]

    def get_neuron_by_hbmIdx(self, idx, core=0):
        masterIdx = self.coreArrHbm[core][idx]
        return self.neuronArr[masterIdx]


    #this function has problems
    def get_axon_by_idx(self, idx): #get axon by coreTypeIdx
        for key in self.connectomeDict:
            neuron = self.neuronArr[self.connectomeDict[key]]
            if (
                neuron.get_neuron_type() == "axon"
                and neuron.get_coreTypeIdx() == idx
            ):
                return neuron

    def obj2string(self):
        string = ""
        for key in self.connectomeDict:
            string = string + str(self.connectomeDict[key]) + "\n"
        return string

    def get_axons(self):
        return [self.neuronArr[i] for i in self.axonArr]

    def get_neurons(self):
        return [self.neuronArr[i] for i in self.pureNeuronArr]

    #this should probably get deprecated
    def get_merged_neurons(self): #update and get dictionary off all axons/neurons indexed by gloabal index
        # Update get_merge_neurons dictionary
        for key in self.connectomeDict:
            neuron = self.neuronArr[self.connectomeDict[key]]
            self.mergedNeurons[neuron.get_globalIdx()] = neuron
        return self.mergedNeurons

    def update_class_ordered_coreIdx(self):
        """Update coreTypeIdx for neurons sorted by their neuron model."""
        neurons = self.get_neurons()
        neurons.sort(key=lambda x: x.neuronModel)
        for idx, neuron in enumerate(neurons):
            neuron.coreTypeIdx = idx

    def get_class_ordered_list(self, core=0):
        """Return neurons for a specific core sorted by neuron model, with per-core hbmIdx assigned.

        Parameters
        ----------
        core : int
            The core index to get neurons for.

        Returns
        -------
        list of tuple
            List of (userKey, neuron) tuples sorted by neuron model for the specified core.
        """
        neurons = self.get_neurons()
        neuron_list = [(n.get_user_key(), n) for n in neurons if n.get_core() == core]
        neuron_list.sort(key=lambda x: x[1].neuronModel)

        # Ensure coreArrHbm is large enough for this core index
        while len(self.coreArrHbm) <= core:
            self.coreArrHbm.append([])

        core_lookup = []
        for idx, (key, neuron) in enumerate(neuron_list):
            neuron.set_hbmIdx(idx)
            core_lookup.append(self.connectomeDict[key])
        self.coreArrHbm[core] = core_lookup

        return neuron_list


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
            currNeuron = self.neuronArr[self.connectomeDict[key]]
            if (
                currNeuron.get_output() == True
                and currNeuron.get_neuron_type() == "neuron"
                and currNeuron.get_core() == core
            ):
                outputs.append(currNeuron.get_coreTypeIdx())
        return outputs

    def get_neuron_by_model(self, model):
        """Return all neurons with the specified neuron model."""
        return [n for n in self.get_neurons() if n.get_neuronModel() == model]

    def pad_models(self):
        padIdx = 0
        cutoffs = []
        cutoff = 0
        for model in self.neuronModels:
            # model is a list of indices into self.neuronArr
            currNeuronModel = self.neuronArr[model[0]].get_neuronModel()
            numNeurons = len(model)
            if numNeurons % 32 != 0:
                remainder = 32 - (numNeurons % 32)
                for i in range(remainder):
                    padNeuron = neuron('pad'+str(padIdx), neuronType="neuron", neuronModel=currNeuronModel, output=False, dummy=False)
                    padIdx = padIdx + 1
                    self.addNeuron(padNeuron)
                cutoff += numNeurons + remainder
            else:
                cutoff += numNeurons
            cutoffs.append(cutoff)
        self.update_class_ordered_coreIdx()
        self.cutoffs = cutoffs




    def get_outputs_idx(self):
        return self.outputs

    def get_output_neurons(self):
        return [self.neuronArr[self.pureNeuronArr[i]] for i in self.outputs]

    def get_models(self):
        """Return unique neuron model objects in the order they appear in neuronModels."""
        return [self.neuronArr[model_indices[0]].get_neuronModel()
                for model_indices in self.neuronModels]

    def apply_partition(self, membership):
        """Apply a partition to the network, assigning neurons to cores.

        Parameters
        ----------
        membership : dict
            Mapping of globalIdx -> core assignment.
        """
        mergedNeurons = self.get_merged_neurons()
        for key in membership.keys():
            mergedNeurons[key].set_core(membership[key])
        dictkeys = list(self.connectomeDict.keys())
        for neuronKey in dictkeys:
            self.neuronArr[self.connectomeDict[neuronKey]].set_synapseTypes()
