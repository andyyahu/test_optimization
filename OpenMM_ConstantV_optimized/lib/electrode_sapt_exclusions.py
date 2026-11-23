#********** OpenMM Drivers
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import *

#*************************** README  **************************************
#  This module defines methods that introduce additional exclusions
#  for electrodes, and specific electrolyte molecules within the SAPT force field.
#  This module should thus be considered "part of" the force field,
#  as (non-electrode) exclusions are force-field specific
#**************************************************************************

"""
    Set up exclusions for electrodes and certain molecules in SAPT force field.
    Exclusions should be implemented separately for different molecule types.
    This SAPT_FF_exclusion class should figure out which methods to subsequently call
    based on the system to generate molecule-specific exclusions
"""


#**********************************
# Adds exclusions for intra-electrode interactions of atoms
# Call this method with lists of electrode atoms 'electrode1' and 'electrode2'
# and all long range interactions between atoms on this electrode(s) will
# be excluded.  These lists could be the same or different, and we take two lists to allow versatility
#**********************************
def exclusion_Electrode_NonbondedForce(sim, system, electrode1, electrode2, customNonbondedForce, nbondedForce, add_custom_exclusions=False):
    nb_add_exception = nbondedForce.addException

    if not add_custom_exclusions:
        if electrode1 == electrode2:
            n = len(electrode1)
            for i in range(n):
                indexi = electrode1[i]
                for j in range(i + 1, n):
                    nb_add_exception(indexi, electrode2[j], 0, 1, 0, True)
        else:
            for indexi in electrode1:
                for indexj in electrode2:
                    nb_add_exception(indexi, indexj, 0, 1, 0, True)
        return

    # If customNonbondedForce exclusions are explicitly requested, use canonical integer tuple set
    flagexclusions = set()
    for i in range(customNonbondedForce.getNumExclusions()):
        p1, p2 = customNonbondedForce.getExclusionParticles(i)
        flagexclusions.add((p1, p2) if p1 < p2 else (p2, p1))

    cnb_add_exclusion = customNonbondedForce.addExclusion
    if electrode1 == electrode2:
        n = len(electrode1)
        for i in range(n):
            indexi = electrode1[i]
            for j in range(i + 1, n):
                indexj = electrode2[j]
                pair = (indexi, indexj) if indexi < indexj else (indexj, indexi)
                if pair in flagexclusions:
                    continue
                flagexclusions.add(pair)
                cnb_add_exclusion(indexi, indexj)
                nb_add_exception(indexi, indexj, 0, 1, 0, True)
    else:
        for indexi in electrode1:
            for indexj in electrode2:
                pair = (indexi, indexj) if indexi < indexj else (indexj, indexi)
                if pair in flagexclusions:
                    continue
                flagexclusions.add(pair)
                cnb_add_exclusion(indexi, indexj)
                nb_add_exception(indexi, indexj, 0, 1, 0, True)


#*********************************
# Configure CustomNonbondedForce interaction groups for electrodes and electrolyte.
# Excludes intra-electrode interactions and all zero-parameter particles
# (virtual electrodes Cgtc/Cgtd, Drude particles Sh, dummy hydrogens Hgt/Hgts),
# while also handling hybrid water model (water-other vs notwater-notwater) if present.
#*********************************
def setup_electrode_customnonbonded_groups(MMsys, water_name='HOH', include_water_exclusions=True):
    customNonbondedForce = MMsys.customNonbondedForce

    cathode_all = {atom.atom_index for atom in MMsys.Cathode.electrode_atoms}
    for chain_atoms in MMsys.Cathode.electrode_extra_exclusions:
        cathode_all.update(chain_atoms)

    anode_all = {atom.atom_index for atom in MMsys.Anode.electrode_atoms}
    for chain_atoms in MMsys.Anode.electrode_extra_exclusions:
        anode_all.update(chain_atoms)

    conductor_groups = []
    conductor_all = set()
    for Conductor in MMsys.Conductor_list:
        c_set = {atom.atom_index for atom in Conductor.electrode_atoms}
        c_set.update(atom.atom_index for atom in Conductor.electrode_atoms_real)
        conductor_groups.append(c_set)
        conductor_all.update(c_set)

    all_electrode_atoms = cathode_all | anode_all | conductor_all

    water_atoms = set()
    if include_water_exclusions:
        for res in MMsys.simmd.topology.residues():
            if res.name == water_name:
                for atom in res._atoms:
                    water_atoms.add(atom.index)

    active_electrode = set()
    active_conductors = [set() for _ in conductor_groups]
    active_water = set()
    active_notwater_electrolyte = set()

    num_params = customNonbondedForce.getNumPerParticleParameters()
    check_param_indices = [
        i for i in range(num_params)
        if customNonbondedForce.getPerParticleParameterName(i) != 'Bexp'
    ]
    get_params = customNonbondedForce.getParticleParameters
    for idx in range(customNonbondedForce.getNumParticles()):
        if num_params > 0:
            p = get_params(idx)
            is_active = any(p[i] != 0.0 for i in check_param_indices) if check_param_indices else any(v != 0.0 for v in p)
        else:
            is_active = True
        if is_active:
            if idx in all_electrode_atoms:
                if idx in conductor_all:
                    for c_idx, c_set in enumerate(conductor_groups):
                        if idx in c_set:
                            active_conductors[c_idx].add(idx)
                            break
                else:
                    active_electrode.add(idx)
            elif idx in water_atoms:
                active_water.add(idx)
            else:
                active_notwater_electrolyte.add(idx)

    active_electrolyte_all = active_water | active_notwater_electrolyte

    if active_electrode and active_electrolyte_all:
        customNonbondedForce.addInteractionGroup(active_electrolyte_all, active_electrode)

    for c_idx, act_cond in enumerate(active_conductors):
        if act_cond:
            if active_electrolyte_all:
                customNonbondedForce.addInteractionGroup(active_electrolyte_all, act_cond)
            if active_electrode:
                customNonbondedForce.addInteractionGroup(active_electrode, act_cond)
            for c_idx2 in range(c_idx + 1, len(active_conductors)):
                if active_conductors[c_idx2]:
                    customNonbondedForce.addInteractionGroup(act_cond, active_conductors[c_idx2])

    if active_water:
        print('Creating Interaction Groups for CustomNonBonded.  These interactions will be computed between water-other, not water-water.')
        if active_notwater_electrolyte:
            customNonbondedForce.addInteractionGroup(active_water, active_notwater_electrolyte)

    if active_notwater_electrolyte:
        customNonbondedForce.addInteractionGroup(active_notwater_electrolyte, active_notwater_electrolyte)

    # If no non-empty interaction group was added (e.g. vacuum electrode-only system),
    # add an empty interaction group so OpenMM does not fall back to all-pairs evaluation.
    if customNonbondedForce.getNumInteractionGroups() == 0:
        customNonbondedForce.addInteractionGroup(set(), set())


#*********************************
# water models generally use specific water-water interaction parameters
# and separate interaction parameters for water-other
# here assume water-water interactions are in NonbondedForce, and
# water-other are in CustomNonbonded force, and create interaction
# groups for CustomNonbonded force...
#**********************************
def generate_exclusions_water(sim, customNonbondedForce, watername):
    # If interaction groups were already configured (e.g. by setup_electrode_customnonbonded_groups), skip duplicate addition
    if customNonbondedForce.getNumInteractionGroups() > 0:
        return

    # create Interaction Groups for hybrid water model
    water = set()
    notwater = set()
    print('Creating Interaction Groups for CustomNonBonded.  These interactions will be computed between water-other, not water-water.')
    for res in sim.topology.residues():
        if res.name == watername:
            for i in range(len(res._atoms)):
                water.update([res._atoms[i].index])
        else:
            for i in range(len(res._atoms)):
                notwater.update([res._atoms[i].index])

    customNonbondedForce.addInteractionGroup(water, notwater)
    customNonbondedForce.addInteractionGroup(notwater, notwater)


class SAPT_FF_exclusions(object):
    def __init__(self, MMsys):

        #************************** List of molecule types that have special exclusions ******
        # now see what molecule types are present.  These name are hardcoded in, i don't see any way around this since special exclusions
        # have to be molecule specific
        self.watername = 'HOH'
        self.TFSIname  = 'Tf2N'
        #***********************************************

        # local copies of force classes
        self.nbondedForce = MMsys.nbondedForce
        self.customNonbondedForce = MMsys.customNonbondedForce
        self.drudeForce = MMsys.drudeForce
        # will only have this for certain molecules
        self.custombond = MMsys.custombond

        # ************* Add Water exclusions
        for res in MMsys.simmd.topology.residues():
            if res.name == self.watername:
                generate_exclusions_water(MMsys.simmd, self.customNonbondedForce, self.watername)
                break

        # ************* Add TFSI exclusions, note might refer to this as Tf2N
        for res in MMsys.simmd.topology.residues():
            if res.name == self.TFSIname:
                self.generate_exclusions_TFSI(MMsys.simmd, MMsys.system)
                break

    def generate_exclusions_TFSI(self, sim, system):
        """
        This creates exclusions for TFSI nonbonded interactions, and update
        Screened Drude interactions.  1-5 non-Coulomb interaction are accounted for
        using CustomBondForce
        """
        print('Creating Exclusions for TFSI')

        # map from global particle index to drudeforce object index
        particleMap = {}
        for i in range(self.drudeForce.getNumParticles()):
            particleMap[self.drudeForce.getParticleParameters(i)[0]] = i

        # can't add duplicate customNonbonded exclusions, so store what we already have
        # Note: CustomNonbondedForce (bondCutoff=3) already has the exact same 1-2, 1-3, 1-4
        # exclusions as NonbondedForce exceptions, so scanning nbondedForce.getNumExceptions()
        # is redundant.
        flagexclusions = set()
        for i in range(self.customNonbondedForce.getNumExclusions()):
            p1, p2 = self.customNonbondedForce.getExclusionParticles(i)
            flagexclusions.add((p1, p2) if p1 < p2 else (p2, p1))

        # add exclusions for all atom pairs on TFSI residues, and when a drude pair is
        # excluded add a corresponding screened thole interaction in its place
        for res in sim.topology.residues():
            if res.name == self.TFSIname:
                for i in range(len(res._atoms) - 1):
                    for j in range(i + 1, len(res._atoms)):
                        indi, indj = res._atoms[i].index, res._atoms[j].index
                        # here it doesn't matter if we already have this, since we pass the "True" flag
                        self.nbondedForce.addException(indi, indj, 0, 1, 0, True)
                        # make sure we don't already exclude this customnonbond
                        pair = (indi, indj) if indi < indj else (indj, indi)
                        if pair in flagexclusions:
                            continue
                        flagexclusions.add(pair)
                        self.customNonbondedForce.addExclusion(indi, indj)
                        # add thole if we're excluding two drudes
                        if indi in particleMap and indj in particleMap:
                            drudei = particleMap[indi]
                            drudej = particleMap[indj]
                            self.drudeForce.addScreenedPair(drudei, drudej, 2.0)
