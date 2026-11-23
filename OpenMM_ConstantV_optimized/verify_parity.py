#!/usr/bin/env python
"""
Comprehensive verification and benchmark script comparing OpenMM_ConstantV_original and OpenMM_ConstantV_optimized.
Verifies:
1. Exclusions / Exceptions reduction & initialization speedup + per-force-group energy and per-atom force parity (CUDA FP64)
2. Poisson_solver_fixed_voltage charge parity across voltages (CUDA FP64) & solver speedup (CUDA mixed)
3. Multi-step Constant-V MD trajectory parity (CUDA FP64)
4. MC_Barostat_step trial move coordinate parity & multi-step MC equilibration parity (CUDA FP64)
5. Vacuum electrode-only (no-electrolyte) CustomNonbondedForce energy/force parity
6. Buckyball_Virtual & Nanotube_Virtual (Conductor_list non-empty) energy/force & Poisson solver parity + atom.charge sync invariant
7. Hybrid water (HOH) + electrode 3-group CustomNonbondedForce parity
8. Standalone sapt_generate_exclusions(sim, system, positions) parity
"""
import sys
import os
import time
import random
import tempfile
import numpy as np
from simtk.openmm import Platform, Vec3, VerletIntegrator
from simtk.openmm.app import PDBFile, Modeller, Topology, Simulation, Element, ForceField, CutoffPeriodic
from simtk.unit import kilojoule_per_mole, nanometer, bar

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORIG_DIR = os.path.join(ROOT_DIR, 'OpenMM_ConstantV_original')
OPT_DIR  = os.path.join(ROOT_DIR, 'OpenMM_ConstantV_optimized')


def load_modules(base_dir):
    for mod_name in ['MM_classes', 'Fixed_Voltage_routines', 'electrode_sapt_exclusions', 'sapt_exclusions']:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    sys.path.insert(0, os.path.join(base_dir, 'lib'))
    sys.path.insert(0, base_dir)
    import MM_classes
    import Fixed_Voltage_routines
    import electrode_sapt_exclusions
    import sapt_exclusions
    sys.path.pop(0)
    sys.path.pop(0)
    return MM_classes, Fixed_Voltage_routines, electrode_sapt_exclusions, sapt_exclusions


def build_system(MM_cls, base_dir, voltage=1.0, precision='mixed', seed=42):
    ffdir = os.path.join(base_dir, 'ffdir') + '/'
    pdb = os.path.join(base_dir, 'nvt_0V_15ns.pdb')
    res_xml = [ffdir + 'sapt_residues.xml', ffdir + 'graph_residue_c.xml', ffdir + 'graph_residue_n.xml']
    ff_xml  = [ffdir + 'sapt_noDB_2sheets.xml', ffdir + 'graph_c_freeze.xml', ffdir + 'graph_n_freeze.xml']

    t0 = time.perf_counter()
    m = MM_cls(pdb_list=[pdb], residue_xml_list=res_xml, ff_xml_list=ff_xml)
    m.integrator.setRandomNumberSeed(seed)
    m.set_periodic_residue(True)
    m.platform = Platform.getPlatformByName('CUDA')
    m.properties = {'Precision': precision, 'DeterministicForces': 'true'}
    m.simmd = Simulation(m.modeller.topology, m.system, m.integrator, m.platform, m.properties)
    m.simmd.context.setPositions(m.modeller.positions)
    m.initialize_electrodes(voltage, cathode_identifier=(0, 2), anode_identifier=(1, 3), chain=True, exclude_element=('H',))
    m.initialize_electrolyte(Natom_cutoff=100)
    t_ex0 = time.perf_counter()
    m.generate_exclusions(flag_SAPT_FF_exclusions=True)
    t1 = time.perf_counter()
    return m, t1 - t0, t1 - t_ex0


def test_mc_barostat(MM_orig_mod, MM_opt_mod):
    print('\n======================================================================')
    print('4. MC_Barostat_step Trial Move & Multi-Step Parity (CUDA double)')
    print('======================================================================')
    m1, _, _ = build_system(MM_orig_mod.MM, ORIG_DIR, voltage=0.0, precision='double', seed=777)
    m2, _, _ = build_system(MM_opt_mod.MM,  OPT_DIR,  voltage=0.0, precision='double', seed=777)
    celldim1 = m1.simmd.topology.getUnitCellDimensions()
    celldim2 = m2.simmd.topology.getUnitCellDimensions()
    m1.MC = MM_orig_mod.MC_parameters(m1.temperature, celldim1, electrode_move="Anode", pressure=1.0*bar, barofreq=20, shiftscale=0.2)
    m2.MC = MM_opt_mod.MC_parameters(m2.temperature,  celldim2, electrode_move="Anode", pressure=1.0*bar, barofreq=20, shiftscale=0.2)

    state_rng = random.Random(999).getstate()
    for step_idx in range(3):
        random.setstate(state_rng)
        m1.MC_Barostat_step()
        next_rng = random.getstate()

        random.setstate(state_rng)
        m2.MC_Barostat_step()
        state_rng = next_rng

    pos1 = m1.simmd.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
    pos2 = m2.simmd.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
    dPos = float(np.max(np.abs(pos1 - pos2)))
    dLcell = abs(m1.Lcell - m2.Lcell)
    print(f'  After 3 MC_Barostat_step calls: naccept=({m1.MC.naccept}, {m2.MC.naccept}), dLcell={dLcell:.3e} nm, max|dPos|={dPos:.3e} nm')
    assert m1.MC.naccept == m2.MC.naccept
    assert dLcell < 1e-12 and dPos < 1e-7


def test_vacuum_electrodes(MM_orig_mod, MM_opt_mod, tmpdir):
    print('\n======================================================================')
    print('5. Vacuum Electrode-Only (No Electrolyte) CustomNonbondedForce Parity')
    print('======================================================================')
    ffdir = os.path.join(ORIG_DIR, 'ffdir') + '/'
    res_xml = [ffdir + 'sapt_residues.xml', ffdir + 'graph_residue_c.xml', ffdir + 'graph_residue_n.xml']
    ff_xml  = [ffdir + 'sapt_noDB_2sheets.xml', ffdir + 'graph_c_freeze.xml', ffdir + 'graph_n_freeze.xml']
    for r in res_xml:
        Topology().loadBondDefinitions(r)

    pdb = PDBFile(os.path.join(ORIG_DIR, 'nvt_0V_15ns.pdb'))
    mod = Modeller(pdb.topology, pdb.positions)
    mod.delete([r for r in mod.topology.residues() if r.chain.index >= 4])
    vac_pdb = os.path.join(tmpdir, 'vacuum.pdb')
    with open(vac_pdb, 'w') as f:
        PDBFile.writeFile(mod.topology, mod.positions, f)

    def run_vac(MM_cls):
        m = MM_cls(pdb_list=[vac_pdb], residue_xml_list=res_xml, ff_xml_list=ff_xml)
        m.set_periodic_residue(True)
        m.set_platform('Reference')
        m.initialize_electrodes(1.0, cathode_identifier=(0, 2), anode_identifier=(1, 3), chain=True, exclude_element=('H',))
        m.initialize_electrolyte(Natom_cutoff=100)
        m.generate_exclusions(flag_SAPT_FF_exclusions=False)
        for j in range(m.system.getNumForces()):
            if type(m.system.getForce(j)).__name__ == 'CustomNonbondedForce':
                return m.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)

    e_orig = run_vac(MM_orig_mod.MM)
    e_opt  = run_vac(MM_opt_mod.MM)
    print(f'  Vacuum CustomNonbondedForce Energy: orig={e_orig:.6e} kJ/mol, opt={e_opt:.6e} kJ/mol')
    assert abs(e_orig - e_opt) == 0.0


def test_buckyball_and_nanotube(MM_orig_mod, MM_opt_mod, tmpdir):
    print('\n======================================================================')
    print('6. Buckyball_Virtual & Nanotube_Virtual Parity + atom.charge Invariant')
    print('======================================================================')
    with open(os.path.join(ORIG_DIR, 'ffdir', 'graph_residue_s.xml')) as f:
        s_res = f.read()
    virt_res_path = os.path.join(tmpdir, 'cond_virtual_res.xml')
    with open(virt_res_path, 'w') as f:
        f.write(s_res.replace('grps', 'grpd').replace('graphites-', 'graphited-').replace('<Atom name="H0" type="graphited-H0"/>', ''))

    with open(os.path.join(ORIG_DIR, 'ffdir', 'graph_s_freeze.xml')) as f:
        s_ff = f.read()
    virt_ff_path = os.path.join(tmpdir, 'cond_virtual_ff.xml')
    with open(virt_ff_path, 'w') as f:
        f.write(
            s_ff.replace('grps', 'grpd')
                .replace('graphites-', 'graphited-')
                .replace('Cgts', 'Cgtd')
                .replace('<Type name="graphited-H0" class="Hgts" element="H" mass="0.0"/>', '')
                .replace('<Atom name="H0" type="graphited-H0"/>', '')
        )

    ffdir = os.path.join(ORIG_DIR, 'ffdir') + '/'
    res_xml = [ffdir + 'sapt_residues.xml', ffdir + 'graph_residue_c.xml', ffdir + 'graph_residue_n.xml', ffdir + 'graph_residue_s.xml', virt_res_path]
    ff_xml  = [ffdir + 'sapt_noDB_2sheets.xml', ffdir + 'graph_c_freeze.xml', ffdir + 'graph_n_freeze.xml', ffdir + 'graph_s_freeze.xml', virt_ff_path]
    for r in res_xml:
        Topology().loadBondDefinitions(r)

    pdb = PDBFile(os.path.join(ORIG_DIR, 'nvt_0V_15ns.pdb'))
    old_pos = pdb.positions.value_in_unit(nanometer)
    keep_chains = list(pdb.topology.chains())[:4]
    bmim_res = [r for r in pdb.topology.residues() if r.name == 'BMIM'][:4]
    tf2n_res = [r for r in pdb.topology.residues() if r.name == 'Tf2N'][:4]

    n_c = 195
    center = np.array([2.4, 2.4, 1.95])
    radius = 0.42

    # Build sphere points (Buckyball) and cylinder points (Nanotube)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    sphere_pts = []
    for i in range(n_c):
        y = 1.0 - (i / float(n_c - 1)) * 2.0
        r_xy = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        sphere_pts.append(center + radius * np.array([np.cos(theta) * r_xy, y, np.sin(theta) * r_xy]))

    cyl_pts = []
    xs = np.linspace(-1.8, 1.8, 13)
    for ix in range(13):
        for it in range(15):
            theta = 2.0 * np.pi * it / 15
            cyl_pts.append(np.array([center[0] + xs[ix], center[1] + radius * np.cos(theta), center[2] + radius * np.sin(theta)]))

    for mode, pts in [('Buckyball', sphere_pts), ('Nanotube', cyl_pts)]:
        top = Topology()
        top.setPeriodicBoxVectors(pdb.topology.getPeriodicBoxVectors())
        pos = []
        for c in keep_chains:
            nc = top.addChain()
            for r in c.residues():
                nr = top.addResidue(r.name, nc)
                for a in r.atoms():
                    top.addAtom(a.name, a.element, nr)
                    pos.append(Vec3(*old_pos[a.index]))
        for res_name, has_h in [('grpd', False), ('grps', True)]:
            nc = top.addChain()
            nr = top.addResidue(res_name, nc)
            for i in range(n_c):
                top.addAtom(f'C{i}', Element.getBySymbol('C'), nr)
                pos.append(Vec3(*pts[i]))
            if has_h:
                top.addAtom('H0', Element.getBySymbol('H'), nr)
                pos.append(Vec3(*center))
        nc_el = top.addChain()
        for r in bmim_res + tf2n_res:
            nr = top.addResidue(r.name, nc_el)
            res_pts = np.array([old_pos[a.index] for a in r.atoms()])
            shift = np.array([0.0, 0.0, 1.5]) if np.min(np.abs(res_pts[:, 2] - center[2])) < 0.85 else np.zeros(3)
            for a in r.atoms():
                top.addAtom(a.name, a.element, nr)
                pos.append(Vec3(*(old_pos[a.index] + shift)))
        top.createStandardBonds()
        cond_pdb = os.path.join(tmpdir, f'{mode}.pdb')
        with open(cond_pdb, 'w') as f:
            PDBFile.writeFile(top, pos * nanometer, f)

        def build_cond(MM_cls):
            m = MM_cls(pdb_list=[cond_pdb], residue_xml_list=res_xml, ff_xml_list=ff_xml)
            m.set_periodic_residue(True)
            m.platform = Platform.getPlatformByName('CUDA')
            m.properties = {'Precision': 'double', 'DeterministicForces': 'true'}
            m.simmd = Simulation(m.modeller.topology, m.system, m.integrator, m.platform, m.properties)
            m.simmd.context.setPositions(m.modeller.positions)
            kw = {'BuckyBalls': [(4, 5)]} if mode == 'Buckyball' else {'NanoTubes': [(4, 5)], 'nanotube_axis': [(1.0, 0.0, 0.0)]}
            m.initialize_electrodes(4.0, cathode_identifier=(0, 2), anode_identifier=(1, 3), chain=True, exclude_element=('H',), **kw)
            m.initialize_electrolyte(Natom_cutoff=100)
            m.generate_exclusions(flag_SAPT_FF_exclusions=True)
            return m

        m1 = build_cond(MM_orig_mod.MM)
        m2 = build_cond(MM_opt_mod.MM)
        m1.Poisson_solver_fixed_voltage(Niterations=4)
        m2.Poisson_solver_fixed_voltage(Niterations=4)
        qc1 = np.array([a.charge for a in m1.Cathode.electrode_atoms])
        qc2 = np.array([a.charge for a in m2.Cathode.electrode_atoms])
        qa1 = np.array([a.charge for a in m1.Anode.electrode_atoms])
        qa2 = np.array([a.charge for a in m2.Anode.electrode_atoms])
        qk1 = np.array([a.charge for a in m1.Conductor_list[0].electrode_atoms])
        qk2 = np.array([a.charge for a in m2.Conductor_list[0].electrode_atoms])
        dqc = float(np.max(np.abs(qc1 - qc2)))
        dqa = float(np.max(np.abs(qa1 - qa2)))
        dqk = float(np.max(np.abs(qk1 - qk2)))
        sync_diff_anod = float(np.max(np.abs(qa2 - m2.Anode.electrode_charges_np)))
        sync_diff_cath = float(np.max(np.abs(qc2 - m2.Cathode.electrode_charges_np)))
        sync_diff_cond = float(np.max(np.abs(qk2 - m2.Conductor_list[0].electrode_charges_np)))
        print(f'  [{mode:9s}] max|dq_cath|={dqc:.3e}, max|dq_anod|={dqa:.3e}, max|dq_cond|={dqk:.3e}, sync_diff=({sync_diff_cath:.1e}, {sync_diff_anod:.1e}, {sync_diff_cond:.1e})')
        assert dqc < 1e-7 and dqa < 1e-7 and dqk < 1e-7
        assert sync_diff_anod == 0.0 and sync_diff_cath == 0.0 and sync_diff_cond == 0.0

        # Test direct external mutation of atom.charge
        old_val = m2.Cathode.electrode_atoms[0].charge
        m2.Cathode.electrode_atoms[0].charge = old_val + 0.125
        assert m2.Cathode.electrode_charges_np[0] == old_val + 0.125
        assert abs(m2.Cathode.get_total_charge() - float(np.sum(qc2) + 0.125)) < 1e-12


def test_hybrid_water_and_standalone_sapt(MM_orig_mod, MM_opt_mod, sapt_orig_mod, sapt_opt_mod, tmpdir):
    print('\n======================================================================')
    print('7. Hybrid Water (HOH) + Electrode & Standalone sapt_exclusions Parity')
    print('======================================================================')
    swm4_xml = '''<ForceField>
 <AtomTypes>
  <Type name="swm4ndp-O" class="OH" element="O" mass="15.5994"/>
  <Type name="swm4ndp-H" class="HO" element="H" mass="1.008"/>
  <Type name="swm4ndp-M" class="Sh" mass="0.0"/>
  <Type name="swm4ndp-DO" class="Sh" mass="0.4"/>
 </AtomTypes>
 <Residues>
  <Residue name="HOH">
   <Atom name="O" type="swm4ndp-O"/>
   <Atom name="H1" type="swm4ndp-H"/>
   <Atom name="H2" type="swm4ndp-H"/>
   <Atom name="M" type="swm4ndp-M"/>
   <Atom name="DO" type="swm4ndp-DO"/>
   <VirtualSite type="average3" index="3" atom1="0" atom2="1" atom3="2" weight1="0.589781071" weight2="0.2051094645" weight3="0.2051094645"/>
   <Bond from="0" to="1"/>
   <Bond from="0" to="2"/>
  </Residue>
 </Residues>
 <HarmonicBondForce>
  <Bond class1="OH" class2="HO" length="0.09572" k="462750.4"/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="HO" class2="OH" class3="HO" angle="1.824218134" k="836.8"/>
 </HarmonicAngleForce>
 <NonbondedForce coulomb14scale="0" lj14scale="0">
  <Atom type="swm4ndp-O" charge="1.71636" sigma="0.318395" epsilon="0.88257"/>
  <Atom type="swm4ndp-H" charge="0.55733" sigma="1.0" epsilon="0.0"/>
  <Atom type="swm4ndp-M" charge="-1.11466" sigma="1.0" epsilon="0.0"/>
  <Atom type="swm4ndp-DO" charge="-1.71636" sigma="1.0" epsilon="0.0"/>
 </NonbondedForce>
 <DrudeForce>
  <Particle type1="swm4ndp-DO" type2="swm4ndp-O" charge="-1.71636" polarizability="0.00097822" thole="1"/>
 </DrudeForce>
</ForceField>
'''
    swm4_path = os.path.join(tmpdir, 'swm4.xml')
    with open(swm4_path, 'w') as f:
        f.write(swm4_xml)

    ffdir = os.path.join(ORIG_DIR, 'ffdir') + '/'
    res_xml = [ffdir + 'sapt_residues.xml', ffdir + 'graph_residue_c.xml', ffdir + 'graph_residue_n.xml']
    ff_xml  = [ffdir + 'sapt_noDB_2sheets.xml', ffdir + 'graph_c_freeze.xml', ffdir + 'graph_n_freeze.xml', swm4_path]
    for r in res_xml:
        Topology().loadBondDefinitions(r)

    pdb = PDBFile(os.path.join(ORIG_DIR, 'nvt_0V_15ns.pdb'))
    top = Topology()
    top.setPeriodicBoxVectors(pdb.topology.getPeriodicBoxVectors())
    pos = []
    old_pos = pdb.positions.value_in_unit(nanometer)
    keep_chains = list(pdb.topology.chains())[:4]
    bmim_res = [r for r in pdb.topology.residues() if r.name == 'BMIM'][:3]
    tf2n_res = [r for r in pdb.topology.residues() if r.name == 'Tf2N'][:3]

    for c in keep_chains:
        nc = top.addChain()
        for r in c.residues():
            nr = top.addResidue(r.name, nc)
            for a in r.atoms():
                top.addAtom(a.name, a.element, nr)
                pos.append(Vec3(*old_pos[a.index]))
    nc_el = top.addChain()
    for r in bmim_res + tf2n_res:
        nr = top.addResidue(r.name, nc_el)
        for a in r.atoms():
            top.addAtom(a.name, a.element, nr)
            pos.append(Vec3(*old_pos[a.index]))
    for c_pt in [np.array([1.5, 1.5, 1.55]), np.array([1.85, 1.5, 1.58]), np.array([2.2, 2.2, 2.5])]:
        nr = top.addResidue('HOH', nc_el)
        top.addAtom('O', Element.getBySymbol('O'), nr)
        pos.append(Vec3(*c_pt))
        top.addAtom('H1', Element.getBySymbol('H'), nr)
        pos.append(Vec3(*(c_pt + np.array([0.0757, 0.0586, 0.0]))))
        top.addAtom('H2', Element.getBySymbol('H'), nr)
        pos.append(Vec3(*(c_pt + np.array([-0.0757, 0.0586, 0.0]))))
    top.createStandardBonds()
    hoh_pdb = os.path.join(tmpdir, 'hoh.pdb')
    with open(hoh_pdb, 'w') as f:
        PDBFile.writeFile(top, pos * nanometer, f)

    def build_hoh(MM_cls):
        m = MM_cls(pdb_list=[hoh_pdb], residue_xml_list=res_xml, ff_xml_list=ff_xml)
        m.set_periodic_residue(True)
        m.platform = Platform.getPlatformByName('CUDA')
        m.properties = {'Precision': 'double', 'DeterministicForces': 'true'}
        m.simmd = Simulation(m.modeller.topology, m.system, m.integrator, m.platform, m.properties)
        m.simmd.context.setPositions(m.modeller.positions)
        m.initialize_electrodes(1.0, cathode_identifier=(0, 2), anode_identifier=(1, 3), chain=True, exclude_element=('H',))
        m.initialize_electrolyte(Natom_cutoff=100)
        m.generate_exclusions(flag_SAPT_FF_exclusions=True)
        return m

    m1 = build_hoh(MM_orig_mod.MM)
    m2 = build_hoh(MM_opt_mod.MM)
    for j in range(m1.system.getNumForces()):
        e1 = m1.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        e2 = m2.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        F1 = m1.simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
        F2 = m2.simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
        assert abs(e1 - e2) < 1e-7 and float(np.max(np.abs(F1 - F2))) < 1e-6
    print('  [Hybrid Water HOH] All force groups energy & per-atom forces match (groups=3 vs 2)!')

    # Also test standalone sapt_generate_exclusions without prior sim.context.setPositions
    mod_only_el = Modeller(top, pos * nanometer)
    mod_only_el.delete([r for r in mod_only_el.topology.residues() if r.chain.index < 4])
    ff = ForceField(*ff_xml)
    mod_only_el.addExtraParticles(ff)
    sys1 = ff.createSystem(mod_only_el.topology, nonbondedCutoff=1.4*nanometer)
    sys2 = ff.createSystem(mod_only_el.topology, nonbondedCutoff=1.4*nanometer)
    sim1 = Simulation(mod_only_el.topology, sys1, VerletIntegrator(0.001), Platform.getPlatformByName('Reference'))
    sim2 = Simulation(mod_only_el.topology, sys2, VerletIntegrator(0.001), Platform.getPlatformByName('Reference'))
    # Do NOT call sim.context.setPositions before sapt_generate_exclusions
    sapt_orig_mod.sapt_generate_exclusions(sim1, sys1, mod_only_el.positions)
    sapt_opt_mod.sapt_generate_exclusions(sim2, sys2, mod_only_el.positions)
    pe1 = sim1.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    pe2 = sim2.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
    print(f'  [Standalone sapt_generate_exclusions] dPE={abs(pe1 - pe2):.3e} kJ/mol')
    assert abs(pe1 - pe2) < 1e-9


def main():
    MM_orig_mod, _, _, sapt_orig_mod = load_modules(ORIG_DIR)
    MM_opt_mod,  _, _, sapt_opt_mod  = load_modules(OPT_DIR)

    print('======================================================================')
    print('1. Initialization & Force-Group Energy/Force Parity (CUDA double)')
    print('======================================================================')
    m_orig, t_init_orig, t_ex_orig = build_system(MM_orig_mod.MM, ORIG_DIR, voltage=1.0, precision='double')
    m_opt,  t_init_opt,  t_ex_opt  = build_system(MM_opt_mod.MM,  OPT_DIR,  voltage=1.0, precision='double')

    print(f'Original  init={t_init_orig:.3f}s (generate_exclusions={t_ex_orig:.3f}s) | CustomNB excl={m_orig.customNonbondedForce.getNumExclusions()}, groups={m_orig.customNonbondedForce.getNumInteractionGroups()} | NB excep={m_orig.nbondedForce.getNumExceptions()}')
    print(f'Optimized init={t_init_opt:.3f}s (generate_exclusions={t_ex_opt:.3f}s) | CustomNB excl={m_opt.customNonbondedForce.getNumExclusions()}, groups={m_opt.customNonbondedForce.getNumInteractionGroups()} | NB excep={m_opt.nbondedForce.getNumExceptions()}')
    print(f'Speedup: total_init={t_init_orig/t_init_opt:.2f}x, generate_exclusions={t_ex_orig/t_ex_opt:.2f}x')

    for j in range(m_orig.system.getNumForces()):
        f = m_orig.system.getForce(j)
        e1 = m_orig.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        e2 = m_opt.simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy().value_in_unit(kilojoule_per_mole)
        F1 = m_orig.simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
        F2 = m_opt.simmd.context.getState(getForces=True, groups=2**j).getForces(asNumpy=True).value_in_unit(kilojoule_per_mole/nanometer)
        dE = abs(e1 - e2)
        dF = float(np.max(np.abs(F1 - F2)))
        print(f'  Force {j} ({type(f).__name__:22s}): dE={dE:.3e} kJ/mol, max|dF|={dF:.3e} kJ/(mol*nm)')
        assert dE < 1e-7, f'Energy mismatch on Force {j}: {dE}'
        assert dF < 1e-6, f'Force mismatch on Force {j}: {dF}'

    print('\n======================================================================')
    print('2. Poisson Solver Parity (CUDA double) & Speedup (CUDA mixed)')
    print('======================================================================')
    for V in [0.0, 1.0, -1.5]:
        m1, _, _ = build_system(MM_orig_mod.MM, ORIG_DIR, voltage=V, precision='double')
        m2, _, _ = build_system(MM_opt_mod.MM,  OPT_DIR,  voltage=V, precision='double')
        m1.Poisson_solver_fixed_voltage(Niterations=4)
        m2.Poisson_solver_fixed_voltage(Niterations=4)
        qc1 = np.array([a.charge for a in m1.Cathode.electrode_atoms])
        qc2 = np.array([a.charge for a in m2.Cathode.electrode_atoms])
        qa1 = np.array([a.charge for a in m1.Anode.electrode_atoms])
        qa2 = np.array([a.charge for a in m2.Anode.electrode_atoms])
        dqc = float(np.max(np.abs(qc1 - qc2)))
        dqa = float(np.max(np.abs(qa1 - qa2)))
        dQ  = abs(m1.Cathode.Q_analytic - m2.Cathode.Q_analytic)
        print(f'  [FP64 V={V:4.1f}V] max|dq_cath|={dqc:.3e}, max|dq_anod|={dqa:.3e}, |dQ_analytic|={dQ:.3e}')
        assert dqc < 1e-7 and dqa < 1e-7 and dQ < 1e-11

    m1_mix, _, _ = build_system(MM_orig_mod.MM, ORIG_DIR, voltage=1.0, precision='mixed')
    m2_mix, _, _ = build_system(MM_opt_mod.MM,  OPT_DIR,  voltage=1.0, precision='mixed')
    m1_mix.Poisson_solver_fixed_voltage(Niterations=4)
    m2_mix.Poisson_solver_fixed_voltage(Niterations=4)
    t0 = time.perf_counter()
    for _ in range(10):
        m1_mix.Poisson_solver_fixed_voltage(Niterations=4)
    t_p_orig = (time.perf_counter() - t0) / 10.0 * 1000.0
    t0 = time.perf_counter()
    for _ in range(10):
        m2_mix.Poisson_solver_fixed_voltage(Niterations=4)
    t_p_opt = (time.perf_counter() - t0) / 10.0 * 1000.0
    print(f'  [CUDA mixed] Poisson solver (Niterations=4): orig={t_p_orig:.2f} ms, opt={t_p_opt:.2f} ms ({t_p_orig/t_p_opt:.2f}x speedup)')

    print('\n======================================================================')
    print('3. Multi-Step Constant-V MD Trajectory Parity (CUDA double, 5x50 steps)')
    print('======================================================================')
    m1, _, _ = build_system(MM_orig_mod.MM, ORIG_DIR, voltage=1.0, precision='double', seed=1234)
    m2, _, _ = build_system(MM_opt_mod.MM,  OPT_DIR,  voltage=1.0, precision='double', seed=1234)
    for _ in range(5):
        m1.Poisson_solver_fixed_voltage(Niterations=4)
        m2.Poisson_solver_fixed_voltage(Niterations=4)
        m1.simmd.step(50)
        m2.simmd.step(50)
    st1 = m1.simmd.context.getState(getEnergy=True, getPositions=True)
    st2 = m2.simmd.context.getState(getEnergy=True, getPositions=True)
    dPE = abs(st1.getPotentialEnergy().value_in_unit(kilojoule_per_mole) - st2.getPotentialEnergy().value_in_unit(kilojoule_per_mole))
    dPos = float(np.max(np.abs(st1.getPositions(asNumpy=True).value_in_unit(nanometer) - st2.getPositions(asNumpy=True).value_in_unit(nanometer))))
    print(f'  After 250 steps: dPE={dPE:.3e} kJ/mol, max|dPos|={dPos:.3e} nm')
    assert dPE < 1e-4 and dPos < 1e-7

    test_mc_barostat(MM_orig_mod, MM_opt_mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        test_vacuum_electrodes(MM_orig_mod, MM_opt_mod, tmpdir)
        test_buckyball_and_nanotube(MM_orig_mod, MM_opt_mod, tmpdir)
        test_hybrid_water_and_standalone_sapt(MM_orig_mod, MM_opt_mod, sapt_orig_mod, sapt_opt_mod, tmpdir)

    print('\nALL PARITY ASSERTIONS PASSED!')


if __name__ == '__main__':
    main()
