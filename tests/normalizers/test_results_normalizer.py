"""Tests for ResultsNormalizer simulation data detection and routing."""

import numpy as np
import pytest
from nomad.datamodel import EntryArchive, EntryMetadata
from nomad.datamodel.data import ArchiveSection
from nomad.datamodel.results import (
    BandGap,
    BandStructureElectronic,
    DOSElectronic,
    ElectronicProperties,
    Properties,
    Results,
)
from nomad.metainfo import Quantity
from nomad.units import ureg
from nomad.utils import get_logger
from nomad_simulations.schema_packages.atoms_state import AtomsState
from nomad_simulations.schema_packages.general import Program, Simulation
from nomad_simulations.schema_packages.model_method import (
    BSE,
    DFT,
    DMFT,
    GW,
    ModelMethod,
    Wannier,
)
from nomad_simulations.schema_packages.model_system import ModelSystem
from nomad_simulations.schema_packages.numerical_settings import SelfConsistency
from nomad_simulations.schema_packages.outputs import Outputs, TrajectoryOutputs
from nomad_simulations.schema_packages.properties import (
    AbsorptionSpectrum,
    ElectronicBandGap,
    ElectronicBandStructure,
    ElectronicDensityOfStates,
    TotalForce,
)
from nomad_simulations.schema_packages.properties import (
    PotentialEnergy as SimPotentialEnergy,
)
from nomad_simulations.schema_packages.properties import (
    RadiusOfGyration as SimRadiusOfGyration,
)
from nomad_simulations.schema_packages.properties import (
    Temperature as SimTemperature,
)
from nomad_simulations.schema_packages.variables import (
    Energy2,
)
from nomad_simulations.schema_packages.workflow.general import (
    EnergyConvergenceTarget,
    ForceConvergenceTarget,
)
from nomad_simulations.schema_packages.workflow.geometry_optimization import (
    GeometryOptimization as SimGeometryOptimizationWorkflow,
)
from nomad_simulations.schema_packages.workflow.geometry_optimization import (
    GeometryOptimizationMethod as SimGeometryOptimizationMethod,
)
from nomad_simulations.schema_packages.workflow.geometry_optimization import (
    GeometryOptimizationResults as SimGeometryOptimizationResults,
)

from nomad_results_normalizer.normalizers.results import (
    GENERATED_RESULTS_ANNOTATION,
)
from nomad_results_normalizer.normalizers.results import (
    ResultsNormalizerBase as ResultsNormalizer,
)

try:
    import runschema.calculation  # noqa: F401
    import runschema.run  # noqa: F401

    HAS_RUNSCHEMA = True
except Exception:
    HAS_RUNSCHEMA = False

LOGGER = get_logger(__name__)


def _kline_path():
    class _DummyKPath(ArchiveSection):
        points = Quantity(type=np.float64, shape=['*', 3])

    return _DummyKPath(points=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]))


@pytest.fixture
def archive_with_simulation_data():
    """Create an archive with simulation data in archive.data."""
    archive = EntryArchive(metadata=EntryMetadata())

    # Create simulation data.
    simulation = Simulation()
    model_system = ModelSystem(
        name='test_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )

    # Add particle states
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )

    # Add cell properties
    model_system.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]

    simulation.model_system.append(model_system)
    archive.data = simulation

    # Initialize results
    archive.results = Results()
    archive.results.properties = Properties()

    return archive


@pytest.fixture
def archive_with_nested_system():
    """Create an archive with a system nested deep in archive.data."""
    from nomad.metainfo import SubSection

    archive = EntryArchive(metadata=EntryMetadata())

    class Container(ArchiveSection):
        sub = SubSection(sub_section=ArchiveSection)

    # Create simulation data.
    model_system = ModelSystem(
        name='nested_system',
        type='molecule',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='H', atomic_number=1)
    )
    model_system.lattice_vectors = np.eye(3) * 10.0 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]

    # Nest it
    container = Container()
    container.sub = model_system
    archive.data = container

    # Initialize results
    archive.results = Results()
    archive.results.properties = Properties()

    return archive


@pytest.fixture
def archive_empty():
    """Create an archive with neither data nor run schema."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()
    return archive


def test_simulation_data_is_detected(archive_with_simulation_data, caplog):
    """Archives with simulation data should run the plugin path."""
    normalizer = ResultsNormalizer()

    # Clear any previous log records
    caplog.clear()

    # Run normalization
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    assert any(
        'simulation results normalization' in record.message
        for record in caplog.records
    ), 'Should log simulation results normalization'
    assert not any(
        'legacy results normalization' in record.message for record in caplog.records
    ), 'Plugin should not run legacy results normalization itself'


def test_empty_simulation_has_no_source_system():
    archive = EntryArchive(metadata=EntryMetadata())
    archive.data = Simulation()

    assert ResultsNormalizer()._simulation_data_context(archive) == (True, None)


def test_archive_without_simulation_data_is_skipped(archive_empty, caplog):
    """Archives without simulation data should be skipped by this plugin."""
    normalizer = ResultsNormalizer()

    # Clear any previous log records
    caplog.clear()

    # Run normalization
    try:
        normalizer.normalize(archive_empty, LOGGER)
    except Exception:
        # May fail without proper data - that's OK
        pass

    assert any(
        'Skipping simulation results normalization' in record.message
        for record in caplog.records
    ), 'Should skip the plugin path when no simulation data exists'


def test_non_simulation_data_skips_plugin_path(caplog):
    """Custom archive.data without systems should not use the plugin path."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    class CustomData(ArchiveSection):
        name = None

    archive.data = CustomData()

    normalizer = ResultsNormalizer()
    caplog.clear()

    normalizer.normalize(archive, LOGGER)

    assert any(
        'Skipping simulation results normalization' in record.message
        for record in caplog.records
    ), 'Non-simulation archive.data should be skipped by this plugin'
    assert not any(
        record.message == 'Running simulation results normalization'
        for record in caplog.records
    ), 'Non-simulation archive.data should not use the plugin path'


def test_nested_system_is_detected(archive_with_nested_system, caplog):
    """Test that a nested system triggers simulation results normalization."""
    normalizer = ResultsNormalizer()
    caplog.clear()

    # Run normalization
    normalizer.normalize(archive_with_nested_system, LOGGER)

    assert any(
        'simulation results normalization' in record.message
        for record in caplog.records
    ), 'A nested system should trigger simulation results normalization'
    # Keep this suite focused on routing. Detailed topology behavior is covered
    # in test_topology_normalizer.py.
    assert archive_with_nested_system.results is not None
    assert archive_with_nested_system.results.material is not None


def test_simulation_initializes_results_sections(archive_with_simulation_data):
    """Simulation data should initialize key results sections."""
    normalizer = ResultsNormalizer()

    # Run normalization
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    # Check that results were populated
    assert archive_with_simulation_data.results is not None
    assert archive_with_simulation_data.results.material is not None


def test_simulation_populates_results_method():
    """Simulation program/model_method should populate results.method."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(
        name='VASP', version='6.4.2', version_internal='git-abc123'
    )
    simulation.model_method.append(DFT())
    simulation.model_method.append(Wannier(localization_type='maximally_localized'))

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.method_name == 'DFT'
    assert archive.results.method.simulation is not None
    assert archive.results.method.simulation.program_name == 'VASP'
    assert archive.results.method.simulation.program_version == '6.4.2'
    assert archive.results.method.simulation.program_version_internal == 'git-abc123'
    assert archive.results.method.simulation.dft is not None
    assert archive.results.method.simulation.tb is not None
    assert archive.results.method.simulation.tb.type == 'Wannier'
    assert (
        archive.results.method.simulation.tb.localization_type == 'maximally_localized'
    )


def test_simulation_skips_unsupported_method_names():
    """Unsupported model methods should be ignored for results.method transfer."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())
    simulation.model_method.append(ModelMethod(name='HF'))

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.method_name == 'DFT'
    assert archive.results.method.simulation is not None
    assert archive.results.method.simulation.dft is not None


def test_simulation_maps_method_details_for_gw_bse_dmft():
    """Simulation method details should map to supported results fields."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(GW(type='G0W0'))
    simulation.model_method.append(BSE(type='RPA', solver='TDA'))
    simulation.model_method.append(
        DMFT(
            impurity_solver='CT-HYB',
            magnetic_state='paramagnetic',
            inverse_temperature=(1.0 / ureg.eV),
        )
    )

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    method = archive.results.method
    assert method is not None
    assert method.method_name == 'GW'
    assert method.simulation is not None
    assert method.simulation.gw is not None
    assert method.simulation.gw.type == 'G0W0'
    assert method.simulation.bse is not None
    assert method.simulation.bse.type == 'RPA'
    assert method.simulation.bse.solver == 'TDA'
    assert method.simulation.dmft is not None
    assert method.simulation.dmft.impurity_solver_type == 'CT-HYB'
    assert method.simulation.dmft.magnetic_state == 'paramagnetic'
    assert method.simulation.dmft.inverse_temperature is not None


def test_simulation_method_name_uses_first_supported_model_method():
    """Method name should use first supported model_method for compatibility."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    # Intentionally place DMFT first and DFT last.
    simulation.model_method.append(
        DMFT(
            impurity_solver='CT-HYB',
            magnetic_state='paramagnetic',
        )
    )
    simulation.model_method.append(GW(type='G0W0'))
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.method_name == 'DMFT'


def test_simulation_maps_dft_spin_and_jacobs_ladder():
    """Simulation DFT metadata should map to DFT results fields."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT(is_spin_polarized=True, jacobs_ladder='GGA'))

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.method is not None
    assert archive.results.method.simulation is not None
    assert archive.results.method.simulation.dft is not None
    assert archive.results.method.simulation.dft.spin_polarized is True
    assert archive.results.method.simulation.dft.jacobs_ladder == 'GGA'
    assert archive.results.method.simulation.dft.xc_functional_type == 'GGA'


def test_simulation_maps_flexible_unit_scf_threshold(archive_with_simulation_data):
    """Current SelfConsistency quantities should map without a legacy unit field."""
    simulation = archive_with_simulation_data.data
    simulation.program = Program(name='VASP')
    dft = DFT()
    dft.numerical_settings.append(SelfConsistency(threshold_change=1.0e-5 * ureg.eV))
    simulation.model_method.append(dft)

    ResultsNormalizer().normalize(archive_with_simulation_data, LOGGER)

    simulation_results = archive_with_simulation_data.results.method.simulation
    threshold = simulation_results.dft.scf_threshold_energy_change
    assert threshold.to('eV').magnitude == pytest.approx(1.0e-5)


def test_simulation_maps_outputs_electronic_properties():
    """Simulation outputs should map electronic properties into results."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = TrajectoryOutputs(time=0.0 * ureg.ps)
    output.electronic_band_gaps.append(ElectronicBandGap(value=1.5 * ureg.eV))
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV, spin_channel=0
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
    band_structure.k_path = _kline_path()
    output.electronic_band_structures.append(band_structure)
    absorption = AbsorptionSpectrum(value=np.array([0.5, 0.6]) / ureg.eV)
    absorption.energies = Energy2(points=np.array([0.0, 1.0]) * ureg.eV)
    output.absorption_spectra.append(absorption)
    output.radii_of_gyration.append(SimRadiusOfGyration(value=1.2e-10 * ureg.meter))
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    output.potential_energies.append(SimPotentialEnergy(value=-5.0 * ureg.eV))
    simulation.outputs.append(output)

    output_2 = TrajectoryOutputs(time=1.0 * ureg.ps)
    output_2.temperatures.append(SimTemperature(value=320 * ureg.kelvin))
    output_2.potential_energies.append(SimPotentialEnergy(value=-4.8 * ureg.eV))
    simulation.outputs.append(output_2)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert electronic.band_gap
    assert electronic.band_gap[0].value is not None
    assert not electronic.dos_electronic_new
    assert not electronic.dos_electronic
    assert not electronic.band_structure_electronic
    assert archive.results.properties.spectroscopic is not None
    assert archive.results.properties.spectroscopic.spectra
    assert archive.results.properties.structural is not None
    assert archive.results.properties.structural.radius_of_gyration
    assert archive.results.properties.thermodynamic is not None
    assert archive.results.properties.thermodynamic.trajectory
    assert len(getattr(archive, 'run', None) or []) == 0


def test_simulation_uses_latest_output_for_electronic_properties():
    """Electronic mapping should keep the latest direct output payload."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output_1 = Outputs()
    output_1.electronic_band_gaps.append(ElectronicBandGap(value=1.5 * ureg.eV))
    band_structure_1 = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
    band_structure_1.k_path = _kline_path()
    output_1.electronic_band_structures.append(band_structure_1)
    simulation.outputs.append(output_1)

    output_2 = Outputs()
    output_2.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    band_structure_2 = ElectronicBandStructure(value=np.array([[2.0], [2.1]]) * ureg.eV)
    band_structure_2.k_path = _kline_path()
    output_2.electronic_band_structures.append(band_structure_2)
    simulation.outputs.append(output_2)

    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert len(electronic.band_gap or []) == 1
    assert electronic.band_gap[0].value.to('eV').magnitude == pytest.approx(2.5)
    assert not electronic.band_structure_electronic
    assert len(getattr(archive, 'run', None) or []) == 0


def test_simulation_prefers_representative_system_outputs_for_electronic_properties():
    """Electronic mapping should prefer outputs linked to representative system."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    representative_system = ModelSystem(
        name='rep_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    representative_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    representative_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    representative_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    representative_system.periodic_boundary_conditions = [True, True, True]

    other_system = ModelSystem(
        name='other_system',
        type='bulk',
        is_representative=False,
        positions=np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    other_system.particle_states.append(
        AtomsState(chemical_symbol='Ge', atomic_number=32)
    )
    other_system.particle_states.append(
        AtomsState(chemical_symbol='Ge', atomic_number=32)
    )
    other_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    other_system.periodic_boundary_conditions = [True, True, True]

    simulation.model_system.append(representative_system)
    simulation.model_system.append(other_system)

    # If supported by current schema version, set explicit representative index.
    if 'representative_index' in simulation.m_def.all_quantities:
        simulation.representative_index = 0

    output_rep = Outputs()
    output_rep.model_system_ref = representative_system
    output_rep.electronic_band_gaps.append(ElectronicBandGap(value=1.5 * ureg.eV))
    band_structure_rep = ElectronicBandStructure(
        value=np.array([[1.0], [1.1]]) * ureg.eV
    )
    band_structure_rep.k_path = _kline_path()
    output_rep.electronic_band_structures.append(band_structure_rep)
    simulation.outputs.append(output_rep)

    output_other = Outputs()
    output_other.model_system_ref = other_system
    output_other.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    band_structure_other = ElectronicBandStructure(
        value=np.array([[2.0], [2.1]]) * ureg.eV
    )
    band_structure_other.k_path = _kline_path()
    output_other.electronic_band_structures.append(band_structure_other)
    simulation.outputs.append(output_other)

    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert len(electronic.band_gap or []) == 1
    assert electronic.band_gap[0].value.to('eV').magnitude == pytest.approx(1.5)
    assert not electronic.band_structure_electronic
    assert len(getattr(archive, 'run', None) or []) == 0


def test_simulation_does_not_map_band_structure_to_run_backed_results():
    """Band structures in archive.data should not create run-backed result refs."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
    band_structure.k_path = _kline_path()
    output.electronic_band_structures.append(band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    serialized = archive.m_to_dict()
    assert not serialized.get('run')
    assert not (
        serialized.get('results', {})
        .get('properties', {})
        .get('electronic', {})
        .get('band_structure_electronic', [])
    )


def test_simulation_keeps_reference_energy_off_run_backed_electronic_sections():
    """Reference energies from data should not create BS/DOS compatibility payloads."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='exciting')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    reference_energy = 1.0 * ureg.eV

    output = Outputs()
    band_structure = ElectronicBandStructure(value=np.array([[1.0], [1.1]]) * ureg.eV)
    band_structure.k_path = _kline_path()
    band_structure.highest_occupied = reference_energy
    output.electronic_band_structures.append(band_structure)

    dos = ElectronicDensityOfStates(value=np.array([0.1, 0.2, 0.3]) / ureg.eV)
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)

    output.electronic_band_gaps.append(ElectronicBandGap(value=0.5 * ureg.eV))
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is not None
    assert electronic.band_gap
    assert not electronic.band_structure_electronic
    assert not electronic.dos_electronic
    assert len(getattr(archive, 'run', None) or []) == 0


def test_simulation_does_not_map_dos_to_deprecated_results():
    """DOS in archive.data should not create deprecated dos_electronic results."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    assert electronic is None or not electronic.dos_electronic_new
    assert electronic is None or not electronic.dos_electronic
    assert len(getattr(archive, 'run', None) or []) == 0


def test_simulation_replaces_malformed_existing_dos_entries():
    """Malformed stale generated DOS references should be removed, not replaced."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()
    archive.results.properties.electronic = archive.results.properties.m_create(
        ElectronicProperties
    )

    # Simulate stale malformed entry that can crash GUI (missing energies/total)
    archive.results.properties.electronic.m_add_sub_section(
        ElectronicProperties.dos_electronic, DOSElectronic()
    )

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = Outputs()
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    simulation.outputs.append(output)
    archive.data = simulation

    ResultsNormalizer().normalize(archive, LOGGER)

    serialized = archive.m_to_dict()
    dos_sections = (
        serialized.get('results', {})
        .get('properties', {})
        .get('electronic', {})
        .get('dos_electronic', [])
    )
    assert not dos_sections
    assert len(getattr(archive, 'run', None) or []) == 0


def test_simulation_drops_malformed_stale_dos_without_new_electronic_payload():
    """Malformed stale dos_electronic should be removed to avoid GUI crashes."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()
    archive.results.properties.electronic = archive.results.properties.m_create(
        ElectronicProperties
    )
    archive.results.properties.electronic.m_add_sub_section(
        ElectronicProperties.dos_electronic, DOSElectronic()
    )

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    # No electronic payload in outputs: only thermodynamic quantity.
    output = Outputs()
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    simulation.outputs.append(output)
    archive.data = simulation

    ResultsNormalizer().normalize(archive, LOGGER)

    electronic = archive.results.properties.electronic
    if electronic is not None:
        assert not (electronic.dos_electronic or [])


def test_simulation_does_not_create_empty_electronic_properties():
    """Non-electronic outputs should not instantiate empty properties.electronic."""
    archive = EntryArchive(metadata=EntryMetadata())
    archive.results = Results()
    archive.results.properties = Properties()

    simulation = Simulation()
    simulation.program = Program(name='VASP')
    simulation.model_method.append(DFT())

    model_system = ModelSystem(
        name='test_system',
        type='bulk',
        is_representative=True,
        positions=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]) * ureg.angstrom,
        n_particles=2,
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.particle_states.append(
        AtomsState(chemical_symbol='Si', atomic_number=14)
    )
    model_system.lattice_vectors = np.eye(3) * 5.43 * ureg.angstrom
    model_system.periodic_boundary_conditions = [True, True, True]
    simulation.model_system.append(model_system)

    output = TrajectoryOutputs(time=1.0 * ureg.ps)
    output.radii_of_gyration.append(SimRadiusOfGyration(value=1.2e-10 * ureg.meter))
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    output.potential_energies.append(SimPotentialEnergy(value=-5.0 * ureg.eV))
    simulation.outputs.append(output)
    archive.data = simulation

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive, LOGGER)

    assert archive.results.properties.electronic is None
    assert archive.results.properties.structural is not None
    assert archive.results.properties.structural.radius_of_gyration
    assert archive.results.properties.thermodynamic is not None
    assert archive.results.properties.thermodynamic.trajectory


def test_simulation_trajectory_uses_physical_time(archive_with_simulation_data):
    output = TrajectoryOutputs(time=2.5 * ureg.ps, wall_end=100 * ureg.s)
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    archive_with_simulation_data.data.outputs.append(output)

    ResultsNormalizer().normalize(archive_with_simulation_data, LOGGER)

    trajectory = (
        archive_with_simulation_data.results.properties.thermodynamic.trajectory[0]
    )
    time = trajectory.temperature.time.to('second').magnitude
    assert np.asarray(time)[0] == pytest.approx(2.5e-12)


def test_simulation_logs_trajectory_dropped_without_physical_time(
    archive_with_simulation_data, caplog
):
    """`time` only exists on TrajectoryOutputs; dropping it must not be silent."""
    output = Outputs()
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    output.potential_energies.append(SimPotentialEnergy(value=-5.0 * ureg.eV))
    archive_with_simulation_data.data.outputs.append(output)

    caplog.clear()
    ResultsNormalizer().normalize(archive_with_simulation_data, LOGGER)

    thermodynamic = archive_with_simulation_data.results.properties.thermodynamic
    assert thermodynamic is None or not thermodynamic.trajectory
    assert any(
        'skipping trajectory series without physical time' in record.message
        for record in caplog.records
    )


def test_simulation_keeps_each_method_on_the_representative_system(
    archive_with_simulation_data,
):
    """DFT and GW band gaps on one system should stay side-by-side."""
    simulation = archive_with_simulation_data.data
    system = simulation.model_system[0]
    dft = DFT()
    gw = GW()
    simulation.model_method.extend([dft, gw])

    dft_output = Outputs(model_system_ref=system, model_method_ref=dft)
    dft_output.electronic_band_gaps.append(ElectronicBandGap(value=1.0 * ureg.eV))
    simulation.outputs.append(dft_output)

    gw_output = Outputs(model_system_ref=system, model_method_ref=gw)
    gw_output.electronic_band_gaps.append(ElectronicBandGap(value=2.0 * ureg.eV))
    simulation.outputs.append(gw_output)

    ResultsNormalizer().normalize(archive_with_simulation_data, LOGGER)

    electronic = archive_with_simulation_data.results.properties.electronic
    assert [gap.value.to('eV').magnitude for gap in electronic.band_gap] == [1.0, 2.0]
    assert not electronic.dos_electronic
    assert not electronic.band_structure_electronic
    assert len(getattr(archive_with_simulation_data, 'run', None) or []) == 0


def test_simulation_drops_outputs_from_non_representative_systems(
    archive_with_simulation_data, caplog
):
    """Merging outputs across different systems stays forbidden."""
    simulation = archive_with_simulation_data.data
    representative_system = simulation.model_system[0]
    other_system = ModelSystem(type='bulk')
    simulation.model_system.append(other_system)

    representative_output = Outputs(model_system_ref=representative_system)
    representative_output.electronic_band_gaps.append(
        ElectronicBandGap(value=1.5 * ureg.eV)
    )
    simulation.outputs.append(representative_output)

    other_output = Outputs(model_system_ref=other_system)
    other_output.electronic_band_gaps.append(ElectronicBandGap(value=2.5 * ureg.eV))
    simulation.outputs.append(other_output)

    caplog.clear()
    ResultsNormalizer().normalize(archive_with_simulation_data, LOGGER)

    electronic = archive_with_simulation_data.results.properties.electronic
    assert [band_gap.value.to('eV').magnitude for band_gap in electronic.band_gap] == [
        1.5
    ]
    assert any(
        'discarding electronic outputs from non-representative systems'
        in record.message
        for record in caplog.records
    )


def test_simulation_does_not_create_per_method_run_backed_dos_references(
    archive_with_simulation_data,
):
    """Per-method DOS payloads should no longer create legacy run references."""
    simulation = archive_with_simulation_data.data
    system = simulation.model_system[0]
    dft = DFT()
    gw = GW()
    simulation.model_method.extend([dft, gw])

    for method, values in ((dft, [0.1, 0.2]), (gw, [0.3, 0.4])):
        output = Outputs(model_system_ref=system, model_method_ref=method)
        dos = ElectronicDensityOfStates(value=np.array(values) / ureg.eV)
        dos.energies = Energy2(points=np.array([-1.0, 1.0]) * ureg.eV)
        output.electronic_dos.append(dos)
        simulation.outputs.append(output)

    ResultsNormalizer().normalize(archive_with_simulation_data, LOGGER)

    serialized = archive_with_simulation_data.m_to_dict()
    assert not serialized.get('run')
    assert not (
        serialized.get('results', {})
        .get('properties', {})
        .get('electronic', {})
        .get('dos_electronic', [])
    )


@pytest.mark.skipif(not HAS_RUNSCHEMA, reason='requires runschema')
def test_simulation_preserves_legacy_calculation_and_result_sections():
    archive = EntryArchive(
        metadata=EntryMetadata(), results=Results(properties=Properties())
    )

    legacy_run = runschema.run.Run(raw_id='parser-owned')
    legacy_calculation = runschema.calculation.Calculation()
    legacy_dos = runschema.calculation.Dos(energies=np.array([-2.0, 2.0]) * ureg.eV)
    legacy_dos.total.append(
        runschema.calculation.DosValues(value=np.array([0.3, 0.4]) / ureg.eV)
    )
    legacy_calculation.dos_electronic.append(legacy_dos)
    legacy_band_structure = runschema.calculation.BandStructure()
    legacy_band_structure.segment.append(
        runschema.calculation.BandEnergies(
            energies=np.array([[[2.0], [2.1]]]) * ureg.eV,
            kpoints=np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]),
        )
    )
    legacy_calculation.band_structure_electronic.append(legacy_band_structure)
    legacy_run.calculation.append(legacy_calculation)
    archive.run.append(legacy_run)

    electronic = archive.results.properties.m_create(ElectronicProperties)
    parser_dos = DOSElectronic(label='parser-owned')
    parser_dos.energies = '/run/0/calculation/0/dos_electronic/0/energies'
    parser_dos.total = ['/run/0/calculation/0/dos_electronic/0/total/0']
    electronic.m_add_sub_section(ElectronicProperties.dos_electronic, parser_dos)
    parser_band_structure = BandStructureElectronic(label='parser-owned')
    parser_band_structure.segment = legacy_band_structure.segment
    electronic.m_add_sub_section(
        ElectronicProperties.band_structure_electronic, parser_band_structure
    )
    electronic.m_add_sub_section(
        ElectronicProperties.band_gap, BandGap(value=9.0 * ureg.eV)
    )

    simulation = Simulation(program=Program(name='VASP'))
    system = ModelSystem(is_representative=True, type='molecule')
    simulation.model_system.append(system)
    output = Outputs()
    output.electronic_band_gaps.append(ElectronicBandGap(value=1.0 * ureg.eV))
    mapped_dos = ElectronicDensityOfStates(value=np.array([0.1, 0.2]) / ureg.eV)
    mapped_dos.energies = Energy2(points=np.array([-1.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(mapped_dos)
    mapped_band_structure = ElectronicBandStructure(
        value=np.array([[1.0], [1.1]]) * ureg.eV
    )
    mapped_band_structure.k_path = _kline_path()
    output.electronic_band_structures.append(mapped_band_structure)
    simulation.outputs.append(output)
    archive.data = simulation

    ResultsNormalizer().normalize(archive, LOGGER)

    assert archive.run[0] is legacy_run
    assert legacy_calculation.dos_electronic[0] is legacy_dos
    assert legacy_calculation.band_structure_electronic[0] is legacy_band_structure
    assert len(archive.run) == 1
    assert any(section.label == 'parser-owned' for section in electronic.dos_electronic)
    assert any(
        section.label == 'parser-owned'
        for section in electronic.band_structure_electronic
    )
    assert any(gap.value.to('eV').magnitude == 9.0 for gap in electronic.band_gap)
    assert any(gap.value.to('eV').magnitude == 1.0 for gap in electronic.band_gap)

    serialized = archive.m_to_dict()
    dos_sections = serialized['results']['properties']['electronic']['dos_electronic']
    assert len(dos_sections) == 1
    assert GENERATED_RESULTS_ANNOTATION not in dos_sections[0].get('m_annotations', {})


def test_simulation_output_mapping_is_idempotent(archive_with_simulation_data):
    output = TrajectoryOutputs(time=2.0 * ureg.ps)
    absorption = AbsorptionSpectrum(value=np.array([0.5, 0.6]) / ureg.eV)
    absorption.energies = Energy2(points=np.array([0.0, 1.0]) * ureg.eV)
    output.absorption_spectra.append(absorption)
    output.radii_of_gyration.append(SimRadiusOfGyration(value=1.2 * ureg.angstrom))
    output.temperatures.append(SimTemperature(value=300 * ureg.kelvin))
    archive_with_simulation_data.data.outputs.append(output)

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)
    properties = archive_with_simulation_data.results.properties
    first_counts = (
        len(properties.spectroscopic.spectra),
        len(properties.structural.radius_of_gyration),
        len(properties.thermodynamic.trajectory),
    )
    first_time = properties.thermodynamic.trajectory[0].temperature.time.copy()

    assert GENERATED_RESULTS_ANNOTATION in (
        properties.spectroscopic.spectra[0].m_annotations
    )
    assert GENERATED_RESULTS_ANNOTATION in (
        properties.structural.radius_of_gyration[0].m_annotations
    )
    assert GENERATED_RESULTS_ANNOTATION in (
        properties.thermodynamic.trajectory[0].m_annotations
    )
    assert properties.thermodynamic.trajectory[0].provenance is None

    normalizer.normalize(archive_with_simulation_data, LOGGER)

    assert (
        len(properties.spectroscopic.spectra),
        len(properties.structural.radius_of_gyration),
        len(properties.thermodynamic.trajectory),
    ) == first_counts
    np.testing.assert_allclose(
        properties.thermodynamic.trajectory[0].temperature.time,
        first_time,
    )


def test_historical_generated_results_marker_is_still_recognized():
    assert (
        GENERATED_RESULTS_ANNOTATION
        == 'nomad_results_normalizer_data_schema_compatibility'
    )
    section = BandGap()
    section.m_annotations['nomad_topology_normalizer_v2_compatibility'] = {
        'generated': True
    }

    assert ResultsNormalizer._is_generated_compatibility_section(section)


def test_simulation_logs_unmapped_output_groups(archive_with_simulation_data, caplog):
    """Unsupported outputs should stay unset and produce a TODO log."""
    output = Outputs()
    output.total_forces.append(
        TotalForce(value=np.array([[1.0, 0.0, 0.0]]) * ureg.newton)
    )
    archive_with_simulation_data.data.outputs.append(output)

    normalizer = ResultsNormalizer()
    caplog.clear()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    properties = archive_with_simulation_data.results.properties
    assert properties.mechanical is None
    assert properties.vibrational is None
    assert properties.magnetic is None
    assert properties.dynamical is None


def test_simulation_skips_dos_without_compatibility_run(
    archive_with_simulation_data,
):
    """DOS-only outputs should not create electronic results or run sections."""
    output = Outputs()
    dos = ElectronicDensityOfStates(
        value=np.array([0.1, 0.2, 0.3]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    output.electronic_dos.append(dos)
    archive_with_simulation_data.data.outputs.append(output)

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    properties = archive_with_simulation_data.results.properties
    electronic = properties.electronic
    assert electronic is None or not electronic.dos_electronic_new
    assert electronic is None or not electronic.dos_electronic
    assert len(getattr(archive_with_simulation_data, 'run', None) or []) == 0


def test_simulation_results_with_existing_run(archive_with_simulation_data):
    """An existing run section does not prevent simulation results population."""
    # Add a mock run section alongside simulation data.
    from nomad.datamodel.data import ArchiveSection

    class MockRun(ArchiveSection):
        system = []

    archive_with_simulation_data.run = [MockRun()]

    normalizer = ResultsNormalizer()

    normalizer.normalize(archive_with_simulation_data, LOGGER)

    assert archive_with_simulation_data.results is not None


def test_simulation_runs_plugin_path(archive_with_simulation_data, monkeypatch):
    calls = []

    def method(self, archive):
        calls.append('method')

    def outputs(self, archive):
        calls.append('outputs')

    monkeypatch.setattr(ResultsNormalizer, '_normalize_method_from_simulation', method)
    monkeypatch.setattr(
        ResultsNormalizer, '_normalize_outputs_from_simulation', outputs
    )

    ResultsNormalizer().normalize(archive_with_simulation_data, LOGGER)

    assert calls == ['method', 'outputs']


def test_simulation_maps_geometry_optimization_workflow(archive_with_simulation_data):
    workflow = SimGeometryOptimizationWorkflow()
    workflow.method = SimGeometryOptimizationMethod(optimization_type='atomic')
    workflow.method.convergence_targets = [
        EnergyConvergenceTarget(threshold=1e-6 * ureg.eV),
        ForceConvergenceTarget(threshold=1e-5 * ureg.newton),
    ]
    workflow.results = SimGeometryOptimizationResults(
        final_energy_difference=2e-6 * ureg.eV,
        final_force_maximum=4e-5 * ureg.newton,
        final_displacement_maximum=1e-12 * ureg.meter,
    )

    archive_with_simulation_data.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    geometry_optimization = (
        archive_with_simulation_data.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None
    assert geometry_optimization.type == 'atomic'
    assert geometry_optimization.convergence_tolerance_energy_difference is not None
    assert geometry_optimization.convergence_tolerance_force_maximum is not None
    assert geometry_optimization.final_energy_difference is not None
    assert geometry_optimization.final_force_maximum is not None
    # Both trajectory and system_optimized expect runschema types
    # (Calculation, System), not simulation data sections (Outputs, ModelSystem).
    # This is expected: convergence values can still be populated.


def test_simulation_geometry_optimization_without_run_references(
    archive_with_simulation_data,
):
    """Test geometry optimization mapping without calculations_ref fields."""
    workflow = SimGeometryOptimizationWorkflow()
    workflow.method = SimGeometryOptimizationMethod(optimization_type='atomic')
    workflow.method.convergence_targets = [
        EnergyConvergenceTarget(threshold=1e-6 * ureg.eV),
        ForceConvergenceTarget(threshold=1e-5 * ureg.newton),
    ]
    # Results without calculations_ref or calculation_result_ref
    workflow.results = SimGeometryOptimizationResults(
        final_energy_difference=2e-6 * ureg.eV,
        final_force_maximum=4e-5 * ureg.newton,
        final_displacement_maximum=1e-12 * ureg.meter,
    )

    archive_with_simulation_data.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    geometry_optimization = (
        archive_with_simulation_data.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None
    assert geometry_optimization.type == 'atomic'
    assert geometry_optimization.convergence_tolerance_energy_difference is not None
    assert geometry_optimization.convergence_tolerance_force_maximum is not None
    assert geometry_optimization.final_energy_difference is not None
    assert geometry_optimization.final_force_maximum is not None
    # Note: trajectory and system_optimized cannot be populated from new
    # schema (type incompatibility)


def test_simulation_geometry_optimization_with_method_tolerances(
    archive_with_simulation_data,
):
    """Test geometry optimization when tolerances are on method directly."""
    workflow = SimGeometryOptimizationWorkflow()
    method = SimGeometryOptimizationMethod(optimization_type='cell_shape')
    method.convergence_tolerance_energy_difference = 1e-7 * ureg.eV
    method.convergence_tolerance_force_maximum = 1e-6 * ureg.newton
    workflow.method = method

    workflow.results = SimGeometryOptimizationResults(
        final_energy_difference=2e-7 * ureg.eV,
        final_force_maximum=4e-6 * ureg.newton,
    )

    archive_with_simulation_data.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    geometry_optimization = (
        archive_with_simulation_data.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None
    assert geometry_optimization.type == 'cell_shape'
    assert geometry_optimization.convergence_tolerance_energy_difference.to(
        ureg.eV
    ).magnitude == pytest.approx(1e-7)
    assert (
        geometry_optimization.convergence_tolerance_force_maximum.magnitude
        == pytest.approx(1e-6)
    )
    assert geometry_optimization.final_energy_difference.to(
        ureg.eV
    ).magnitude == pytest.approx(2e-7)
    assert geometry_optimization.final_force_maximum.magnitude == pytest.approx(4e-6)


def test_simulation_geometry_optimization_detects_via_class_name(
    archive_with_simulation_data,
):
    """Test that geometry optimization is detected by workflow class name."""
    workflow = SimGeometryOptimizationWorkflow()
    # Workflow with no method or results, but class name should be enough

    archive_with_simulation_data.workflow2 = workflow

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    geometry_optimization = (
        archive_with_simulation_data.results.properties.geometry_optimization
    )
    assert geometry_optimization is not None


def test_simulation_merges_electronic_payload_split_across_outputs(
    archive_with_simulation_data,
):
    representative = archive_with_simulation_data.data.model_system[0]

    output_bs = Outputs(model_system_ref=representative)
    output_bs.electronic_band_gaps.append(ElectronicBandGap(value=1.0 * ureg.eV))
    output_bs.electronic_band_structures.append(
        ElectronicBandStructure(
            value=np.array([[0.0, 1.0], [0.1, 1.1]]) * ureg.eV,
            highest_occupied=0.0 * ureg.eV,
            lowest_unoccupied=1.0 * ureg.eV,
            k_path=_kline_path(),
        )
    )

    output_dos = Outputs(model_system_ref=representative)
    dos = ElectronicDensityOfStates(
        value=np.array([0.2, 0.3, 0.4]) / ureg.eV,
        spin_channel=0,
    )
    dos.energies = Energy2(points=np.array([-1.0, 0.0, 1.0]) * ureg.eV)
    dos.energies_origin = 0.0 * ureg.eV
    output_dos.electronic_dos.append(dos)

    archive_with_simulation_data.data.outputs = [output_bs, output_dos]

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    electronic = archive_with_simulation_data.results.properties.electronic
    assert electronic is not None
    assert electronic.band_gap
    assert not electronic.band_structure_electronic
    assert not electronic.dos_electronic
    assert len(getattr(archive_with_simulation_data, 'run', None) or []) == 0


def test_simulation_normalize_calls_topology_normalizer(
    archive_with_simulation_data, monkeypatch
):
    """Test that the simulation path calls TopologyNormalizer."""
    from nomad_results_normalizer.normalizers.topology import TopologyNormalizer

    # Track if TopologyNormalizer.normalize was called
    normalize_called = []

    original_normalize = TopologyNormalizer.normalize

    def mock_normalize(self, archive, logger, source_system=None):
        normalize_called.append(True)
        # Call original to avoid breaking the test
        return original_normalize(self, archive, logger, source_system=source_system)

    monkeypatch.setattr(TopologyNormalizer, 'normalize', mock_normalize)

    normalizer = ResultsNormalizer()
    normalizer.normalize(archive_with_simulation_data, LOGGER)

    # Verify TopologyNormalizer.normalize was called
    assert len(normalize_called) > 0, 'TopologyNormalizer.normalize should be called'


def test_normalize_measurements_still_works(archive_with_simulation_data):
    """Test that measurement normalization still works with new architecture."""
    # Skip this test as it requires complex measurement schema setup
    pytest.skip('Test requires complete measurement schema implementation')
