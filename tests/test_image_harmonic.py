"""Independent generated units, Bloch and lead-embedding controls."""
import json
import h5py
import numpy as np
import pytest
from negfpy.anharmonic import image_harmonic as h
UNIT_OMEGA = np.sqrt(h.EV_J/h.ANGSTROM_M**2/h.AMU_KG)
CELL = np.diag([1.,2.,3.])

def chain(strengths=(1.,), **kwargs):
    images={(0,0,0):2*sum(strengths)*np.eye(3)}
    for distance,k in enumerate(strengths,1):
        images[distance,0,0]=-k*np.eye(3)
        images[-distance,0,0]=-k*np.eye(3)
    return h.ImageHarmonics(images,[1.],CELL,omega0_rad_s=UNIT_OMEGA,**kwargs)


def diatomic():
    spring0,spring1=1.3,.7
    onsite=np.kron([[spring0+spring1,-spring0],[-spring0,spring0+spring1]],np.eye(3))
    forward=np.zeros((6,6));forward[3:,:3]=-spring1*np.eye(3)
    return h.ImageHarmonics({(0,0,0):onsite,(1,0,0):forward,(-1,0,0):forward.T},
                           [1.,2.],CELL,omega0_rad_s=UNIT_OMEGA)


def test_mass_units_and_independent_bloch_analytic_dispersion():
    model=chain((1.,.2))
    assert model.principal_cells==2
    for phase in (.0,.2,1.1,np.pi):
        expected=2*(1-np.cos(phase))+.4*(1-np.cos(2*phase))
        assert model.bulkD(phase)==pytest.approx(expected*np.eye(3),abs=1e-14)
    raw={(0,0,0):np.eye(6),(1,0,0):np.eye(6),(-1,0,0):np.eye(6)}
    masses=np.array([12.,95.95]);model=h.ImageHarmonics(raw,masses,CELL)
    factor=h.EV_J/h.ANGSTROM_M**2/h.AMU_KG/h.OMEGA0_RAD_S**2
    assert np.diag(model.image_dynamical[0,0,0])==pytest.approx(factor/np.repeat(masses,3))


def test_transverse_image_phases_are_not_minimum_image_folded():
    raw={(0,0,0):8*np.eye(3),(1,1,0):-np.eye(3),(-1,-1,0):-np.eye(3),
         (2,-1,1):-.3*np.eye(3),(-2,1,-1):-.3*np.eye(3)}
    model=h.ImageHarmonics(raw,[1.],CELL,omega0_rad_s=UNIT_OMEGA)
    k=np.array([.31,-.57,.73]);expected=sum(v*np.exp(1j*np.dot(k,image)) for image,v in raw.items())
    assert model.bulkD(*k)==pytest.approx(expected,abs=1e-14)
    point=model.at_q(k[1],k[2],3)
    blocks=model.transverse_blocks(k[1],k[2])
    assert point.dynamical[:3,6:9]==pytest.approx(blocks[2])


@pytest.mark.parametrize("strengths",[(1.,),(1.,.23),(1.,.2,.1,.05)])
def test_actual_range_and_generic_grouped_lead_couplings(strengths):
    model=chain(strengths);r=len(strengths);point=model.at_q(0,0,r+1)
    b=model.cell_ndof
    assert point.lead.d00.shape==(r*b,r*b)
    source=model.transverse_blocks(0,0);zero=np.zeros((b,b))
    for i in range(r):
        for j in range(r):
            assert point.lead.d00[i*b:(i+1)*b,j*b:(j+1)*b]==pytest.approx(source.get(j-i,zero))
            assert point.lead.d01[i*b:(i+1)*b,j*b:(j+1)*b]==pytest.approx(source.get(r+j-i,zero))
            assert point.device_to_lead_left[i*b:(i+1)*b,j*b:(j+1)*b]==pytest.approx(source.get(-r+j-i,zero))
            assert point.device_to_lead_right[i*b:(i+1)*b,j*b:(j+1)*b]==pytest.approx(source.get(r+j-i,zero))


def test_short_device_keeps_overlapping_contact_rows_and_rejects_bypass():
    model=chain((1.,.3,.1))
    with pytest.raises(ValueError,match="bypass"):
        model.at_q(0,0,2)
    point=model.at_q(0,0,3)
    assert np.array_equal(point.contact_left_indices,point.contact_right_indices)
    assert point.dynamical.shape==(9,9)
    assert point.device_to_lead_left.shape==(9,9)


def test_explicit_reciprocity_average_records_raw_defect_without_asr_repair():
    raw={(0,0,0):2*np.eye(3),(1,0,0):-np.eye(3),(-1,0,0):-1.01*np.eye(3)}
    with pytest.raises(ValueError,match="reciprocity defect"):
        h.ImageHarmonics(raw,[1.],CELL,omega0_rad_s=UNIT_OMEGA)
    model=h.ImageHarmonics(raw,[1.],CELL,omega0_rad_s=UNIT_OMEGA,reciprocity="average")
    assert model.reciprocity['raw_max_abs_ev_A2']==pytest.approx(.01)
    assert model.reciprocity['maximum_applied_correction_ev_A2']==pytest.approx(.005)
    assert model.reciprocity['asr_repair'] is False
    assert model.bulkD(0)[0,0]==pytest.approx(-.01)
    with pytest.raises(h.HarmonicInstability) as error:
        model.stability_audit([(0.,0.)],negative_tolerance_thz2=0)
    assert error.value.audit['minimum_squared_frequency_thz2']<0
    assert error.value.audit['eigenvalues_clipped'] is False


def test_missing_reciprocal_image_and_resource_excess_fail_closed():
    with pytest.raises(ValueError,match="reciprocal partner"):
        h.ImageHarmonics({(0,0,0):np.eye(3),(1,0,0):-np.eye(3)},[1.],CELL)
    with pytest.raises(ValueError,match="max_numeric_bytes"):
        chain(max_numeric_bytes=4)
    with pytest.raises(ValueError,match="max_numeric_bytes"):
        chain(max_numeric_bytes=4096).at_q(0,0,20)


def write_ifc2(path, *, complete=True):
    with h5py.File(path,'w') as out:
        out.attrs.update(schema=h.IFC2_SCHEMA,complete=complete,metadata_json=json.dumps({'pbc':[True,False,False]}))
        out['images']=[[0,0,0],[1,0,0],[-1,0,0]]
        out['blocks_ev_A2']=np.array([2*np.eye(3),-np.eye(3),-np.eye(3)])
        out['masses_amu']=[1.]
        out['cell_rows_angstrom']=CELL
        out['fractional_positions']=[[0.,0.,0.]]
        out.create_dataset('symbols',data=['Si'],dtype=h5py.string_dtype('utf-8'))


def test_hdf_load_schema_units_copy_and_metadata(tmp_path):
    path=tmp_path/'fc2.hdf5';write_ifc2(path)
    data=h.load_ifc2(path)
    assert data.symbols==('Si',) and data.metadata['pbc']==[True,False,False]
    assert len(data.source_sha256)==64
    assert data.blocks_ev_A2.flags.writeable is False
    model=h.ImageHarmonics.from_data(data)
    assert model.principal_cells==1
    with pytest.raises(ValueError,match="max_bytes"):
        h.load_ifc2(path,max_bytes=10)
    with h5py.File(path,'r+') as out:
        out.attrs['complete']=False
    with pytest.raises(ValueError,match="complete"):
        h.load_ifc2(path)
