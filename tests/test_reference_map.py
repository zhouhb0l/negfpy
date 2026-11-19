"""Tiny synthetic map equivalence and analytic feedback derivative checks."""
from pathlib import Path
import importlib.util
import unittest

import numpy as np

from negfpy.anharmonic.transverse_channel_feedback import (
    ChannelSpectrum, evaluate_channel_born_map, plan_channel_born_map,
)
from negfpp.reference.map_adapter import evaluate_reference_map, tangent_map, plan_reference_map


source = Path(__file__).with_name('test_anharmonic_transverse_channel_feedback.py')
spec = importlib.util.spec_from_file_location('canonical_synthetic_channel_fixture', source)
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


class MapAdapterTests(unittest.TestCase):
    def setUp(self):
        self.args = fixture_module._fixture(nq=2)
        self.kw = dict(workers=1, external_tile=1, bin_tile=2,
                       max_bytes=128*1024**2, max_contraction_work=10**10,
                       max_dyson_evaluations=100)

    def test_initial_exact_zero_baseline_is_canonical_map(self):
        got, diag = evaluate_reference_map(*self.args, **self.kw)
        reference = evaluate_channel_born_map(*self.args, **self.kw)
        self.assertEqual(diag['arithmetic'], 'direct_baseline')
        for group in ('green', 'map_sigma'):
            for field in ('retarded', 'statistical', 'spectral'):
                np.testing.assert_array_equal(getattr(getattr(got,group),field),
                                               getattr(getattr(reference,group),field))
        self.assertEqual(got.input_currents, reference.input_currents)

    def test_reference_increment_recovers_direct_map_and_currents(self):
        anchor = evaluate_channel_born_map(*self.args, **self.kw)
        sigma = fixture_module._sigma(self.args[0], self.args[1])
        got, diag = evaluate_reference_map(*self.args, sigma_input=sigma,
            reference_green=anchor.green, reference_mapped=anchor.map_sigma, **self.kw)
        direct = evaluate_channel_born_map(*self.args, sigma_input=sigma, **self.kw)
        for group in ('green','map_sigma'):
            for field in ('retarded','statistical','spectral'):
                np.testing.assert_allclose(getattr(getattr(got,group),field),
                    getattr(getattr(direct,group),field), rtol=2e-13, atol=3e-15)
        for field in ('left','right','collision_power'):
            np.testing.assert_allclose(getattr(got.input_currents,field),
                getattr(direct.input_currents,field), rtol=2e-13, atol=1e-18)
        self.assertEqual(diag['dyson_solves'], 20)
        self.assertEqual(set(got.checks),set(direct.checks))

    def test_nonzero_reference_and_tangent_of_complete_map(self):
        base = fixture_module._sigma(self.args[0], self.args[1])
        evaluation = evaluate_channel_born_map(*self.args,sigma_input=base,**self.kw)
        direction = ChannelSpectrum(*(getattr(base,name)*2 for name in
                                      ('retarded','statistical','spectral')))
        derivative, diag = tangent_map(*self.args,sigma_base=base,
            base_green=evaluation.green,direction=direction,**self.kw)
        self.assertFalse(diag['signed_tangent_has_physical_psd_gate'])
        errors = []
        for h in (.2,.1):
            images=[]
            for sign in (1,-1):
                current=ChannelSpectrum(*(getattr(base,name)+sign*h*getattr(direction,name)
                    for name in ('retarded','statistical','spectral')))
                got,_ = evaluate_reference_map(*self.args,sigma_input=current,
                    reference_sigma=base,reference_green=evaluation.green,
                    reference_mapped=evaluation.map_sigma,**self.kw)
                images.append(got.map_sigma)
            error=0.
            for name in ('retarded','statistical','spectral'):
                finite_difference=(getattr(images[0],name)-getattr(images[1],name))/(2*h)
                error += np.linalg.norm(finite_difference-getattr(derivative,name))**2
            errors.append(error**.5)
        self.assertGreater(errors[0]/errors[1],3.8)
        self.assertLess(errors[0]/errors[1],4.2)

    def test_explicit_double_work_and_dyson_budget(self):
        args=self.args
        n_q,n_w,d,a=args[1].mesh.n_qpoints,args[0].n_points,args[1].full_ndof,args[1].ndof
        base=plan_channel_born_map(n_q,n_w,d,a,2,2,workers=1,external_tile=1,bin_tile=2)
        current=plan_reference_map(n_q,n_w,d,a,2,2,workers=1,external_tile=1,bin_tile=2)
        self.assertEqual(current.contraction_work,2*base.contraction_work)
        self.assertEqual(current.dyson_evaluations,2*base.dyson_evaluations)
        self.assertGreater(current.peak_owned_bytes,base.peak_owned_bytes)
        anchor=evaluate_channel_born_map(*args,**self.kw)
        limited=dict(self.kw,max_dyson_evaluations=base.dyson_evaluations)
        with self.assertRaisesRegex(ValueError,'max_dyson_evaluations'):
            evaluate_reference_map(*args,reference_green=anchor.green,
                                   reference_mapped=anchor.map_sigma,**limited)


if __name__=='__main__':
    unittest.main()
