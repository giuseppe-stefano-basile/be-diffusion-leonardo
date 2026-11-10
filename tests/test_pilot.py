import csv
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from be_pilot.common import REPO, DT, TYPES, read, sha, cn, read_data, write_data, dump_frames, seed
from be_pilot import engine

CFG=read(REPO/'campaign.json')
PARENT=CFG['parents'][0]


def output(attempt,duration,corrupt=False):
    xyz,box=read_data(REPO/PARENT['seed_file'])
    steps=round(duration/DT)
    with (attempt/'trajectory.lammpstrj').open('w') as f:
        for step in range(0,steps+1,10):
            f.write(f'ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n376\nITEM: BOX BOUNDS pp pp pp\n')
            for lo,hi in box:f.write(f'{lo:.17g} {hi:.17g}\n')
            f.write('ITEM: ATOMS id type x y z\n')
            for i,(t,v) in enumerate(zip(TYPES,xyz),1):f.write(f'{i} {t} {v[0]:.17g} {v[1]:.17g} {v[2]:.17g}\n')
    with (attempt/'COLVAR').open('w') as f:
        f.write('#! FIELDS time cn25 cn27\n')
        for step in range(0,steps+1,10):f.write(f'{step*DT:.16g} {cn(xyz,box,2.5):.16g} {cn(xyz,box,2.7)+(1 if corrupt else 0):.16g}\n')
    write_data(attempt/'final.data',xyz,box)
    (attempt/'final.restart').write_bytes(b'fake restart: tests only')
    (attempt/'log.lammps').write_text('Step Time Temp Press Volume Density TotEng PotEng Fmax\n0 0 298 0 1000 1 -1 -2 0.1\nLoop time of 1 on 1 procs\n')


class GeometryTests(unittest.TestCase):
    def test_all_exported_parents_and_lineage(self):
        self.assertEqual(len(CFG['parents']),60)
        self.assertEqual(len({p['source_group'] for p in CFG['parents']}),60)
        for p in CFG['parents']:
            self.assertEqual(sha(REPO/p['seed_file']),p['seed_sha256'])
            xyz,box=read_data(REPO/p['seed_file'])
            self.assertAlmostEqual(cn(xyz,box,2.5),p['seed_cn25'],delta=2e-8)
            self.assertAlmostEqual(cn(xyz,box,2.7),p['seed_cn27'],delta=6e-6)
    def test_production_unbiased_dual_cv(self):
        text=engine.plumed()
        self.assertNotIn('WALLS',text);self.assertNotIn('METAD',text)
        self.assertIn('cn25',text);self.assertIn('cn27',text)
        self.assertIn('STRIDE=10',text)
        self.assertIn('KAPPA=20000',engine.plumed(PARENT))
    def test_random_seeds_unique(self):
        values=[]
        for p in CFG['parents']:
            for role in (['primary','control'] if p['control'] else ['primary']):
                for r in range(4):
                    values.extend(seed(p['parent_id'],f'{role}_rep{r}',kind) for kind in ['velocity','thermostat'])
        self.assertEqual(len(values),528);self.assertEqual(len(values),len(set(values)))
    def test_atom_reordering_rejected(self):
        xyz,box=read_data(REPO/PARENT['seed_file'])
        from be_pilot.common import check_geometry
        with self.assertRaises(ValueError):check_geometry(TYPES[::-1],xyz,box)
    def test_periodic_translation(self):
        xyz,box=read_data(REPO/PARENT['seed_file']);shift=box[:,1]-box[:,0]
        other=xyz.copy();other[1::3]+=shift
        self.assertAlmostEqual(cn(xyz,box,2.5),cn(other,box,2.5),places=12)


class StageTests(unittest.TestCase):
    def setUp(self):
        engine.STOP.clear();self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.model=self.root/'model.pt';self.model.write_bytes(b'fake test model')
        self.runtime=dict(model=str(self.model),model_sha256=sha(self.model),binary='/not-executed',binary_sha256='test',fingerprint='test',box=CFG['box_bounds_A'])
    def tearDown(self): self.temp.cleanup();engine.STOP.clear()
    def fake(self,*args,**kwargs):
        output(Path(kwargs['cwd']),.01)
        class P:
            pid=12345678
            def wait(self):return 0
        return P()
    def stage(self):return engine.stage(self.root/'stage',REPO/PARENT['seed_file'],.01,PARENT,'FAKE',self.runtime)
    def test_completed_stage_is_not_overwritten(self):
        with patch.object(engine.subprocess,'Popen',side_effect=self.fake) as popen:
            first,_=self.stage();again,_=self.stage()
            self.assertEqual(first,again);self.assertEqual(popen.call_count,1)
    def test_corrupted_completed_output_rejected(self):
        with patch.object(engine.subprocess,'Popen',side_effect=self.fake):attempt,_=self.stage()
        with (attempt/'COLVAR').open('a') as f:f.write('corruption\n')
        with self.assertRaisesRegex(ValueError,'Completed output changed'):self.stage()
    def test_changed_initial_input_rejected(self):
        with patch.object(engine.subprocess,'Popen',side_effect=self.fake):self.stage()
        changed=self.root/'new.data';changed.write_text('different')
        with self.assertRaisesRegex(ValueError,'input changed'):
            engine.stage(self.root/'stage',changed,.01,PARENT,'FAKE',self.runtime)
    def test_interrupted_attempt_preserved_on_retry(self):
        def interrupted(*args,**kwargs):
            class P:
                pid=12345678
                def wait(self):return 143
            return P()
        with patch.object(engine.subprocess,'Popen',side_effect=interrupted):
            with self.assertRaises(RuntimeError):self.stage()
        with patch.object(engine.subprocess,'Popen',side_effect=self.fake):attempt,_=self.stage()
        self.assertEqual(attempt.name,'attempt_001')
        self.assertTrue((self.root/'stage/attempt_000/FAILED.json').exists())
    def test_incomplete_colvar_rejected(self):
        attempt=self.root/'out';attempt.mkdir();(attempt/'plumed.dat').write_text(engine.plumed());output(attempt,.01)
        (attempt/'COLVAR').write_text('#! FIELDS time cn25 cn27\n0 4 5\n')
        with self.assertRaisesRegex(ValueError,'timeline'):engine.validate(attempt,.01,CFG['box_bounds_A'])
    def test_coordinate_cv_disagreement_rejected(self):
        attempt=self.root/'out';attempt.mkdir();(attempt/'plumed.dat').write_text(engine.plumed());output(attempt,.01,True)
        with self.assertRaisesRegex(ValueError,'disagreement'):engine.validate(attempt,.01,CFG['box_bounds_A'])
    def test_restraints_rejected_in_production(self):
        attempt=self.root/'out';attempt.mkdir();(attempt/'plumed.dat').write_text(engine.plumed(PARENT))
        with self.assertRaisesRegex(ValueError,'Biased'):engine.validate(attempt,.01,CFG['box_bounds_A'])
    def test_stop_prevents_launch(self):
        engine.STOP.set()
        with patch.object(engine.subprocess,'Popen') as p:
            with self.assertRaises(InterruptedError):self.stage()
            p.assert_not_called()


class CLITests(unittest.TestCase):
    def test_smoke_requests_four_gpus_and_uses_external_environment(self):
        import sys
        from types import SimpleNamespace
        from be_pilot.__main__ import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);inputs=root/'inputs';inputs.mkdir();(inputs/'SHA256SUMS.json').write_text('{}')
            env=root/'my runtime.env';env.write_text('# fake runtime for scheduler test')
            args=['be_pilot','smoke','--inputs',str(inputs),'--run-dir',str(root/'run'),'--runtime-env',str(env)]
            with patch.object(sys,'argv',args),patch('be_pilot.runtime.input_checks',return_value={}),patch('subprocess.run',return_value=SimpleNamespace(stdout='12345\n')) as submit:
                main()
            cmd=submit.call_args.args[0]
            self.assertIn('--gres=gpu:4',cmd);self.assertIn('--account=INF26_biophys',cmd);self.assertIn('--qos=boost_qos_dbg',cmd)
            script=next((root/'run').glob('job_smoke_*.sh')).read_text()
            self.assertIn('my runtime.env',script);self.assertNotIn('simone-mace',script)
    def test_production_requires_smoke_qualification(self):
        import sys
        from be_pilot.__main__ import main
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);env=root/'runtime.env';env.write_text('# test')
            args=['be_pilot','submit','--inputs',str(root),'--run-dir',str(root/'run'),'--runtime-env',str(env)]
            with patch.object(sys,'argv',args),patch('be_pilot.runtime.input_checks',return_value={}),patch('subprocess.run') as submit:
                with self.assertRaisesRegex(RuntimeError,'QUALIFIED'):main()
                submit.assert_not_called()


class CountingTests(unittest.TestCase):
    def test_original_frame_phase_and_nonoverlap(self):
        from be_pilot.analysis import count_one
        a=np.c_[np.arange(401)*.005,np.full(401,4.)]
        cfg=dict(edges=np.linspace(2.2,5.7,41),frame_dt_ps=.1,lag_ps=.1,discard_ps=.1)
        c,n,_=count_one(a,cfg)
        self.assertEqual(n,19);self.assertEqual(c.sum(),19)
        cfg.update(frame_dt_ps=.005,lag_ps=.05)
        _,n,_=count_one(a,cfg);self.assertEqual(n,38)
    def test_fine_samples_do_not_inflate_r25_counts(self):
        from be_pilot.analysis import count_one
        cfg=dict(edges=np.linspace(2.2,5.7,41),frame_dt_ps=.1,lag_ps=.1,discard_ps=.1)
        fine=np.c_[np.arange(401)*.005,np.full(401,4.)]
        self.assertTrue(np.array_equal(count_one(fine,cfg)[0],count_one(fine[::20],cfg)[0]))


class PreparationTests(unittest.TestCase):
    def test_no_eligible_frame_stops_after_twelve_ps(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls=[];p=dict(PARENT,lo=100.,hi=101.)
            xyz,box=read_data(REPO/PARENT['seed_file'])
            def fake(folder,initial,duration,*args,**kwargs):
                folder=Path(folder);a=folder/'attempt_000';a.mkdir(parents=True);(a/'final.restart').write_bytes(b'test')
                calls.append(duration);return a,{}
            with patch.object(engine,'stage',side_effect=fake),patch.object(engine,'dump_frames',side_effect=lambda path: iter([(16000,xyz,box)])):
                initial,meta=engine.prepare(root,p,'FAKE',{},8)
            self.assertIsNone(initial);self.assertEqual(calls,[8.,2.,2.])
            self.assertTrue((root/'preparations'/p['parent_id']/'hold_8/QUARANTINE.json').exists())
    def test_control_continues_same_parent_to_sixteen_ps(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);calls=[];xyz,box=read_data(REPO/PARENT['seed_file']);v=cn(xyz,box,2.5)
            p=dict(PARENT,lo=v-.01,hi=v+.01,cv='R25')
            last_duration=[0.]
            def fake(folder,initial,duration,*args,**kwargs):
                from be_pilot.common import atomic
                folder=Path(folder);a=folder/'attempt_000';a.mkdir(parents=True);(a/'final.restart').write_bytes(b'test')
                atomic(folder/'COMPLETE.json',dict(attempt=a.name,input_key='test',outputs={}))
                calls.append((duration,kwargs.get('label')));last_duration[0]=duration
                return a,{}
            def frames(path):return iter([(round(last_duration[0]/DT),xyz,box)])
            with patch.object(engine,'stage',side_effect=fake),patch.object(engine,'dump_frames',side_effect=frames):
                initial,meta=engine.prepare(root,p,'FAKE',{},16)
            self.assertEqual([x[0] for x in calls],[8.,8.]);self.assertEqual(meta['actual_preparation_ps'],16.)
            self.assertEqual(meta['source_group'],p['source_group']);self.assertTrue(initial.exists())


@unittest.skipUnless(os.environ.get('BE_TEST_INPUTS'),'Set BE_TEST_INPUTS for real frozen-input regression')
class FrozenRegressionTests(unittest.TestCase):
    def test_all_eight_native_baseline_rates(self):
        from be_pilot.kinetics import baseline_regression
        self.assertEqual(len(baseline_regression(Path(os.environ['BE_TEST_INPUTS'])/'baseline')),8)
    def test_checked_arithmetic_and_sampler_on_synthetic_counts(self):
        from be_pilot.common import REPO
        import sys
        sys.path.insert(0,str(REPO/'vendor'))
        import hummer_high_precision as HP
        import hummer_bounded_bayesian as B
        counts=np.array([[30,2,0,0],[3,25,2,0],[0,3,20,2],[0,0,1,35]])
        g=np.array([.1,.2,.3]);k=np.array([1.,1.2,.8])
        q=HP.evaluate(g,k,counts,.1,precision='quad')
        mp=HP.evaluate(g,k,counts,.1,precision='mpfr')
        self.assertAlmostEqual(q,mp,places=9)
        result=B.sample(counts,np.arange(4.)+.5,.1,g,k,10.,1000.,3,steps=80,burn=20,thin=5,seed=123)
        self.assertEqual(result['D'].shape,(12,3));self.assertTrue(np.all(result['D']>0))
        self.assertTrue(np.isfinite(result['log_likelihood']).all())
    def test_completed_cpu_chain_is_verified_and_reused(self):
        from be_pilot.analysis import sample_chain,npz_atomic
        from be_pilot.common import atomic
        import hummer_bounded_bayesian as B
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            atomic(folder/'configuration.json',dict(anchor_bin=3,edges=[0.,1.,2.,3.,4.],lag_ps=.1))
            npz_atomic(folder/'counts.npz',counts=np.array([[30,2,0,0],[3,25,2,0],[0,3,20,2],[0,0,1,35]]))
            task=('R25','bins32',0,str(folder),'unused',80,20,5)
            sample_chain(task)
            with patch.object(B,'sample') as sampler:
                sample_chain(task);sampler.assert_not_called()
            (folder/'chain_00.npz').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError,'Completed MCMC chain changed'):sample_chain(task)
    def test_both_original_transition_matrices(self):
        from be_pilot.analysis import count_dataset
        for cv in ['R25','R27']:
            cfg,counts,*_=count_dataset(Path(os.environ['BE_TEST_INPUTS'])/'baseline',cv,[],'primary')
            self.assertGreater(counts.sum(),0)
    def test_full_input_integrity(self):
        from be_pilot.runtime import input_checks
        self.assertEqual(input_checks(os.environ['BE_TEST_INPUTS'])['primary_shots'],240)

if __name__=='__main__':unittest.main()
