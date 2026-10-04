import importlib
import io
import json
import pickle
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.preprocessing import StandardScaler

from backend.pipeline import CLASSES, alarm_series, infer, load_bundle, read_csv, restricted_joblib, tensorflow, training_windows, windows


@pytest.fixture(scope="session", params=[8, 7], ids=["eight-class", "seven-class-v2"])
def package(tmp_path_factory, request):
    path = tmp_path_factory.mktemp("fixture-model")
    tf = tensorflow()
    tf.keras.utils.set_random_seed(42)
    for filename, units, activation in [("lstm_leak_classifier.keras", 1, "sigmoid"), ("lstm_leak_location.keras", request.param, "softmax")]:
        net = tf.keras.Sequential([tf.keras.layers.Input((3, 2)), tf.keras.layers.LSTM(3), tf.keras.layers.Dense(units, activation=activation)])
        net.save(path / filename)
    joblib.dump(StandardScaler().fit([[0., 0.], [1., 2.], [2., 4.]]), path / "scaler.pkl")
    joblib.dump({"feature_cols": ["P", "TAVG"], "time_steps": 3, "stride": 2}, path / "config.pkl")
    (path / "manifest.json").write_text(json.dumps({"id":"fixture", "name":"Test-only tiny LSTM", "classes":CLASSES if request.param == 8 else list(reversed(CLASSES[1:])), "sample_interval":1, "threshold":.7, "persistence":3}))
    return path


def csv_bytes(seed=1, count=16):
    rng=np.random.default_rng(seed)
    return pd.DataFrame({"TIME":np.arange(count), "P":rng.normal(size=count), "TAVG":rng.normal(size=count)}).to_csv(index=False).encode()


@pytest.fixture
def client(tmp_path, monkeypatch):
    module=importlib.import_module("backend.app")
    for key in ("MODELS", "DATA"):
        path=tmp_path/key.lower();path.mkdir();monkeypatch.setattr(module,key,path)
    monkeypatch.setattr(module,"CACHE",{})
    return TestClient(module.app)


def import_fixture(client, package):
    names=json.loads((package/"manifest.json").read_text())["classes"]
    files=[("files", ("lstm_leak_location_v2.keras" if p.name=="lstm_leak_location.keras" and len(names)==7 else p.name,p.read_bytes())) for p in package.iterdir() if p.suffix!=".json"]
    response=client.post("/api/models/import",files=files,data={"name":"Fixture baseline", "sample_interval":"1", "class_order_confirmed":"true", "location_classes":json.dumps(names)})
    assert response.status_code==200,response.text
    return response.json()


def test_model_bundle_and_causal_windows(package):
    bundle=load_bundle(package)
    df=read_csv(csv_bytes(),bundle["config"],1)
    x,times=windows(df,bundle)
    assert times.tolist()==[2,4,6,8,10,12,14]
    full=infer(df,bundle,"full")
    prefix=infer(df.iloc[:9],bundle,"prefix")
    assert np.allclose([p["score"] for p in full["predictions"][:4]],[p["score"] for p in prefix["predictions"]])
    assert x.shape==(7,3,2)


def test_alarm_requires_consecutive_windows_and_latches():
    locations=np.zeros((7,8));locations[:,3]=1
    pred,events=alarm_series([11,16,21,26,31,36,41],[.8,.1,.8,.9,.8,.1,.2],locations,.7,3)
    assert [p["hits"] for p in pred]==[1,0,1,2,3,0,0]
    assert [e["time"] for e in events if e["kind"]=="alarm"]==[31]
    assert pred[-1]["alarm"] and pred[-1]["disagreement"]


def test_rejects_invalid_sensor_schema_and_time(package):
    cfg=load_bundle(package)["config"]
    for raw,message in [(b"TIME,Dose\n0,1\n1,2\n2,3\n","Missing required sensors"),(b"TIME,P,TAVG\n0,1,2\n1,2,3\n1,3,4\n","increase strictly"),(b"TIME,P,TAVG\n0,1,2\n2,2,3\n4,3,4\n","Sampling interval")]:
        with pytest.raises(ValueError,match=message):read_csv(raw,cfg,1)


def test_pickle_rejects_arbitrary_code(tmp_path):
    class Untrusted:
        def __reduce__(self):return (eval,("1+1",))
    path=tmp_path/"bad.pkl";path.write_bytes(pickle.dumps(Untrusted()))
    with pytest.raises(ValueError,match="Unsupported object"):restricted_joblib(path)


def test_import_activate_export_and_reload(client,package):
    model=import_fixture(client,package)
    assert client.get('/api/models').json()['active'] is None
    assert client.post(f"/api/models/{model['id']}/activate").status_code==200
    module=importlib.import_module('backend.app');module.CACHE.clear()
    assert module.registry()['active']==model['id']
    assert module.get_bundle(model['id'])['meta']['name']=='Fixture baseline'
    result=client.get(f"/api/models/{model['id']}/export")
    with zipfile.ZipFile(io.BytesIO(result.content)) as archive:assert set(archive.namelist())==module.REQUIRED|{'manifest.json'}
    imported=client.post('/api/models/import',files={'files':('package.zip',result.content)},data={'name':'Updated model','sample_interval':'999','class_order_confirmed':'true'})
    assert imported.status_code==200,imported.text
    assert imported.json()['sample_interval']==1
    assert client.get('/api/models').json()['active']==model['id']


def test_incomplete_import_is_actionable(client,package):
    response=client.post('/api/models/import',files={'files':('lstm_leak_classifier.keras',(package/'lstm_leak_classifier.keras').read_bytes())},data={'name':'Incomplete','sample_interval':'1','class_order_confirmed':'true'})
    assert response.status_code==422
    assert 'scaler.pkl' in response.json()['detail']
    assert client.get('/api/models').json()['versions']==[]


def test_rejects_fractional_alarm_persistence(client,package):
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as archive:
        for p in package.iterdir():
            if p.name!='manifest.json':archive.writestr(p.name,p.read_bytes())
        archive.writestr('manifest.json',json.dumps({'classes':CLASSES,'sample_interval':1,'threshold':.7,'persistence':2.5}))
    response=client.post('/api/models/import',files={'files':('model.zip',stream.getvalue())},data={'name':'Bad policy','sample_interval':1,'class_order_confirmed':'true'})
    assert response.status_code==422 and 'alarm threshold or persistence' in response.json()['detail']


def test_fine_tuning_preserves_baseline_and_split_isolation(client,package):
    model=import_fixture(client,package)
    client.post(f"/api/models/{model['id']}/activate")
    module=importlib.import_module('backend.app')
    before=(module.MODELS/model['id']/'lstm_leak_classifier.keras').read_bytes()
    rows=[]
    for n,split in enumerate(['train','train','validation','validation','test','test']):
        response=client.post('/api/recordings',files={'file':(f'run-{n}.csv',csv_bytes(n+10))},data={'model_id':model['id']})
        assert response.status_code==200,response.text
        row=response.json()
        rows.append({'id':row['id'],'group':f'group-{n}','split':split,'label':'No leak' if n%2==0 else 'SGATR','origin':'existing' if n==0 else 'new'})
    payload={'model_id':model['id'],'name':'Test candidate','rows':rows,'epochs':1,'independent_holdout_confirmed':True}
    broken=json.loads(json.dumps(payload));broken['rows'][2]['group']='group-0'
    rejection=client.post('/api/training',json=broken)
    assert rejection.status_code==422 and 'crosses data splits' in rejection.json()['detail']
    duplicate=json.loads(json.dumps(payload));duplicate['rows'][2]['id']=rows[0]['id']
    rejection=client.post('/api/training',json=duplicate)
    assert rejection.status_code==422 and 'Duplicate sensor traces' in rejection.json()['detail']
    response=client.post('/api/training',json=payload)
    assert response.status_code==200,response.text
    job=response.json()
    deadline=time.monotonic()+90
    while job['status'] in ('queued','running') and time.monotonic()<deadline:
        time.sleep(.2);job=client.get(f"/api/training/{job['id']}").json()
    assert job['status']=='complete',job
    assert client.get('/api/training').json()[0]['id']==job['id']
    assert job['evaluation']['candidate']['recordings']==2
    assert job['evaluation']['candidate']['median_delay_seconds'] is None
    assert client.get('/api/models').json()['active']==model['id']
    assert (module.MODELS/model['id']/'lstm_leak_classifier.keras').read_bytes()==before
    candidate=module.get_bundle(job['model_id'])
    assert candidate['meta']['parent_id']==model['id']
    assert candidate['meta']['training']['scaler_refitted'] is False
    assert candidate['meta']['training']['locator_training_clip_limit']==(6 if len(model['classes'])==7 else None)
    replay=client.get(f"/api/recordings/{rows[0]['id']}/replay",params={'model_id':job['model_id']})
    assert replay.status_code==200,replay.text
    assert replay.json()['illustrative'] is False
    assert candidate['meta']['classes']==model['classes']
    if len(model['classes'])==7:
        assert job['evaluation']['candidate']['location_evaluation_scope']=='Detected leak recordings, first-three-clip vote'
        assert job['evaluation']['candidate']['location_total'] in (0,1)
        assert replay.json()['location_conditional'] is True
        assert 'No leak' not in replay.json()['predictions'][0]['locations']


def test_seven_class_location_order_is_explicit_and_detector_gated():
    order = list(reversed(CLASSES[1:]))
    locations = np.zeros((4, 7)); locations[:, order.index('SGATR')] = 1
    pred, events = alarm_series([2,4,6,8], [.1,.8,.8,.1], locations, .7, 2, order)
    assert [p['component'] for p in pred] == ['No leak','SGATR','SGATR','No leak']
    assert all(p['location_component']=='SGATR' for p in pred)
    assert not any(p['disagreement'] for p in pred)
    assert 'No leak' not in pred[0]['locations']
    assert pred[-1]['alarm'] and events[-1]['component']=='SGATR'


def test_version_suffix_import_rejects_duplicate_roles(client, package):
    files=[('files', (p.name,p.read_bytes())) for p in package.iterdir() if p.suffix!='.json']
    files.append(('files', ('lstm_leak_location_v2.keras', (package/'lstm_leak_location.keras').read_bytes())))
    response=client.post('/api/models/import', files=files, data={'name':'Ambiguous','sample_interval':1,'class_order_confirmed':'true'})
    assert response.status_code==422 and 'Duplicate package role' in response.json()['detail']
    assert client.get('/api/models').json()['versions']==[]


def test_import_rejects_locator_width_mismatch(client, package):
    names=json.loads((package/'manifest.json').read_text())['classes']
    wrong=CLASSES[1:] if len(names)==8 else CLASSES
    files=[('files', (p.name,p.read_bytes())) for p in package.iterdir() if p.suffix!='.json']
    response=client.post('/api/models/import',files=files,data={'name':'Wrong contract','sample_interval':1,'class_order_confirmed':'true','location_classes':json.dumps(wrong)})
    assert response.status_code==422 and 'selected locator format' in response.json()['detail']
    assert client.get('/api/models').json()['versions']==[]


def test_v2_locator_runs_only_after_alarm_and_votes_first_recording_clips(package):
    if len(json.loads((package/'manifest.json').read_text())['classes']) != 7:
        return
    bundle=load_bundle(package); order=bundle['meta']['classes']
    df=read_csv(csv_bytes(),bundle['config'],1)
    class Detector:
        def __init__(self, alarm): self.alarm=alarm
        def __call__(self,x,training=False):
            scores=np.full((len(x),1),.1,dtype=np.float32)
            if self.alarm: scores[3:6]=.9
            return scores
    class Locator:
        def __init__(self): self.calls=[]
        def __call__(self,x,training=False):
            self.calls.append(np.asarray(x).copy())
            result=np.zeros((len(x),7),dtype=np.float32)
            for i,c in enumerate(['SGATR','LOCA','SGATR']): result[i,order.index(c)]=1
            return result
    locator=Locator();bundle['locator']=locator;bundle['detector']=Detector(False)
    quiet=infer(df,bundle,'quiet')
    assert locator.calls==[] and quiet['localization'] is None
    assert all(p['locations']=={} for p in quiet['predictions'])
    bundle['detector']=Detector(True)
    result=infer(df,bundle,'alarm')
    expected_x,_=windows(df,bundle)
    assert len(locator.calls)==1 and np.allclose(locator.calls[0],expected_x[:3])
    assert result['localization']['clip_times']==[2,4,6]
    assert result['localization']['available_at']==12
    assert result['localization']['votes']==['SGATR','LOCA','SGATR']
    assert all(p['locations']=={} for p in result['predictions'] if p['time']<12)
    assert all(p['component']=='SGATR' for p in result['predictions'] if p['time']>=12)
    assert result['predictions'][-1]['score']<.7 and result['predictions'][-1]['alarm']


def test_v3_location_tie_matches_pandas_string_mode(package):
    bundle=load_bundle(package);order=bundle['meta']['classes']
    if len(order)!=7: return
    class Detector:
        def __call__(self,x,training=False): return np.full((len(x),1),.9,dtype=np.float32)
    class Locator:
        def __call__(self,x,training=False):
            result=np.zeros((3,7),dtype=np.float32)
            result[0,order.index('FLB')]=.51;result[0,order.index('LOCA')]=.49
            result[1,order.index('LOCA')]=1
            result[2,order.index('SGATR')]=.8;result[2,order.index('LOCA')]=.2
            return result
    bundle['detector']=Detector();bundle['locator']=Locator()
    result=infer(read_csv(csv_bytes(),bundle['config'],1),bundle,'tie')
    votes=result['localization']['votes']
    assert votes==['FLB','LOCA','SGATR']
    assert result['localization']['tied'] is True
    assert result['predictions'][-1]['component']==pd.Series(votes).mode()[0]=='FLB'
    assert max(result['predictions'][-1]['locations'],key=result['predictions'][-1]['locations'].get)=='LOCA'


def test_v3_training_uses_first_six_recording_clips_with_onset_filter(package):
    bundle=load_bundle(package)
    df=read_csv(csv_bytes(),bundle['config'],1)
    x,labels,eligible=training_windows(df,bundle,{'label':'SGATR','onset':9})
    assert len(x)==7 and labels.tolist()==[0,0,0,0,3,3,3]
    conditional=len(bundle['meta']['classes'])==7
    assert eligible.tolist()==([False,False,False,False,True,True,False] if conditional else [True]*7)
    _,_,eligible=training_windows(df,bundle,{'label':'No leak','onset':None})
    assert eligible.tolist()==([False]*7 if conditional else [True]*7)
    _,_,eligible=training_windows(df,bundle,{'label':'SGATR','onset':None})
    assert eligible.tolist()==([True]*6+[False] if conditional else [True]*7)
