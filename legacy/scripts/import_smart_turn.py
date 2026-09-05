#!/usr/bin/env python
"""Recover trainable weights from the pinned public FP32 ONNX; enforce numerical parity.

No executable remote model code is loaded. MatMul transposes are recovered using the
exporter's module-name metadata; every state tensor must be present and shape compatible.
"""
import ast, json
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from eot.model import EOTConfig, EOTModel, save_checkpoint
from eot.data import sha256_file

REVISION='f766f81d3cfdf7737ac64aad813d91bbfd56bf93'

def main():
    torch.set_num_threads(2)
    p=Path(hf_hub_download('pipecat-ai/smart-turn-v3','smart-turn-v3.2-gpu.onnx',revision=REVISION))
    graph=onnx.load(p).graph
    arrays={v.name:onnx.numpy_helper.to_array(v).copy() for v in graph.initializer}
    state={k.removeprefix('inner.').replace('pool_attention.','pool.net.'):torch.from_numpy(v.copy())
           for k,v in arrays.items() if k.startswith('inner.')}
    for node in graph.node:
        if node.op_type!='MatMul' or node.input[1] not in arrays: continue
        props={v.key:v.value for v in node.metadata_props}
        scopes=ast.literal_eval(props.get('pkg.torch.onnx.name_scopes','[]'))
        if len(scopes)<2: continue
        name=scopes[-2].removeprefix('inner.').replace('pool_attention.','pool.net.')+'.weight'
        state[name]=torch.from_numpy(arrays[node.input[1]].T.copy())
    cfg=EOTConfig(use_fvad=False,normalize_audio=True)
    m=EOTModel(cfg,pretrained=False).eval()
    m.load_state_dict({k:state[k] for k in m.state_dict()},strict=True)
    opts=ort.SessionOptions(); opts.intra_op_num_threads=2
    sess=ort.InferenceSession(str(p),opts)
    maxdiff=0.
    for n in (1,2,8):
        feats=torch.randn(n,80,800)
        with torch.no_grad(): ref=m(feats)['p_eot'].numpy()
        got=sess.run(None,{'input_features':feats.numpy()})[0].reshape(-1)
        maxdiff=max(maxdiff,float(np.abs(got-ref).max()))
    if maxdiff>1e-4: raise RuntimeError(f'public import parity failed: {maxdiff}')
    out=Path('models/smart-turn-public');out.mkdir(parents=True,exist_ok=True)
    save_checkpoint(m,str(out/'model.pt'))
    (out/'provenance.json').write_text(json.dumps(dict(repository='pipecat-ai/smart-turn-v3',
        revision=REVISION,onnx_sha256=sha256_file(p),checkpoint_sha256=sha256_file(out/'model.pt'),
        import_max_abs_diff=maxdiff,normalize_audio=True),indent=2))
    print('import parity',maxdiff)
if __name__=='__main__': main()
