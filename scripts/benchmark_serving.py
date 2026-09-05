#!/usr/bin/env python
"""Reproducible Docker HTTP concurrency/thread grid; no GPU and no response caching."""
import argparse,asyncio,json,subprocess,time,platform
from pathlib import Path
import httpx
from eot.load_test import run_level,_payload

async def main(args):
    results=[]
    for artifact in args.models:
        for threads in args.threads:
            label=f'{artifact.parent.name}/{artifact.name}:threads={threads}'
            name=f'eot-perf-{__import__("os").getpid()}'
            started=time.perf_counter()
            cid=subprocess.check_output(['docker','run','--rm','-d','--name',name,'-p','127.0.0.1::8000',
                '-v',f'{artifact.parent.resolve()}:/models:ro','-e',f'EOT_ONNX=/models/{artifact.name}',
                args.image,'eot-serve','--host','0.0.0.0','--port','8000','--threads',str(threads),'--max-inflight','8'],text=True).strip()
            try:
                mapping=subprocess.check_output(['docker','port',cid,'8000'],text=True).strip()
                url='http://'+mapping
                async with httpx.AsyncClient(timeout=2.) as c:
                    for _ in range(300):
                        try:
                            r=await c.get(url+'/healthz')
                            if r.status_code==200: break
                        except httpx.HTTPError: pass
                        await asyncio.sleep(.1)
                    else: raise RuntimeError('container never became ready')
                    health=r.json()
                startup=(time.perf_counter()-started)*1000
                body=_payload(args.wav,8.)
                levels=[]
                for concurrency in [1,4,8]:
                    result=await run_level(url,body,concurrency,args.requests)
                    levels.append(result)
                    print(label,concurrency,result['client_total_ms']['p95'],result['errors'],flush=True)
                results.append(dict(artifact=str(artifact),threads=threads,health=health,startup_to_ready_ms=startup,levels=levels))
                args.out.parent.mkdir(parents=True,exist_ok=True)
                args.out.write_text(json.dumps(dict(platform=platform.platform(),image=args.image,results=results),indent=2)+'\n')
            finally:
                subprocess.run(['docker','stop',cid],stdout=subprocess.DEVNULL,check=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,nargs='+',required=True)
    p.add_argument('--threads',type=int,nargs='+',default=[1,2,4])
    p.add_argument('--image',default='happyrobot-eot:optimized')
    p.add_argument('--requests',type=int,default=150)
    p.add_argument('--wav',required=True)
    p.add_argument('--out',type=Path,required=True)
    asyncio.run(main(p.parse_args()))
