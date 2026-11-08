"""Calculate paired day-bootstrap intervals for correlated-arrival results."""
import csv, json, hashlib
from pathlib import Path
from analyze_evaluation_uncertainty import analyze
R=Path(__file__).resolve().parents[1]; O=R/'results/pressure21_v1/r107'
def main() -> None:
    """Use the existing paired bootstrap routine without rerunning simulations.

    :return: None.
    """
    with (O/'daily.csv').open(encoding='utf8',newline='') as f:
        rows=[dict(r,method='IL' if r['method']=='Ours' else r['method'],capacity_kw=r['limit']) for r in csv.DictReader(f) if r['n']=='200']
    result=analyze(rows,('capacity_kw',),{'cost':'cost','unfairness':'D'},('UPA',),10000,20261101)
    pairs=[x for x in result if x['comparison']=='UPA']
    payload={'seed':20261101,'samples':10000,'input_sha256':hashlib.sha256((O/'daily.csv').read_bytes()).hexdigest(),'results':pairs}
    with (O/'bootstrap.json').open('x',encoding='utf8') as f: json.dump(payload,f,indent=2)
    print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
