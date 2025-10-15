import json, os
def J(p): 
    with open(p,'r') as f: 
        return json.load(f)

offS = J('siml/results/reaching_off/off/summary.json')
fepS = J('siml/results/reaching_fep/fepgate/summary.json')
offP = J('siml/results/reaching_off/power.json')
fepP = J('siml/results/reaching_fep/power.json')

def dump(tag, S, P):
    print(f"\n== {tag} ==")
    print(json.dumps(S, indent=2))
    print("Power:", json.dumps(P, indent=2))

dump("OFF", offS, offP)
dump("FEP", fepS, fepP)

N = float(offS['N'])
print("\nWh/episode OFF:", offP['gpu_energy_Wh']/N)
print("Wh/episode FEP:", fepP['gpu_energy_Wh']/N)
