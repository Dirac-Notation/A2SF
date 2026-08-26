"""Ensemble fast LB eval: average reward_pred of N policies over LB states, argmax, score via index."""
import argparse, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fast_lb_select import load_agent
from fast_lb_score import score_actions, DEFAULT_INDEX
def main():
    p=argparse.ArgumentParser(); p.add_argument("--ckpts",nargs="+",required=True)
    p.add_argument("--states_path",required=True); p.add_argument("--run_name",required=True)
    p.add_argument("--budget",type=int,default=128); p.add_argument("--index_path",default=DEFAULT_INDEX)
    a=p.parse_args()
    agents=[load_agent(c,False) for c in a.ckpts]; [ag.eval() for ag in agents]
    sd=torch.load(a.states_path,map_location="cpu",weights_only=False)
    datasets=sorted(k[:-len("/states")] for k in sd if k.endswith("/states"))
    actions={}
    with torch.no_grad():
        for ds in datasets:
            st=sd[ds+"/states"].float(); order=sd[ds+"/order"]
            pred=sum(ag.forward(st)["reward_pred"] for ag in agents)/len(agents)
            act=pred.argmax(-1)
            arr=[0]*len(order)
            for i,o in enumerate(order): arr[int(o)]=int(act[i])
            actions[ds]=arr
    score_actions(actions,a.run_name,a.budget,a.index_path)
if __name__=="__main__": main()
