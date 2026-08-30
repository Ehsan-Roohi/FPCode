#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from shock_physics import *
def spread(a): return float(np.ptp(a)/abs(float(np.mean(a))))
def main():
 p=argparse.ArgumentParser(); p.add_argument("--mach",type=float,default=3.); p.add_argument("--output",default="outputs/h1_gate0_m3"); a=p.parse_args()
 out=Path(a.output); out.mkdir(parents=True,exist_ok=True); up,dn=normal_shock_states(a.mach); th=np.sqrt(max(up.temperature,dn.temperature)); vrmax=8*th
 vx,vr,w=axisymmetric_quadrature(min(up.u,dn.u)-8*th,max(up.u,dn.u)+8*th,320,vrmax,160)
 x=np.linspace(-8,8,161); f=mott_smith_distribution(x,vx,vr,up,dn,1.); m=moments_and_fluxes(f,vx,vr,w)
 rh=float(np.max(abs(analytic_fluxes(up)-analytic_fluxes(dn))/abs(analytic_fluxes(up)))); fs={k:spread(m[k]) for k in ("mass_flux","momentum_flux","energy_flux")}
 be=float(max(abs(m["rho"][0]/up.rho-1),abs(m["rho"][-1]/dn.rho-1),abs(m["u"][0]/up.u-1),abs(m["u"][-1]/dn.u-1),abs(m["temperature"][0]/up.temperature-1),abs(m["temperature"][-1]/dn.temperature-1)))
 gates={"rankine_hugoniot":rh<1e-12,"positive_distribution":float(f.min())>=0,"mass_flux":fs["mass_flux"]<2e-4,"momentum_flux":fs["momentum_flux"]<2e-4,"energy_flux":fs["energy_flux"]<5e-4,"boundary_states":be<2e-3,"nonequilibrium_stress":float(np.max(abs(m["pxx"]-m["pperp"])))>.05,"nonequilibrium_heat_flux":float(np.max(abs(m["qx"])))>.05}; gates={k:bool(v) for k,v in gates.items()}; status="PASS" if all(gates.values()) else "NO_GO"
 metrics={"stage":"H1_GATE0_MOTT_SMITH","status":status,"note":"Initialization/audit manifold; not an FP solution.","mach":a.mach,"upstream":up.as_dict(),"downstream":dn.as_dict(),"rankine_hugoniot_relative_error":rh,"flux_relative_spreads":fs,"boundary_relative_error":be,"min_distribution":float(f.min()),"max_stress_anisotropy":float(np.max(abs(m["pxx"]-m["pperp"]))),"max_abs_heat_flux":float(np.max(abs(m["qx"]))),"gates":gates}; (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
 keys=("rho","u","temperature","pxx","pperp","qx","mass_flux","momentum_flux","energy_flux"); np.savetxt(out/"profiles.csv",np.column_stack([x]+[m[k] for k in keys]),delimiter=",",header="x,"+",".join(keys),comments="")
 wx=w.sum(1)/(np.pi*vrmax**2); mx=(f*w[None]).sum(2)/wx[None]; np.savez_compressed(out/"shock_gate0.npz",x=x,vx=vx,vr=vr,f=f,marginal_x=mx,**m)
 fig,ax=plt.subplots(1,3,figsize=(13.5,4.4),constrained_layout=True); ax[0].plot(x,m["rho"]/dn.rho,label=r"$\rho/\rho_2$"); ax[0].plot(x,m["u"]/up.u,label=r"$u/u_1$"); ax[0].plot(x,m["temperature"]/dn.temperature,label=r"$T/T_2$"); ax[1].plot(x,m["pxx"]-m["pperp"],label=r"$P_{xx}-P_\perp$"); ax[1].plot(x,m["qx"],label=r"$q_x$")
 for i,l in zip((0,len(x)//2,-1),("upstream","shock center","downstream")): ax[2].plot(vx,mx[i],label=l)
 for z,t in zip(ax,(f"Mach {a.mach:g} shock structure","Nonequilibrium moments","Physical velocity distributions")): z.set_title(t,pad=8); z.grid(alpha=.22); z.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.12),ncol=3)
 fig.savefig(out/"h1_gate0_physics.png",dpi=220,bbox_inches="tight"); plt.close(fig); print("H1_GATE0_METRICS",json.dumps(metrics,sort_keys=True)); print("H1_GATE0_STATUS",status); return 0 if status=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
