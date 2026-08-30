#!/usr/bin/env python3
"""Positive transformed-variable PINN for a steady 1-D Dougherty-FP shock."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from shock_physics import *

tf.keras.backend.set_floatx("float64")
def diff(a,h,axis):
 d=(tf.roll(a,-1,axis)-tf.roll(a,1,axis))/(2*h)
 if axis==0: return tf.concat([(a[1:2]-a[:1])/h,d[1:-1],(a[-1:]-a[-2:-1])/h],0)
 if axis==1: return tf.concat([(a[:,1:2]-a[:,:1])/h,d[:,1:-1],(a[:,-1:]-a[:,-2:-1])/h],1)
 return tf.concat([(a[:,:,1:2]-a[:,:,:1])/h,d[:,:,1:-1],(a[:,:,-1:]-a[:,:,-2:-1])/h],2)

def parse():
 p=argparse.ArgumentParser(); p.add_argument("--output",required=True); p.add_argument("--mach",type=float,default=5); p.add_argument("--epochs",type=int,default=12000); p.add_argument("--nx",type=int,default=65); p.add_argument("--nvx",type=int,default=129); p.add_argument("--nvr",type=int,default=64); p.add_argument("--length",type=float,default=10); p.add_argument("--width",type=int,default=96); p.add_argument("--depth",type=int,default=4); p.add_argument("--lr",type=float,default=2e-4); p.add_argument("--nu",type=float,default=1); p.add_argument("--seed",type=int,default=20260921); p.add_argument("--print-every",type=int,default=100); return p.parse_args()

def main():
 a=parse(); np.random.seed(a.seed); tf.random.set_seed(a.seed); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
 up,dn=normal_shock_states(a.mach); th=np.sqrt(dn.temperature); x=np.linspace(-a.length,a.length,a.nx); vx=np.linspace(dn.u-8*th,up.u+8*th,a.nvx); vr=np.linspace(0,8*th,a.nvr)
 dx=x[1]-x[0]; dvx=vx[1]-vx[0]; dvr=vr[1]-vr[0]; X=tf.constant(vx[None,:,None]); R=tf.constant(vr[None,None,:]); Y=tf.constant((x/a.length)[:,None,None]); alpha=.5*(1-Y); U0=alpha*up.u+(1-alpha)*dn.u; T0=alpha*up.temperature+(1-alpha)*dn.temperature
 fup=maxwellian(vx[None,:,None],vr[None,None,:],up); fdn=maxwellian(vx[None,:,None],vr[None,None,:],dn); f0=tf.constant(alpha.numpy()*fup+(1-alpha.numpy())*fdn); bridge=1-tf.square(Y)
 yy=tf.broadcast_to(Y,(a.nx,a.nvx,a.nvr)); xx=tf.broadcast_to((X-U0)/tf.sqrt(T0),(a.nx,a.nvx,a.nvr)); rr=tf.broadcast_to(R/tf.sqrt(T0),(a.nx,a.nvx,a.nvr)); features=tf.reshape(tf.stack((yy,xx,rr),-1),(-1,3))
 layers=[tf.keras.layers.Input((3,))]+[tf.keras.layers.Dense(a.width,activation="tanh") for _ in range(a.depth)]+[tf.keras.layers.Dense(1,kernel_initializer="zeros")]; model=tf.keras.Sequential(layers)
 opt=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(a.lr,max(a.epochs//3,1),.3,staircase=True)); pi=tf.constant(np.pi,dtype=tf.float64); radial=2*pi*R; eps=tf.constant(1e-30,dtype=tf.float64)
 @tf.function
 def step():
  with tf.GradientTape() as tape:
   g=tf.reshape(model(features,training=True),(a.nx,a.nvx,a.nvr)); f=f0*tf.exp(bridge*4*tf.tanh(g/4)); wf=f*radial*dvx*dvr; rho=tf.reduce_sum(wf,(1,2)); mom=tf.reduce_sum(wf*X,(1,2)); u=mom/rho; c2=tf.square(X-u[:,None,None])+tf.square(R); T=tf.reduce_sum(wf*c2,(1,2))/(3*rho)
   jx=(X-u[:,None,None])*f+T[:,None,None]*diff(f,dvx,1); jr=R*f+T[:,None,None]*diff(f,dvr,2); q=diff(jx,dvx,1)+tf.math.divide_no_nan(diff(R*jr,dvr,2),R); q=tf.concat([2*diff(jr,dvr,2)[:,:,:1],q[:,:,1:]],2)*a.nu
   res=X*diff(f,dx,0)-q; interior=res[1:-1,2:-2,1:-2]; scale=tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(q[1:-1,2:-2,1:-2])))+eps); pde=tf.reduce_mean(tf.square(interior/scale))
   mf=mom; pf=tf.reduce_sum(wf*tf.square(X),(1,2)); ef=.5*tf.reduce_sum(wf*X*(tf.square(X)+tf.square(R)),(1,2)); flux=tf.math.reduce_variance(mf/tf.reduce_mean(mf))+tf.math.reduce_variance(pf/tf.reduce_mean(pf))+tf.math.reduce_variance(ef/tf.reduce_mean(ef)); reg=1e-8*tf.reduce_mean(tf.square(g)); loss=pde+100*flux+reg
  grad=tape.gradient(loss,model.trainable_variables); grad,_=tf.clip_by_global_norm(grad,5.); opt.apply_gradients(zip(grad,model.trainable_variables)); return loss,pde,flux,f,rho,u,T,res,pf,ef
 history=[]; start=time.time()
 for epoch in range(1,a.epochs+1):
  loss,pde,flux,f,rho,u,T,res,pf,ef=step()
  if epoch==1 or epoch%a.print_every==0: history.append([epoch,float(loss),float(pde),float(flux)]); print(f"h1g1 epoch={epoch} loss={loss:.6e} pde={pde:.6e} flux={flux:.6e}",flush=True)
 f=f.numpy(); rho=rho.numpy(); u=u.numpy(); T=T.numpy(); res=res.numpy(); measure=radial.numpy()*dvx*dvr; C=vx[None,:,None]-u[:,None,None]; c2=C*C+vr[None,None,:]**2; mf=np.sum(f*measure*vx[None,:,None],(1,2)); pf=pf.numpy(); ef=ef.numpy(); pxx=np.sum(f*measure*C*C,(1,2)); pperp=.5*np.sum(f*measure*vr[None,None,:]**2,(1,2)); qx=.5*np.sum(f*measure*C*c2,(1,2))
 spreads={"mass":float(np.ptp(mf)/abs(np.mean(mf))),"momentum":float(np.ptp(pf)/abs(np.mean(pf))),"energy":float(np.ptp(ef)/abs(np.mean(ef)))}; rr=float(np.sqrt(np.mean(res[1:-1,2:-2,1:-2]**2))/max(np.sqrt(np.mean((a.nu*f[1:-1,2:-2,1:-2])**2)),1e-30)); boundary=float(max(abs(rho[0]/up.rho-1),abs(rho[-1]/dn.rho-1),abs(u[0]/up.u-1),abs(u[-1]/dn.u-1),abs(T[0]/up.temperature-1),abs(T[-1]/dn.temperature-1)))
 gates={"finite":bool(np.isfinite(f).all()),"positive":bool(f.min()>0),"boundary":boundary<2e-3,"mass_flux":spreads["mass"]<5e-3,"momentum_flux":spreads["momentum"]<5e-3,"energy_flux":spreads["energy"]<8e-3,"residual":rr<.15}; status="PASS" if all(gates.values()) else "NO_GO"
 metrics={"stage":"H1_GATE1_STEADY_DOUGHERTY_FP","status":status,"mach":a.mach,"model":"nonlinear local-moment Dougherty FP","coordinates":"x/L, (vx-u0)/sqrt(T0), vr/sqrt(T0)","residual_relative_rms":rr,"flux_relative_spreads":spreads,"boundary_relative_error":boundary,"min_distribution":float(f.min()),"max_abs_heat_flux":float(np.max(abs(qx))),"max_stress_anisotropy":float(np.max(abs(pxx-pperp))),"epochs":a.epochs,"elapsed_seconds":time.time()-start,"gates":gates,"claim":"Discrete FP feasibility gate; external kinetic-reference validation is Gate 2."}; (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); np.savetxt(out/"profiles.csv",np.column_stack((x,rho,u,T,pxx,pperp,qx,mf,pf,ef)),delimiter=",",header="x,rho,u,temperature,pxx,pperp,qx,mass_flux,momentum_flux,energy_flux",comments=""); np.savez_compressed(out/"gate1_solution.npz",x=x,vx=vx,vr=vr,f=f,residual=res,rho=rho,u=u,temperature=T,pxx=pxx,pperp=pperp,qx=qx); model.save_weights(out/"gate1.weights.h5"); np.savetxt(out/"history.csv",history,delimiter=",",header="epoch,loss,pde,flux",comments="")
 fig,ax=plt.subplots(1,3,figsize=(14,4.5)); fig.subplots_adjust(top=.72,wspace=.28); fig.suptitle(f"H1 Gate 1: steady Dougherty–Fokker–Planck shock, Mach {a.mach:g}",y=.98)
 ax[0].plot(x,rho/dn.rho,label=r"$\rho/\rho_2$"); ax[0].plot(x,u/up.u,label=r"$u/u_1$"); ax[0].plot(x,T/dn.temperature,label=r"$T/T_2$"); ax[1].plot(x,pxx-pperp,label=r"$P_{xx}-P_\perp$"); ax[1].plot(x,qx,label=r"$q_x$"); ax[2].semilogy(np.asarray(history)[:,0],np.maximum(np.asarray(history)[:,2],1e-30),label="FP residual"); ax[2].semilogy(np.asarray(history)[:,0],np.maximum(np.asarray(history)[:,3],1e-30),label="flux variance")
 for z,title in zip(ax,("Macroscopic shock profiles","Nonequilibrium moments","Optimization audit")): z.set_title(title,pad=36); z.grid(alpha=.22); z.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.02),ncol=3)
 fig.savefig(out/"h1_gate1_physics.png",dpi=220,bbox_inches="tight"); plt.close(fig); print("H1_GATE1_METRICS",json.dumps(metrics,sort_keys=True)); return 0 if status=="PASS" else 2
if __name__=="__main__": raise SystemExit(main())
