#!/usr/bin/env python3
"""H1R positive transformed PINN with structural conservation of three fluxes."""
import argparse,json,os,time
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL","1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from shock_physics import *
from fp_operator import simpson_weights
tf.keras.backend.set_floatx("float64")

def diff(a,h,axis):
 d=(tf.roll(a,-1,axis)-tf.roll(a,1,axis))/(2*h)
 if axis==0:return tf.concat([(a[1:2]-a[:1])/h,d[1:-1],(a[-1:]-a[-2:-1])/h],0)
 if axis==1:return tf.concat([(a[:,1:2]-a[:,:1])/h,d[:,1:-1],(a[:,-1:]-a[:,-2:-1])/h],1)
 return tf.concat([(a[:,:,1:2]-a[:,:,:1])/h,d[:,:,1:-1],(a[:,:,-1:]-a[:,:,-2:-1])/h],2)

def parse():
 p=argparse.ArgumentParser(); p.add_argument("--output",required=True); p.add_argument("--mach",type=float,default=5); p.add_argument("--epochs",type=int,default=3000); p.add_argument("--nx",type=int,default=65); p.add_argument("--nvx",type=int,default=129); p.add_argument("--nvr",type=int,default=81); p.add_argument("--length",type=float,default=10); p.add_argument("--width",type=int,default=96); p.add_argument("--depth",type=int,default=4); p.add_argument("--lr",type=float,default=1e-4); p.add_argument("--nu",type=float,default=1); p.add_argument("--seed",type=int,default=20260922); p.add_argument("--print-every",type=int,default=100); p.add_argument("--tilt-steps",type=int,default=12); p.add_argument("--tilt-damping",type=float,default=.8); return p.parse_args()

def main():
 a=parse(); np.random.seed(a.seed); tf.random.set_seed(a.seed); out=Path(a.output); out.mkdir(parents=True,exist_ok=True); up,dn=normal_shock_states(a.mach); th=np.sqrt(dn.temperature)
 x=np.linspace(-a.length,a.length,a.nx); vx=np.linspace(dn.u-8*th,up.u+8*th,a.nvx); vr=np.linspace(0,8*th,a.nvr); dx=x[1]-x[0]; dvx=vx[1]-vx[0]; dvr=vr[1]-vr[0]
 X=tf.constant(vx[None,:,None]); R=tf.constant(vr[None,None,:]); Y=tf.constant((x/a.length)[:,None,None]); alpha=.5*(1-Y); U0=alpha*up.u+(1-alpha)*dn.u; T0=alpha*up.temperature+(1-alpha)*dn.temperature; bridge=1-tf.square(Y)
 fup=maxwellian(vx[None,:,None],vr[None,None,:],up); fdn=maxwellian(vx[None,:,None],vr[None,None,:],dn); f0=tf.constant(alpha.numpy()*fup+(1-alpha.numpy())*fdn)
 shape=(a.nx,a.nvx,a.nvr); yy=tf.broadcast_to(Y,shape); cx=tf.broadcast_to((X-U0)/tf.sqrt(T0),shape); cr=tf.broadcast_to(R/tf.sqrt(T0),shape); features=tf.reshape(tf.stack((yy,cx,cr),-1),(-1,3))
 model=tf.keras.Sequential([tf.keras.layers.Input((3,))]+[tf.keras.layers.Dense(a.width,activation="tanh") for _ in range(a.depth)]+[tf.keras.layers.Dense(1,kernel_initializer="zeros")]); opt=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(a.lr,max(a.epochs//2,1),.3,staircase=True))
 wx=tf.constant(simpson_weights(a.nvx,dvx)[None,:,None]); wr=tf.constant(simpson_weights(a.nvr,dvr)[None,None,:]); measure=wx*wr*(2*np.pi*R); us=up.u; vshape=(1,a.nvx,a.nvr); psi=tf.stack((tf.broadcast_to(X/us,vshape),tf.broadcast_to(tf.square(X)/us**2,vshape),tf.broadcast_to(.5*X*(tf.square(X)+tf.square(R))/us**3,vshape)),-1); psif=tf.broadcast_to(psi,shape+(3,))
 disc=[np.sum(fb[...,None]*measure.numpy()[...,None]*psi.numpy(),axis=(1,2))[0] for fb in (fup,fdn)]; target=tf.constant(np.asarray(analytic_fluxes(up))/np.array([us,us**2,us**3])); eye=tf.eye(3,dtype=tf.float64)
 def tilt(fraw):
  beta=tf.zeros((a.nx,3),tf.float64)
  for _ in range(a.tilt_steps):
   f=fraw*tf.exp(tf.clip_by_value(tf.einsum("xi,xvri->xvr",beta,psif),-20.,20.)); wf=f*measure; mom=tf.einsum("xvr,xvri->xi",wf,psif); jac=tf.einsum("xvr,xvri,xvrj->xij",wf,psif,psif); delta=tf.linalg.solve(jac+1e-10*eye[None],(target[None]-mom)[...,None])[...,0]; beta=beta+a.tilt_damping*delta
  f=fraw*tf.exp(tf.clip_by_value(tf.einsum("xi,xvri->xvr",beta,psif),-20.,20.)); fm=tf.einsum("xvr,xvri->xi",f*measure,psif); return f,fm,beta
 @tf.function
 def step():
  with tf.GradientTape() as tape:
   g=tf.reshape(model(features,training=True),shape); raw=f0*tf.exp(bridge*4*tf.tanh(g/4)); f,fm,beta=tilt(raw); wf=f*measure; rho=tf.reduce_sum(wf,(1,2)); mom=tf.reduce_sum(wf*X,(1,2)); u=mom/rho; c2=tf.square(X-u[:,None,None])+tf.square(R); T=tf.reduce_sum(wf*c2,(1,2))/(3*rho)
   jx=(X-u[:,None,None])*f+T[:,None,None]*diff(f,dvx,1); jr=R*f+T[:,None,None]*diff(f,dvr,2); q=(diff(jx,dvx,1)+tf.math.divide_no_nan(diff(R*jr,dvr,2),R))*a.nu; q=tf.concat([2*a.nu*diff(jr,dvr,2)[:,:,:1],q[:,:,1:]],2); res=X*diff(f,dx,0)-q; ri=res[1:-1,2:-2,1:-2]; qs=q[1:-1,2:-2,1:-2]; pde=tf.reduce_mean(tf.square(ri/(tf.stop_gradient(tf.sqrt(tf.reduce_mean(tf.square(qs))))+1e-30))); terr=tf.reduce_max(abs(fm-target)); loss=pde+1e3*terr**2+1e-8*tf.reduce_mean(g*g)
  grad=tape.gradient(loss,model.trainable_variables); grad,_=tf.clip_by_global_norm(grad,3.); opt.apply_gradients(zip(grad,model.trainable_variables)); return loss,pde,terr,f,rho,u,T,res,fm,beta
 hist=[]; start=time.time()
 for e in range(1,a.epochs+1):
  loss,pde,terr,f,rho,u,T,res,fm,beta=step()
  if e==1 or e%a.print_every==0: hist.append((e,float(loss),float(pde),float(terr))); print(f"h1r epoch={e} loss={loss:.6e} pde={pde:.6e} tilt={terr:.3e}",flush=True)
 f=f.numpy(); rho=rho.numpy(); u=u.numpy(); T=T.numpy(); res=res.numpy(); fm=fm.numpy(); C=vx[None,:,None]-u[:,None,None]; c2=C*C+vr[None,None,:]**2; mn=measure.numpy(); pxx=np.sum(f*mn*C*C,(1,2)); pp=.5*np.sum(f*mn*vr[None,None,:]**2,(1,2)); qx=.5*np.sum(f*mn*C*c2,(1,2)); physical=fm*np.array([us,us**2,us**3]); spreads={k:float(np.ptp(physical[:,i])/abs(np.mean(physical[:,i]))) for i,k in enumerate(("mass","momentum","energy"))}; rr=float(np.sqrt(np.mean(res[1:-1,2:-2,1:-2]**2))/max(np.sqrt(np.mean((a.nu*f[1:-1,2:-2,1:-2])**2)),1e-30)); be=float(max(abs(rho[0]/up.rho-1),abs(rho[-1]/dn.rho-1),abs(u[0]/up.u-1),abs(u[-1]/dn.u-1),abs(T[0]/up.temperature-1),abs(T[-1]/dn.temperature-1)))
 tilt_rel=float(np.max(abs(fm-target.numpy()[None]))/max(np.max(abs(target.numpy())),1e-30)); gates={"finite":bool(np.isfinite(f).all()),"positive":bool(f.min()>0),"boundary":be<2e-3,"mass_flux":spreads["mass"]<2e-3,"momentum_flux":spreads["momentum"]<2e-3,"energy_flux":spreads["energy"]<2e-3,"tilt_convergence":tilt_rel<1e-6,"residual":rr<.20}; status="PILOT_PASS" if all(gates.values()) else "NO_GO"; metrics={"stage":"H1R2_FULL_DOMAIN_FLUX_PILOT","status":status,"mach":a.mach,"residual_relative_rms":rr,"flux_relative_spreads":spreads,"flux_tilt_relative_error":tilt_rel,"boundary_relative_error":be,"min_distribution":float(f.min()),"max_abs_heat_flux":float(np.max(abs(qx))),"max_stress_anisotropy":float(np.max(abs(pxx-pp))),"tilt_beta_max":float(np.max(abs(beta.numpy()))),"tilt_steps":a.tilt_steps,"tilt_damping":a.tilt_damping,"epochs":a.epochs,"elapsed_seconds":time.time()-start,"gates":gates,"claim":"Structural-conservation pilot; held-out reference qualification remains required."}; (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n"); np.savetxt(out/"profiles.csv",np.column_stack((x,rho,u,T,pxx,pp,qx,physical)),delimiter=",",header="x,rho,u,temperature,pxx,pperp,qx,mass_flux,momentum_flux,energy_flux",comments=""); np.savez_compressed(out/"h1r_solution.npz",x=x,vx=vx,vr=vr,f=f,residual=res,rho=rho,u=u,temperature=T,pxx=pxx,pperp=pp,qx=qx); model.save_weights(out/"h1r.weights.h5"); np.savetxt(out/"history.csv",hist,delimiter=",",header="epoch,loss,pde,tilt_error",comments="")
 fig,ax=plt.subplots(1,3,figsize=(14,4.5)); fig.subplots_adjust(top=.72,wspace=.28); ax[0].plot(x,rho/dn.rho,label=r"$\rho/\rho_2$"); ax[0].plot(x,u/up.u,label=r"$u/u_1$"); ax[0].plot(x,T/dn.temperature,label=r"$T/T_2$"); ax[1].plot(x,pxx-pp,label=r"$P_{xx}-P_\perp$"); ax[1].plot(x,qx,label=r"$q_x$"); h=np.asarray(hist); ax[2].semilogy(h[:,0],h[:,2],label="FP residual"); ax[2].semilogy(h[:,0],np.maximum(h[:,3],1e-30),label="flux tilt error"); fig.suptitle(f"H1R structural-conservation pilot, Mach {a.mach:g}",y=.98)
 for z,t in zip(ax,("Macroscopic profiles","Nonequilibrium moments","Optimization audit")): z.set_title(t,pad=36); z.grid(alpha=.22); z.legend(frameon=False,loc="lower center",bbox_to_anchor=(.5,1.02),ncol=3)
 fig.savefig(out/"h1r_physics.png",dpi=220,bbox_inches="tight"); plt.close(fig); print("H1R_METRICS",json.dumps(metrics,sort_keys=True)); return 0 if status=="PILOT_PASS" else 2
if __name__=="__main__": raise SystemExit(main())
