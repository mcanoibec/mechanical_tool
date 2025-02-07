import matplotlib.pylab as plt
import numpy as np
from tqdm import tqdm
from copy import copy
from scipy.stats import linregress
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from functools import partial
from os import listdir
from os.path import isfile, join
import inspect

def visualize_curves(folder, show_every=1):
    file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
    samples=np.arange(0,len(file_list)-1,show_every).astype(int)
    samples_files=[file_list[f] for f in samples]
    skipped=0
    plt.figure()
    for i in tqdm(np.arange(len(samples_files)), desc="Plotting curves"):
        filename=rf'{folder}\{samples_files[i]}'
        try:
            cc=mechanical_curve(filename, print_error=0)
            plt.plot(cc.zpiezo, cc.vdef, alpha=0.5)
        except:
            skipped+=1
    plt.xlabel('Z Piezo [m]')
    plt.ylabel('Vertical Deflection [V]')
    print(rf'Plot finalized. {len(samples_files)-skipped} curves shown, {skipped} skipped')
    

def sensitivity_analysis(folder, use_area_in_rough_topo=1):
    file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
    if use_area_in_rough_topo:
        try:
            substrate_idxs=np.loadtxt(r'temporary\cutout_idx.txt').astype(int)
        except:
            raise TypeError('Error: Cutout file missing or broken')
    else:
        substrate_idxs=np.arange(len(file_list)).astype(int)
    skipped=0
    slopes=[]
    for i in tqdm(substrate_idxs):
        try:
            cc=mechanical_curve(rf'{folder}\{file_list[i]}', print_error=0)
            cc.divide_curve()
            cc.fit_idnt()
            slopes.append(abs(cc.slope))
        except:
            skipped+=1
    print(rf'{skipped} curves were skipped')
    return slopes

def get_rough_topo(folder, setpoint, save_as=r'temporary\rough_topo.txt'):
    file_list = [f for f in listdir(folder) if isfile(join(folder, f))]
    rough_topo=[]
    skipped=0
    for i in tqdm(np.arange(len(file_list))):
        filename=rf'{folder}\{file_list[i]}'
        try:
            cc=mechanical_curve(filename, print_error=0)
            cc.get_height(setpoint=setpoint, use_vdef=1)
            rough_topo.append(cc.topo_height)
        except:
            rough_topo.append(np.nan)
            skipped+=1
    print(rf'{skipped} curves were skipped')
    rough_topo=pd.DataFrame(rough_topo)
    rough_topo=rough_topo.interpolate()
    rough_topo=np.array(rough_topo)
    rough_topo=rough_topo.reshape(int(np.sqrt(len(rough_topo)+1)),int(np.sqrt(len(rough_topo)+1)))
    np.savetxt(save_as,rough_topo)
    return rough_topo

def sweep_linfit(x,y, backward=0):
    x=np.array(x)
    y=np.array(y)
    sweep=np.arange(len(x)-1)+1
    offsets=[]
    slopes=[]
    r2s=[]
    s_stdevs=[]
    o_stdevs=[]
    for ii in sweep:
        if backward:
            linfit=linregress(x=x[ii:], y=y[ii:])
        else:
            linfit=linregress(x=x[0:ii], y=y[0:ii])
        offsets.append(linfit.intercept)
        slopes.append(linfit.slope)
        r2s.append(linfit.rvalue**2)
        s_stdevs.append(linfit.stderr)
        o_stdevs.append(linfit.intercept_stderr)
    out=pd.DataFrame({'m': slopes,
                      'b': offsets,
                      'rsqr': r2s,
                       'm dev': s_stdevs,
                       'b dev': o_stdevs
                      })
    # if backward:
    #     out=out.iloc[::-1].reset_index(drop=True)
    return out

def sweep_for_peak(x, backward=0):
    x=np.array(x)
    for i in np.arange(len(x)):
        if backward:
            try:
                if x[-i]>x[-i-1]:
                    peak_idx=len(x)-i
                    break
            except:
                peak_idx=len(x)-i
        else:
            try:
                if x[i]>x[i+1]:
                    peak_idx=i
                    break
            except:
                peak_idx=i
    return peak_idx

def deform_curve_from_piecewise_fit(curve, view_plots=0):
    fmin = np.min(curve)
    fptp = np.max(curve) - fmin
    y = (curve - fmin) / fptp
    x = np.linspace(0, 1, len(curve), endpoint=True)
    xx=[x[0],x[-1]]
    yy=[y[0],y[-1]]

    linreg=linregress(x=xx, y=yy)
    b=linreg.intercept
    m=linreg.slope
    line=linreg.slope*x+linreg.intercept

    alpha = -np.arctan((m))
    yr = x * np.sin(alpha) + y * np.cos(alpha)

    if view_plots:
        plt.figure()
        plt.plot(x, y)
        plt.plot(x, line)
    return yr

def linear_sample(vector, nsamples):
    sampling=np.linspace(0,len(vector),nsamples, endpoint=False).astype(int)
    sampled=np.full_like(vector, np.nan)
    sampled[sampling]=vector[sampling]
    sampled=pd.DataFrame(sampled)
    sampled=sampled.interpolate(method='linear')
    return sampled

def get_height(idnt, setpoint, view_plot=0):
    backward=0
    idnt=copy(idnt)
    idnt.apply_preprocessing(preprocessing=["compute_tip_position",        # Corrects height for cantiliever bending
                                         "correct_tip_offset",          # Calculate contact point
                                         ],   
                                         )
    zpiezo=idnt["height (measured)"][idnt["segment"] == backward]*(1/1e-9)
    force=idnt["force"][idnt["segment"] == backward]*(1/1e-9)

    cp=np.argwhere(idnt["tip position"][idnt["segment"] == backward] == 0)[0,0]
    linfit=linregress(x=zpiezo[0:cp], y=force[0:cp])
    line=linfit.intercept + linfit.slope*zpiezo
    
    force2=force-line
    idx=(np.abs(force2 - setpoint)).argmin()
    height=zpiezo[idx]
    if view_plot:
        plt.plot(zpiezo, force)
        plt.plot(zpiezo, line)
        plt.xlabel('Z Piezo [nm]')
        plt.ylabel('Force [nN]')
        plt.figure()
        plt.plot(zpiezo, force2)
        plt.scatter(zpiezo[idx], force2[idx], color='red')
        plt.xlabel('Z Piezo [nm]')
        plt.ylabel('Force [nN]')

    return height

def fit_sweep_forward(poc_idx, z, f, model, sample_sweep=1, nsamples=10, plot_sweep=1, plot_fit=1):
    tests=np.arange(poc_idx+1, len(z)-1)
    poc=z[poc_idx]
    residuals=[]
    ymod=[]
    zc=[]
    fvecs=[]
    covars=[]
    for i in tqdm(tests):
        try:
            initial_guess = [1e3, poc]
            params = curve_fit(model, z[0:i], f[0:i], p0=initial_guess, full_output=True)
            fit1=model(np.array(z), *params[0])
            fvec=f[i-1:]-fit1[i-1:]
            res=np.sqrt(np.mean(fvec**2))
            residuals.append(res)
            ymod.append(params[0][0])
            zc.append(params[0][1])
            fvecs.append(fvec)
            covars.append(params[1])
        except:
            residuals.append(1)
            ymod.append(1)
            zc.append(1)

    if sample_sweep:
        sampling=np.linspace(0,len(residuals),nsamples, endpoint=False).astype(int)
        sampled_res=np.full_like(residuals, np.nan)
        for i in sampling:
            try:
                sampled_res[i]=residuals[i]
            except:
                print(i)

        sampled_res=pd.DataFrame(sampled_res)
        sampled_res=sampled_res.interpolate(method='linear')
        sampled_res=(sampled_res).iloc[:,0]
    else:
        sampled_res=residuals

    minim=np.argmin(sampled_res)

    if plot_sweep:
        fig=plt.figure()
        ax=fig.add_subplot(3,1,1)
        ax.plot(z[poc_idx+2:],ymod)
        ax.set_ylabel("EMod [Pa]")
        ax.set_title('Fit sweep')

        ax=fig.add_subplot(3,1,2)
        ax.plot(z[poc_idx+2:],zc)
        ax.set_ylabel("Zc [m]")

        ax=fig.add_subplot(3,1,3)
        ax.plot(z[poc_idx+2:],sampled_res)
        ax.scatter(z[poc_idx+2:][minim],sampled_res[minim], color='r', zorder=10, s=60)
        ax.set_ylabel("RMS residuals [N]")
        ax.set_xlabel("Tip-Sample Position [m]")
    if plot_fit:
        plt.figure()
        fit1=model(np.array(z), ymod[minim],zc[minim])
        plt.plot(z,f, color='b')
        plt.plot(z,fit1, color='r')

class mechanical_curve:
    def __init__(self, filename, backward=0, print_error=1):
        try:
            matrix=np.loadtxt(rf'{filename}', comments='#')


            # Initialize lists to store numerical values
            forw = []
            backw = []

            # Open the file and read line by line
            with open(rf'{filename}', 'r') as file:
                lines = file.readlines()

            # Flag to determine which list to append to
            append_to_first_list = True

            for line in lines:
                line = line.strip()
                if line.startswith('# segment: extend'):
                    is_forward=True
                elif line.startswith('# segment: retract'):
                    is_forward=not is_forward
                
                if line.startswith('#') or line=='':
                    pass                   
                elif is_forward:
                    forw.append(0)
                else:
                    backw.append(0)

            segment=np.concatenate((np.full_like(forw,0, dtype=int), np.full_like(backw, 1,dtype=int)), axis=0)
            del forw, backw
            
            vdef=matrix[:,1][segment==backward]
            zpiezo=matrix[:,0][segment==backward]
            self.zpiezo=zpiezo
            self.raw_zpiezo=copy(zpiezo)
            self.vdef=vdef
            self.divide_curve()
            # self.segment=segment
        except:
            if print_error:
                print('Error importing curve')
            
        
    def divide_curve(self):
        vv=self.vdef
        yr=deform_curve_from_piecewise_fit(vv)
        xr=np.linspace(0,1,len(yr))
        # xr=np.flip(x)
        div=np.argmin(yr)
        self.div=div
        self.yr=yr
        self.xr=xr
    def fit_baseline(self):
        div=self.div
        x=self.xr
        yr=self.yr
        vv=self.vdef
        zz=self.zpiezo

        add=int(len(x)*0.2)

        if (div+add)>len(x)-1:
            add=len(x)-div
        elif (div-add)<0:
            add=div

        fit_base=sweep_linfit(x[0:div+add], yr[0:div+add])


        sampled_r2_based=linear_sample(fit_base['rsqr'], 20)
        peak_r2_base=sweep_for_peak(sampled_r2_based, backward=1)
        m_base=fit_base['m'][peak_r2_base]
        b_base=fit_base['b'][peak_r2_base]
        line_base=m_base*x+b_base #baseline fit on deformed curve
        fit_base['rsqr']=sampled_r2_based

        baseee=linregress(zz[:peak_r2_base], vv[:peak_r2_base])
        baseee=baseee.intercept+baseee.slope*zz #baseline fit on raw -curve

        self.baseline_fit_on_deformed_curve=line_base
        self.baseline_fit=baseee
        self.baseline_stop=peak_r2_base

        self.baseline_sweep=fit_base
        self.add=add
    def correct_baseline(self):
        self.fit_baseline()
        vv=self.vdef
        baseee=self.baseline_fit
        vvnew=vv-baseee
        self.vdef=vvnew

    def fit_idnt(self):
        zz=self.zpiezo
        vvnew=self.vdef
        fit_idnt=sweep_linfit(zz, vvnew, backward=1)
        sampling=np.linspace(0,len(fit_idnt['rsqr']),40, endpoint=False).astype(int)
        sampled_r2_idnt=np.full_like(fit_idnt['rsqr'], np.nan)
        sampled_r2_idnt[sampling]=fit_idnt['rsqr'][sampling]
        sampled_r2_idnt=pd.DataFrame(sampled_r2_idnt)
        sampled_r2_idnt=sampled_r2_idnt.interpolate(method='linear')
        sampled_r2_idnt=(sampled_r2_idnt).iloc[:,0]
        sampled_r2_idnt[-1]=0
        fit_idnt['rsqr']=sampled_r2_idnt



        # peak_r2_idnt=sweep_for_peak(sampled_r2_idnt, backward=0)
        peak_r2_idnt, _ = find_peaks(sampled_r2_idnt, height=0.5, distance=10)
        peaks=pd.DataFrame(sampled_r2_idnt).iloc[peak_r2_idnt]
        npeak = peaks.unstack().nlargest(1).index
        peak_r2_idnt= list(npeak.get_level_values(1))
        # peak_r2_idnt=peak_r2_idnt[npeak]



        line_idnt = fit_idnt['m'].iloc[peak_r2_idnt[0]]*zz + fit_idnt['b'].iloc[peak_r2_idnt[0]]

        self.idnt_fit=line_idnt
        self.idnt_start=peak_r2_idnt
        self.idnt_sweep=fit_idnt
        self.slope=fit_idnt['m'].iloc[peak_r2_idnt[0]]

    def get_height(self, setpoint, use_vdef=1):
        #Should this be calculated with tip position?
        vdef=self.vdef
        zz=self.zpiezo
        rzz=self.raw_zpiezo

        if use_vdef:
            topo=(np.abs(vdef - setpoint)).argmin()
            height=rzz[topo]
        else:
            try:
                force=self.force
                tp=self.tip_position
            except:
                raise Exception('Force or tip position channels not found')
            topo=(np.abs(force - setpoint)).argmin()
            height=rzz[topo]

        self.topo_height=height
        self.topo_idx=topo
    def zero_x_axis(self, poc='baseline_stop'):
        zz=self.zpiezo
        point = getattr(self, poc, None)
        zz=zz-zz[point]
        self.zpiezo=zz
        self.poc=zz[point]
        self.poc_idx=point
    def callibrate(self,sensitivity, k):
        vv=self.vdef
        mm=vv/sensitivity
        ff=mm*k
        self.vdef_m=mm
        self.force=ff
    def correct_bending(self):
        zz=self.zpiezo
        mm=self.vdef_m
        tp=zz+mm
        self.tip_position=tp
    def fit_sweep_backward(self, model, initial_guess, params=None, sample_sweep=1, nsamples=10, plot_sweep=1, plot_fit=1):
        z=self.tip_position
        f=self.force
        poc_idx=self.poc_idx
        poc=z[poc_idx]
        tests=np.arange(poc_idx+1, len(z)-1)

        residuals=[]
        fvecs=[]
        covars=[]

        model_partial = partial(model, **params)
        
        param_list = list(inspect.signature(model).parameters.keys())[1:]
        missing_keys = [key for key in param_list if key not in params.keys()]

        fit_params_matrix=np.zeros(shape=(len(tests), (len(missing_keys))))
        n=0

       
        if len(missing_keys)==len(initial_guess):
            for i in tqdm(tests):
                try:
                    fitted_params, pconv = curve_fit(model_partial, z[i:], f[i:], p0=initial_guess)
                    all_params=np.concatenate((fitted_params, list(params.values())),axis=0)
                    fit2=model(np.array(z), *all_params)
                    fvec=f[i-1:]-fit2[i-1:]
                    res=np.sqrt(np.mean(fvec**2))
                    residuals.append(res)

                    fit_params_matrix[n]=fitted_params
                    fvecs.append(fvec)
                    covars.append(pconv)

                except:
                    residuals.append(1)
                    fit_params_matrix[n]=np.ones_like(missing_keys)
                n+=1
                
            if sample_sweep:
                sampling=np.linspace(0,len(residuals),nsamples, endpoint=False).astype(int)
                sampled_res=np.full_like(residuals, np.nan)
                for i in sampling:
                    try:
                        sampled_res[i]=residuals[i]
                    except:
                        print(i)

                sampled_res=pd.DataFrame(sampled_res)
                sampled_res=sampled_res.interpolate(method='linear')
                sampled_res=(sampled_res).iloc[:,0]
            else:
                sampled_res=residuals

            minim=np.argmin(sampled_res)
            if plot_sweep:
                fig=plt.figure()
                for i in np.arange(len(missing_keys)):
                    ax=fig.add_subplot(3,1,i+1)
                    ax.plot(z[poc_idx+2:],fit_params_matrix[:,i])
                    ax.set_ylabel(rf"{missing_keys[i]}")
                    if i==0:
                        ax.set_title('Backward Fit sweep')
                
                ax=fig.add_subplot(3,1,i+2)
                ax.plot(z[poc_idx+2:],sampled_res)
                ax.scatter(z[poc_idx+2:][minim],sampled_res[minim], color='r', zorder=10, s=60)
                ax.set_ylabel("RMS residuals [N]")
                ax.set_xlabel("Tip-Sample Position [m]")
            

            final_params=np.concatenate((fit_params_matrix[minim], list(params.values())),axis=0)
            fit2=model(np.array(z), *final_params)

            print(rf"Backward sweep fit: {missing_keys} = {fit_params_matrix[minim]}")
            print(rf"Covariances = {np.diag(covars[minim])}")
         
            if plot_fit:
                plt.figure()
                plt.plot(z,f, color='b')
                plt.plot(z,fit2, color='r')
                plt.title('Backward sweep fit')
            self.poc_bw=fit_params_matrix[minim][1]
            self.fit_backward=fit2
            self.sweep_model_bw=model.__name__
        else:
            raise Exception("Error: The number of initial values must correspond to the number of free parameters")
        

    def fit_sweep_forward(self, model, initial_guess, params=None, sample_sweep=1, nsamples=10, plot_sweep=1, plot_fit=1):#REMOVE PLOTS LATER
        z=self.tip_position
        f=self.force
        poc_idx=self.poc_idx
        tests=np.arange(poc_idx+1, len(z)-1)
        poc=z[poc_idx]

        residuals=[]
        fvecs=[]
        covars=[]
        model_partial = partial(model, **params)

        param_list = list(inspect.signature(model).parameters.keys())[1:]
        missing_keys = [key for key in param_list if key not in params.keys()]
    

        fit_params_matrix=np.zeros(shape=(len(tests), (len(missing_keys))))
        n=0

        if len(missing_keys)==len(initial_guess):

            for i in tqdm(tests):
                try:
                    fitted_params, pconv = curve_fit(model_partial, z[0:i], f[0:i], p0=initial_guess)
                    all_params=np.concatenate((fitted_params, list(params.values())),axis=0)
                    fit1=model(np.array(z), *all_params)
                    fvec=f[i-1:]-fit1[i-1:]
                    res=np.sqrt(np.mean(fvec**2))
                    residuals.append(res)

                    fit_params_matrix[n]=fitted_params
                    fvecs.append(fvec)
                    covars.append(pconv)
                except:
                    fit_params_matrix[i]=np.ones_like(missing_keys)
                    residuals.append(1)
                n+=1

            if sample_sweep:
                sampling=np.linspace(0,len(residuals),nsamples, endpoint=False).astype(int)
                sampled_res=np.full_like(residuals, np.nan)
                for i in sampling:
                    try:
                        sampled_res[i]=residuals[i]
                    except:
                        print(i)

                sampled_res=pd.DataFrame(sampled_res)
                sampled_res=sampled_res.interpolate(method='linear')
                sampled_res=(sampled_res).iloc[:,0]
            else:
                sampled_res=residuals

            minim=np.argmin(sampled_res)

            if plot_sweep:
                fig=plt.figure()
                for i in np.arange(len(missing_keys)):
                    ax=fig.add_subplot(3,1,i+1)
                    ax.plot(z[poc_idx+2:],fit_params_matrix[:,i])
                    ax.set_ylabel(rf"{missing_keys[i]}")
                    if i==0:
                        ax.set_title('Forward Fit sweep')
                
                ax=fig.add_subplot(3,1,i+2)
                ax.plot(z[poc_idx+2:],sampled_res)
                ax.scatter(z[poc_idx+2:][minim],sampled_res[minim], color='r', zorder=10, s=60)
                ax.set_ylabel("RMS residuals [N]")
                ax.set_xlabel("Tip-Sample Position [m]")
            
            final_params=np.concatenate((fit_params_matrix[minim], list(params.values())),axis=0)

            fit1=model(np.array(z), *final_params)

            print(rf"Forward sweep fit: {missing_keys} = {fit_params_matrix[minim]}")
            print(rf"Covariances = {np.diag(covars[minim])}")

            if plot_fit:
                plt.figure()
                plt.plot(z,f, color='b')
                plt.plot(z,fit1, color='r')
                plt.title('Forward sweep fit')
            self.fit_forward=fit1
            self.sweep_model_fw=model.__name__
        else:
            raise Exception("Error: The number of initial values must correspond to the number of free parameters")

        
    def merge_fits(self, plot_fit=1):
        poc_bw=self.poc_bw
        
        fit1=self.fit_forward
        fit2=self.fit_backward
        tp=self.tip_position
        ff=self.force
        poc_idx=self.poc_idx
        poc_bw_idx=np.argmin(np.abs(tp-poc_bw))

        transition=np.argmin(abs(fit1[poc_bw_idx:]-fit2[poc_bw_idx:]))
        transition=transition+poc_bw_idx
        fit_final=np.concatenate((fit1[0:transition],fit2[transition:]), axis=0)
        if plot_fit:
            plt.figure()
            plt.plot(tp, ff, color='b', label='Force')
            plt.plot(tp, fit1, color='orange', label=f'Forward Sweep: {self.sweep_model_fw}')
            plt.plot(tp, fit2, color='r', label=f'Backward Sweep: {self.sweep_model_bw}')
            plt.legend()
            plt.xlabel('Tip-Sample [m]')
            plt.ylabel('Force [N]')

            plt.figure()
            plt.plot(tp,ff)
            plt.plot(tp,fit_final)
            plt.xlabel('Tip-Sample [m]')
            plt.ylabel('Force [N]')
            plt.title('Merged fit')

        self.fit_merged=fit_final
        self.merge_transition=transition
    def fit(self, model, initial_guess=[1e3,0], max_indentation=None,params=None,plot=1):
        
        poc_idx=self.poc_idx
        z=self.tip_position
        f=self.force
        poc=z[poc_idx]
        if max_indentation!=None:
            end=np.argmin(abs((z*-1)-max_indentation))
        else:
            end=None
                
        param_list = list(inspect.signature(model).parameters.keys())[1:]
        missing_keys = [key for key in param_list if key not in params.keys()]

        # param_list = inspect.signature(model).parameters
        # param_list = list(param_list.keys())[1:]


        # dict_keys = set(params.keys())
        # keys_set = set(param_list)

        # missing_keys = list(keys_set - dict_keys)

        if len(missing_keys)==len(initial_guess):
            model_partial = partial(model, **params)

            params_fitted, pcov = curve_fit(model_partial, z[:end], f[:end], p0=initial_guess)
            print(rf'{missing_keys} = {params_fitted}')
            final_params=np.concatenate((params_fitted, list(params.values())),axis=0)
            fit=model(np.array(z), *final_params)
            perr = np.sqrt(np.diag(pcov))
            self.fiting=fit
            if plot:
                plt.figure()
                plt.plot(z,f)
                plt.plot(z,fit)
                # plt.title(rf'E ={ params[0]:.2f} Pa, uncertainty = {perr[0]:.2f}; Zc = {params[1]:.2e} m,  uncertainty = {perr[1]:.2e}, ')
                plt.xlabel('Tip-Sample [m]')
                plt.ylabel('Force [N]')
        else:
              raise Exception("Error: The number of initial values must correspond to the number of free parameters")

    def fit_double_regime(self):
        a=1