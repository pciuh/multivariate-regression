import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter, StrMethodFormatter

#import scipy.optimize as sco
#import scipy.stats as scs
#import scipy.interpolate as sci

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import Ridge, LassoLars, SGDRegressor, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse

plt.rcParams["font.size"] = '14'
plt.rcParams['mathtext.fontset'] = 'dejavuserif'


def scat2cfplot(data,cMap,Lvl):

    x,y,zl,zm,zr = data
    lLv,mLv,rLv  = Lvl
    wf = 15
    hf = 3
    fig,(axl,axm,axr) = plt.subplots(1,3,figsize=(wf,hf))

    axl.set_title('$X_{Hdl}$')
    scf = axl.tricontourf(x,y,zl,cmap=cMap,levels=lLv)
    axl.set_xlabel(r'$\beta$')
    axl.set_ylabel(r'$r_{dl}$')
    axl.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
    axl.set_aspect(1.0/axl.get_data_ratio(), adjustable='box')
    fig.colorbar(scf,ax=axl,format='%4.1f',shrink=.75,ticks=lLv[::2])

    axm.set_title('$Y_{Hdl}$')
    scf = axm.tricontourf(x,y,zm,cmap=cMap,levels=mLv)
    axm.set_xlabel(r'$\beta$')
    axm.set_aspect(1.0/axm.get_data_ratio(), adjustable='box')
    axm.set_ylabel(r'$r_{dl}$')
    axm.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
    fig.colorbar(scf,ax=axm,format='%4.1f',shrink=.75,ticks=mLv[::2])

    axr.set_title('$N_{Hdl}$')
    scf = axr.tricontourf(x,y,zr,cmap=cMap,levels=rLv)
    axr.set_xlabel(r'$\beta$')
    axr.set_ylabel(r'$r_{dl}$')
    axr.xaxis.set_major_formatter(EngFormatter(unit=u"°"))
    axr.set_aspect(1.0/axr.get_data_ratio(), adjustable='box')
    fig.colorbar(scf,ax=axr,format='%4.1f',shrink=.75,ticks=rLv[::2])

    return(fig,axl,axm,axr)


def predpoly(X,Y,Xo,est,par_gri,REP=False):
    clf = RandomizedSearchCV(est,par_gri,random_state=12321,n_iter=15)
    clf.fit(X,Y)
    bp = clf.best_params_
    est.set_params(**bp)
    est.fit(X,Y)
    pred = est.predict(X)
    sco,coef,mse_ = (est.score(X,Y),est.coef_,mse(pred,Y))
    prod = est.predict(Xo)

    res = pred-Y
    bia = np.sum(res)/len(pred)
    var = np.var(pred)
    ioa = 1-np.sum(res**2)/np.sum((np.abs(pred-np.mean(Y))+np.abs(Y-np.mean(Y)))**2)

    if REP:
        print('\n Bias:%10.1e'%bia)
        print('  Var:%10.1e'%var)
        print('  IOA:%10.2f'%ioa)
        print('   R2:%10.3f'%sco)
        print('  MSE:%10.1e'%mse_)
        for k in bp:
            try:
                a = float(bp[k])
                print('%16s:%10.2e'%(k,bp[k]))
            except:
                print('%16s:%10s'%(k,bp[k]))
    return(pd.Series(pred.ravel()),pd.Series(prod.ravel()),coef,sco,mse_)

def predpoly_(X,Y,Xo,dop,flag):
    pol = PolynomialFeatures(degree=dop,interaction_only=flag,include_bias=False)
    Xt  = pol.fit_transform(X)
    #clf = linear_model.LinearRegression()
    clf = linear_model.Ridge(alpha=.3,tol=1e-10,solver='svd')
    clf.fit(Xt,Y)
    sco,coef = (clf.score(Xt,Y),clf.coef_)
    pred = clf.predict(Xt)
    polf = pol.get_feature_names(['vm','rdl'])
    mse_ = mse(Y,pred)

    print('\n\nReport:')
    print('Poly:',polf)
    print('PolyDeg:',dop)
    print('Coef:',coef)
    print('Intr:',clf.intercept_)
    print('R2:',sco)
    print('MSE:',mse_)

    Xot = pol.fit_transform(Xo)
    prod = clf.predict(Xot)

    return(pd.Series(pred.T[0]),pd.Series(prod.T[0]),coef,sco,polf)

iDir,pDir = 'input/', 'plots/'

SEED = 12321

cl = '#fa00a0'
cs = '#00afa0'

xLvl = np.linspace(5,20,16)
xLvl = [np.linspace(-0.3,0.3,13),np.linspace(-.6,.6,13),np.linspace(-.3,.3,13)]

df = pd.read_csv(iDir + 'data.csv',sep=';')
data = [df.vm,df.rdl,df.XH,df.YH,df.NH]

X   = np.array([data[0],data[1]]).T
Yx   = data[2].values.reshape(-1,1)
Yy   = data[3].values.reshape(-1,1)
Yn   = data[4].values.reshape(-1,1)


Yx   = data[2].values.ravel()
Yy   = data[3].values.ravel()
Yn   = data[4].values.ravel()

nb,nr = (101,101)

xl,yl = (np.linspace(-.35,.35,nb),np.linspace(-.80,.80,nr))
xg,yg = np.meshgrid(xl,yl)
bg = -np.degrees(np.arcsin(xg))

Xo = np.concatenate((xg.reshape(-1,1),yg.reshape(-1,1)),axis=1).T
v,r = (X.T[0],X.T[1])
Xx = np.array([v**2,v*r,r**2,v**3]).T
Xy = np.array([v,r,v**3,v**2*r,v*r**2,r**3]).T

v,r = (Xo[0],Xo[1])
Xxo = np.array([v**2,v*r,r**2,v**3]).T
Xyo = np.array([v,r,v**3,v**2*r,v*r**2,r**3]).T

m_name = ['Ridge','LassoLars','ElasticNet','SGDRegressor']

model = {'Ridge'        : Ridge(),
         'LassoLars'    : LassoLars(),
         'ElasticNet'   : ElasticNet(),
         'SGDRegressor' : SGDRegressor()}

cMap = {'Ridge'         : 'BuPu',
        'LassoLars'     : 'PuRd',
        'ElasticNet'    : 'BuGn',
        'SGDRegressor'  : 'YlGn'}

par_gri = {'Ridge':{'alpha': np.linspace(0.0,1,3000)},
           'LassoLars':{'alpha': np.linspace(0,1e-3,200)},
           'ElasticNet':{'tol':[1e-6],'alpha': np.linspace(0,1e-3,200), 'l1_ratio':np.linspace(0.5,1.0,25)},
           'SGDRegressor':{'tol':[1e-4,1e-6],'power_t': np.linspace(-.5,.5,1000)}}
vcfx,vcfy,vcfn = [],[],[]
for mn in m_name:
    print('\n\n',mn)
    zPrex,zProx,cfx,r2x,msex = predpoly(Xx,Yx,Xxo,model[mn],par_gri[mn],True)
    zPrey,zProy,cfy,r2y,msey = predpoly(Xy,Yy,Xyo,model[mn],par_gri[mn])
    zPren,zPron,cfn,r2n,msen = predpoly(Xy,Yn,Xyo,model[mn],par_gri[mn])

    vcfx = np.append(vcfx,cfx)
    vcfy = np.append(vcfy,cfy)
    vcfn = np.append(vcfn,cfn)

    key = ['X','Y','N']
    dfr = pd.DataFrame(columns=key,index=['R2','MSE'])
    dfr.X = [r2x,msex]
    dfr.Y = [r2y,msey]
    dfr.N = [r2n,msen]
    print(dfr)

    zgx = zProx.to_numpy().reshape(nb,-1)
    zgy = zProy.to_numpy().reshape(nb,-1)
    zgn = zPron.to_numpy().reshape(nb,-1)
    #fig,axx,axy,axn = scat2cfplot([pd.Series(Xo[0]),pd.Series(Xo[1]),zProx,zProy,zPron],cMap,xLvl)
    data[0] = df.beta

    fig,axx,axy,axn = scat2cfplot(data,cMap[mn],xLvl)
    cx = axx.contour(bg,yg,zgx,colors='#ffffff',levels=xLvl[0],linewidths=.75)
    cy = axy.contour(bg,yg,zgy,colors='#f0f0f0',levels=xLvl[1],linewidths=.75)
    cn = axn.contour(bg,yg,zgn,colors='#ffffff',levels=xLvl[2],linewidths=.75)

    axx.clabel(cx,cx.levels,inline=True,fmt='%3.2f')
    axy.clabel(cy,cy.levels,inline=True,fmt='%3.2f')
    axn.clabel(cn,cn.levels,inline=True,fmt='%3.2f')

    fig.savefig(pDir + 'hf-'+mn+'.png',dpi=300,bbox_inches='tight')

#Xx = np.array([v**2,v*r,r**2,v**3]).T
#Xy = np.array([v,r,v**3,v**2*r,v*r**2,r**3]).T
key = ['Xvv','Xvr','Xrr','Xvvv']
vec = vcfx.reshape(len(m_name),-1)
dfx = pd.DataFrame(dict(zip(key,vec.T)),index=m_name)

key = ['Yv','Yr','Yvvv','Yvvr','Yvrr','Yrrr']
vec = vcfy.reshape(len(m_name),-1)
dfy = pd.DataFrame(dict(zip(key,vec.T)),index=m_name)


key = [x.replace('Y','N') for x in key]
vec = vcfn.reshape(len(m_name),-1)
dfn = pd.DataFrame(dict(zip(key,vec.T)),index=m_name)

print(dfx)
print(dfy)
print(dfn)
