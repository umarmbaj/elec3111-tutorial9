"""Server-side analysis adapted from provided MLR/MLP Python desktop examples."""
import numpy as np

def clean_rows(rows, options):
    arr = [r for r in rows if np.isfinite(r['temperature']) and np.isfinite(r['humidity'])]
    if options.get('removeOutliers') and len(arr) >= 8:
        vals = np.array([[r['temperature'], r['humidity']] for r in arr], dtype=float)
        q1, q3 = np.quantile(vals, [0.25, 0.75], axis=0)
        good = np.all((vals >= q1 - 1.5*(q3-q1)) & (vals <= q3 + 1.5*(q3-q1)), axis=1)
        arr = [r for r, k in zip(arr, good) if k]
    return arr

def prepare_rows(rows, options):
    raw=clean_rows(rows, options)
    mode=options.get('scale','none')
    if mode not in ('normalize','standardize'):mode='none'
    transforms={}; data=[dict(v) for v in raw]
    for key in ('temperature','humidity'):
        a=np.array([r[key] for r in raw],dtype=float)
        if len(a)==0:transforms[key]=(0.0,1.0);continue
        shift=float(a.min()) if mode=='normalize' else float(a.mean()) if mode=='standardize' else 0.0
        scale=float(a.max()-a.min()) if mode=='normalize' else float(a.std()) if mode=='standardize' else 1.0
        scale=scale or 1.0
        transforms[key]=(shift,scale)
        for r in data:r[key]=(r[key]-shift)/scale
    return raw,data,transforms,mode

def undo(v,trans):return np.asarray(v)*trans[1]+trans[0]

def mape(y, yh):
    y, yh = np.asarray(y), np.asarray(yh)
    mask = np.abs(y) > 1e-10
    return float(np.mean(np.abs((y[mask]-yh[mask])/y[mask]))*100) if mask.any() else None

def regression(rows, options):
    raw,r,transforms,mode = prepare_rows(rows, options)
    if len(r) < 5:
        raise ValueError('MLR requires at least 5 valid readings.')
    target = 'humidity' if options.get('target') == 'humidity' else 'temperature'
    other = 'temperature' if target == 'humidity' else 'humidity'
    steps = min(100, max(1, int(options.get('steps', 1))))
    y = np.array([v[target] for v in r], dtype=float)
    x = np.array([v[other] for v in r], dtype=float)
    X = np.column_stack([np.ones(len(r)-1), x[:-1], y[:-1]])
    # Least squares, as in the provided MLR desktop app. Tiny ridge term for constant readings.
    A = X.T @ X + np.eye(3)*1e-6
    b = np.linalg.solve(A, X.T@y[1:]); c = np.linalg.solve(A, X.T@x[1:])
    yh = X@b
    xx, yy = float(x[-1]), float(y[-1]); future = []
    for _ in range(steps):
        nxt_y, nxt_x = b[0]+b[1]*xx+b[2]*yy, c[0]+c[1]*xx+c[2]*yy
        xx, yy = float(nxt_x), float(nxt_y); future.append(round(float(undo(yy,transforms[target])),6))
    corr = float(np.corrcoef(x, y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None
    ssx, ssy = float(np.sum((x-x.mean())**2)), float(np.sum((y-y.mean())**2))
    return dict(algorithm='MLR: VAR(1), ordinary least squares with small ridge stabilizer',target=target,other=other,
        count=len(r),steps=steps,coefficients=dict(zip(['b0','b1','b2','c0','c1','c2'], map(float,[*b,*c]))),
        equation=f'Y(t+1) = {b[0]:.5f} + {b[1]:.5f} X(t) + {b[2]:.5f} Y(t)',
        mape=mape([v[target] for v in raw][1:],undo(yh,transforms[target])), correlation=corr,ssX=ssx,ssY=ssy, scale=mode,
        ssXY=float(np.sum((x-x.mean())*(y-y.mean()))),prediction=future[-1],forecast=future,history=raw,
        notes='MAPE is in-sample for MLR. The regression coefficients are in the selected preprocessing scale; forecast is converted back to physical units.')

# A compact NumPy feedforward MLP with momentum, following supplied mlp_realtime_desktop.py.
class MLP:
    def __init__(self, win, hidden, hidden_layers, act, act_out, lr, mom):
        rng=np.random.default_rng(42)
        dims=[win]+[hidden]*hidden_layers+[1]
        self.layers=[];self.velocity=[]
        for i in range(len(dims)-1):
            limit=np.sqrt(6/(dims[i]+dims[i+1]))
            W=rng.uniform(-limit,limit,(dims[i],dims[i+1]));b=np.zeros((1,dims[i+1]))
            self.layers.append([W,b]);self.velocity.append([np.zeros_like(W),np.zeros_like(b)])
        self.act=act;self.act_out=act_out;self.lr=lr;self.mom=mom
    def activate(self,z,name):
        if name=='tanh':return np.tanh(z)
        if name=='relu':return np.maximum(z,0)
        return z
    def prime(self,z,a,name):
        if name=='tanh':return 1-a*a
        if name=='relu':return (z>0).astype(float)
        return np.ones_like(z)
    def forward(self,X):
        activations=[X]; zs=[]; a=X
        for k,(W,b) in enumerate(self.layers):
            z=a@W+b;zs.append(z)
            a=self.activate(z,self.act_out if k==len(self.layers)-1 else self.act)
            activations.append(a)
        return zs, activations
    def fit(self,X,y,epochs):
        loss=0.
        for _ in range(epochs):
            zs,aa=self.forward(X);err=aa[-1]-y
            loss=float(np.mean(err**2));grads=[None]*len(self.layers)
            delta=err*self.prime(zs[-1],aa[-1],self.act_out)
            for k in range(len(self.layers)-1,-1,-1):
                grads[k]=(aa[k].T@delta/len(X),delta.mean(axis=0,keepdims=True))
                if k>0:delta=(delta@self.layers[k][0].T)*self.prime(zs[k-1],aa[k],self.act)
            for (W,b),(vw,vb),(dw,db) in zip(self.layers,self.velocity,grads):
                vw*=self.mom;vw-=self.lr*np.clip(dw,-5,5);W+=vw
                vb*=self.mom;vb-=self.lr*np.clip(db,-5,5);b+=vb
            if not np.isfinite(loss):raise ValueError('MLP training failed; try smaller learning rate')
        return loss
    def predict(self,X):return self.forward(X)[-1][-1]

def neural(rows, options):
    raw,r,transforms,mode=prepare_rows(rows,options)
    if len(r)<60: raise ValueError('MLP needs at least 60 measurements (~5 minutes at 5-second sampling).')
    target='humidity' if options.get('target')=='humidity' else 'temperature'
    series=np.array([v[target] for v in r],dtype=float)
    win=min(40,max(2,int(options.get('window',10))))
    hidden=min(32,max(2,int(options.get('hidden',12))))
    hidden_layers=min(2,max(1,int(options.get('hiddenLayers',1))))
    epochs=min(600,max(1,int(options.get('epochs',200))))
    steps=min(100,max(1,int(options.get('steps',1))))
    if len(series)<win+15:raise ValueError('Not enough readings for the selected input window.')
    X=np.array([series[i-win:i] for i in range(win,len(series))]);y=series[win:]
    ntr=int(.8*len(X))
    if ntr<10 or len(X)-ntr<2:raise ValueError('Insufficient training/validation rows.')
    mean=X[:ntr].mean(axis=0); std=X[:ntr].std(axis=0);std[std==0]=1
    ym=float(y[:ntr].mean()); ys=float(y[:ntr].std()) or 1
    Xs=(X-mean)/std; yn=((y-ym)/ys).reshape(-1,1)
    act='tanh' if options.get('activation')=='tanh' else 'relu'
    lr=min(.1,max(.0001,float(options.get('lr',.01))))
    mom=min(.99,max(0.,float(options.get('momentum',.9))))
    out_act='tanh' if options.get('outputActivation')=='tanh' else 'linear'
    model=MLP(win,hidden,hidden_layers,act,out_act,lr,mom)
    mse=model.fit(Xs[:ntr],yn[:ntr],epochs)
    validation_pred=model.predict(Xs[ntr:]).reshape(-1)*ys+ym
    window=series[-win:].copy();forecast=[]
    for _ in range(steps):
        p=float(model.predict(((window-mean)/std).reshape(1,-1))[0,0]*ys+ym)
        forecast.append(round(float(undo(p,transforms[target])),6));window=np.r_[window[1:],p]
    return dict(algorithm='MLP: server-side sliding-window feedforward neural network',target=target,
                count=len(series),trainingSamples=ntr,validationSamples=len(y)-ntr,
                window=win,hidden=hidden,hiddenLayers=hidden_layers,epochs=epochs,activation=act,outputActivation=out_act,
                trainMSEScaled=mse,validationMAPE=mape(undo(y[ntr:],transforms[target]),undo(validation_pred,transforms[target])),steps=steps,
                prediction=forecast[-1],forecast=forecast,scale=mode,history=raw,
                notes='MLP reports held-out (last 20%) validation MAPE; predictions depend on training data and settings.')
