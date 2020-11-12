#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
import numpy as np


# In[ ]:


def dydx(x,y):
    
    
    y_derivs = np.zeros(2)
    
    y_derivs[0] =  y[1]
    
    y_derivs[1] =  -1*y[0]
    
    return y_derivs


# In[ ]:


def rk4_mv_core(dydx,xi,yi,nv,h):
    
    k1 = np.zeros(nv)
    k2 = np.zeros(nv)
    k3 = np.zeros(nv)
    k4 = np.zeros(nv)
    
    x_ipoh = xi + 0.5*h
    
    x_ipoh = xi + h
    
    y_temp = np.zeros(nv)
    
    y_derivs = dydx(xi,yi)
    k1[:] = h*y_derivs[:]
    
    y_temp[:] = yi[:] + 0.5*k1[:]
    y_derivs = dydx(x_ipoh,y_temp)
    k2[:] = h*y_derivs[:]
    
    y_temp[:] = yi[:] + 0.5*k2[:]
    y_derivs = dydx(x_ipoh,y_temp)
    k3[:] = h*y_derivs[:]
    
    y_temp[:] = yi[:] + k3[:]
    y_derivs = dydx(x_ipoh,y_temp)
    k4[:] = h*y_derivs[:]
    
    yipo = yi + (k1 + 2*k2 + 2*k3 + k4)/6. 
    
    return yipo
    


# In[ ]:


def rk4_mv_ad(dydx,x_i,y_i,nv,h,tol):
    
    SAFETY    = 0.9
    H_NEW_FAC = 2.0
    
    imax = 10000
    
    i = 0
    
    Delta = np.full(nv,2*tol)
    
    h_step = h
    
    while(Delta.max()/tol > 1.0):
        
        y_2 = rk4_mv_core(dydx,x_i,y_i,nv,h_step)
        y_1 = rk4_mv_core(dydx,x_i,y_i,nv,0.5*h_step)
        y_11 = rk4_mv_core(dydx,x_i+0.5*h_step,y_1,nv,0.5*h_step)
        
        Delta = np.fabs(y_2 - y_11)
        
        if(Delta.max()/tol > 1.0):
            h_step *= SAFETY * (Delta.max()/tol)**(-0.25)
            
        if(i>=imax):
            print("Too many iterations in rk4_mv_ad()")
            raise StopIteration("Ending after i =",i)
            
        i+=1
        
    h_new = np.fmin(h_step * (Delta.max()/tol)**(-0.9), h_step*H_NEW_FAC)
    
    return y_2, h_new, h_step  


# In[ ]:


def rk4_mv(dfdx,a,b,y_a,tol):
    
    xi = a 
    yi = y_a.copy()
    
    h = 1.0e-4 * (b-a)
    
    imax = 10000
    
    i = 0
    
    nv = len(y_a)
    
    x = np.full(1,a)
    y = np.full((1,nv),y_a)
    
    flag = 1
    
    while(flag):
        
        yi_new, h_new, h_step = rk4_mv_ad(dydx,xi,yi,nv,h,tol)
        
        h = h_new
        
        if(xi+h_step>b):
            h = b-xi
            
            yi_new, h_new, h_step = rk4_mv_ad(dydx,xi,yi,nv,h,tol)
            
            flag = 0
            
        xi += h_step
        yi[:] = yi_new[:]
        
        x = np.append(x,xi)
        y_new = np.zeros((len(x),nv))
        y_new[0:len(x)-1,:] = y
        y_new[-1,:] = yi[:]
        del y
        y = y_new
        
        if(i>=imax):
            print("Maximum iterations reached.")
            raise StopIteration("Iteration number =",i)
            
        i += 1
        
        s = "i = %3d\tx = %9.8f\th = %9.8f\tb = %9.8f" % (i,xi, h_step, b)
        print(s)
        
        if(xi==b):
            flag = 0
            
    return x,y


# In[ ]:


a = 0.0 
b = 2.0 * np.pi

y_0 = np.zeros(2)
y_0[0] = 0.0
y_0[1] = 1.0
nv = 2

tolerance = 1.0e-6

x,y = rk4_mv(dydx,a,b,y_0,tolerance)


# In[ ]:


plt.plot(x,y[:,0],'o',label='y(x)')
plt.plot(x,y[:,1],'o',label='dydx(x)')
xx = np.linspace(0,2.0*np.pi,1000)
plt.plot(xx,np.sin(xx),label='sin(x)')
plt.plot(xx,np.cos(xx),label='cos(x)')
plt.xlabel('x')
plt.ylabel('y, dy/dx')
plt.legend(frameon=False)


# In[ ]:


sine = np.sin(x)
cosine = np.cos(x)

y_error = (y[:,0]-sine)
dydx_error = (y[:,1]-cosine)

plt.plot(x, y_error, label="y(x) Error")
plt.plot(x, dydx_error, label="dydx(x) Error")
plt.legend(frameon=False)

