import numpy as np
import math
from scipy import linalg
from decimal import * 
getcontext().prec = 6
############################################################################################
#计算叉乘
def chacheng(a,b):       #a=np.array([[x],[y],[z]]) same as b
    c=np.array([[a[1,0]*b[2,0]-a[2,0]*b[1,0]],
                [a[2,0]*b[0,0]-a[0,0]*b[2,0]],
                [a[0,0]*b[1,0]-a[1,0]*b[0,0]]])
    return(c)

############################################################################################
#定义旋转函数
def xspin(a):
    b=np.array([[1,0,0],
                [0,np.cos(a),-np.sin(a)],
                [0,np.sin(a),np.cos(a)]])
    return(b)
def zspin(a):
    b=np.array([[np.cos(a),-np.sin(a),0],
                [np.sin(a),np.cos(a),0],
                [0,0,1]])
    return(b)
def yspin(a):
    b=np.array([[np.cos(a),0,np.sin(a)],
                [0,1,0],
                [-np.sin(a),0,np.cos(a)]])
    return(b)

############################################################################################
#利用迭代方法解方程组
def solveequation(A,B,delta):        #解线性方程组，雅可比迭代A为系数矩阵,delta为循环精度值
    col=list(np.shape(A))[1]    
    row=list(np.shape(A))[0]     #判断系数矩阵的维度
    D=np.zeros((row,col));L=np.zeros((row,col));U=np.zeros((row,col))
    for i in range(row):
        D[i,i]=A[i,i]
        if A[i,i]==0:
            print('zeros in D!!')
        for j in range(col):
            if i>j:
                L[i,j]=-A[i,j]
            if j>i:
                U[i,j]=-A[i,j]
    BJ=np.dot(np.linalg.inv(D),(L+U));g=np.dot(np.linalg.inv(D),B)
    x1=np.ones((row,1));
    a=delta
    while a>=delta:
        x2=np.dot(BJ,x1)+g
        print(x2)
        a=abs(np.linalg.norm(x2)-np.linalg.norm(x1))
        x1=x2
    return(x1)
def solveequationls(A,B,delta):    #最小二乘法（least square）解超定方程
    B=np.dot(np.transpose(A),B);A=np.dot(np.transpose(A),A)
    x=solveequation(A,B,delta)
    return(x)

#解三点定轨的方程组(得到r,v)
def solve3p(G,F,R,L):
    '''
    for i in range(3):
        loc = locals()
        exec('g'+str(i+1)+'='+str(G[i]))
        exec('f'+str(i+1)+'='+str(F[i]))
        exec('lx'+str(i+1)+'='+str(L[0,i]))
        exec('ly'+str(i+1)+'='+str(L[1,i]))
        exec('lz'+str(i+1)+'='+str(L[2,i]))
        exec('rx'+str(i+1)+'='+str(R[0,i]))
        exec('ry'+str(i+1)+'='+str(R[1,i]))
        exec('rz'+str(i+1)+'='+str(R[2,i]))
     '''
        
    A = np.zeros((len(G)*2,6))
    B = np.zeros((len(G)*2,1))
    for i in range(len(G)):
        l1 = 2*i; l2 = 2*i+1   #line1的编号
        f = F[i]; g = G[i]
        lx = L[0,i]; ly = L[1,i]; lz = L[2,i]
        rx = R[0,i]; ry = R[1,i]; rz = R[2,i]
        A[l1,0] = A[l1,3] = A[l2,2] = A[l2,5] = 0
        A[l1,1] = f*lz; A[l1,2] = -f*ly; A[l1,4] = g*lz; A[l1,5] = -g*ly
        B[l1,0] = ry*lz-rz*ly
        A[l2,0] = f*ly; A[l2,1] = -f*lx; A[l2,3] = g*ly; A[l2,4] = -g*lx
        B[l2,0] = rx*ly-ry*lx
    solution = linalg.solve(A,B)
    r=np.array([[solution[0,0]],
                [solution[1,0]],
                [solution[2,0]]])
    v=np.array([[solution[3,0]],
                [solution[4,0]],
                [solution[5,0]]])
    return r,v

#解多点定轨方程组
def solvenp(G,F,R,L,n):
    A = np.zeros((len(G)*2,6))    
    B = np.zeros((len(G)*2,1))
    for i in range(len(G)):
        l1 = 2*i; l2 = 2*i+1   #line1的编号
        f = F[i]; g = G[i]
        lx = L[0,i]; ly = L[1,i]; lz = L[2,i]
        rx = R[0,i]; ry = R[1,i]; rz = R[2,i]
        A[l1,0] = A[l1,3] = A[l2,2] = A[l2,5] = 0
        A[l1,1] = f*lz; A[l1,2] = -f*ly; A[l1,4] = g*lz; A[l1,5] = -g*ly
        B[l1,0] = ry*lz-rz*ly
        A[l2,0] = f*ly; A[l2,1] = -f*lx; A[l2,3] = g*ly; A[l2,4] = -g*lx
        B[l2,0] = rx*ly-ry*lx
    B=np.dot(np.transpose(A),B);A=np.dot(np.transpose(A),A)  
    solution = linalg.solve(A,B)
    r=np.array([[solution[0,0]],
                [solution[1,0]],
                [solution[2,0]]])
    v=np.array([[solution[3,0]],
                [solution[4,0]],
                [solution[5,0]]])
    return r,v

#解开普勒方程
def f(E,M,e):
    f = E-e*np.sin(E)-M
    return f
def solvekeplerequation(M,e,epsilon):
    E = 1
    while abs(f(E,M,e)) >= epsilon:
        E = E - f(E,M,e)/(1-e*np.cos(E)) 
    return E
##############################################################################################
#时间系统转换
def transfer1950(t_sun):           #平太阳时转换为平恒星时(1965.1.14，t为小时) 单位是弧度
    s0=100.075540;s=360.9856122863
    S=s0+s*(5492+t_sun/24)
    t_star=(S-int(S/360)*360)/180*np.pi
    return(t_star)

#J2000历元从平太阳时到格林尼治平恒星时的变换(t为J2000平太阳时儒略世纪数365.25天，86400秒)，单位为小时  起点为2000年1月1日12：00
def SG2000(t):
    S=18.6973746+879000.0513367*t+0.093104/3600*t**2-6.2/3600*10**(-6)*t**3
    return(S)

############################################################################################
#坐标系转换
'''
def
'''
def transfer1(lamda,phai,t_star,Robserve):          #R(观测站)地固坐标转换为地心天球坐标,t为格林尼治平恒星时 phai为纬度
    R=np.dot(zspin(t_star),np.array([[Robserve*np.cos(phai)*np.cos(lamda)],
                                      [Robserve*np.cos(phai)*np.sin(lamda)],
                                      [Robserve*np.sin(phai)]]))
    return(R)

'''
def transfer2(Rsun_earth,epsilon,R):    #地心天球转日心黄道
'''
##############################################################################################
#计算观测站到卫星的单位矢量L
def calculateL(alpha,delta):
    L = np.zeros([3,1])
    L[0,0] = np.cos(delta)*np.cos(alpha)
    L[1,0] = np.cos(delta)*np.sin(alpha)
    L[2,0] = np.sin(delta)
    return L

#根据sin和cos的值计算确定象限的角度值
def calculateduesin_cos(sin,cos):
    if sin == 0:
        if cos == 1:
            return 0
        else:
            return np.pi
    elif cos == 0:
        if sin == 1:
            return np.pi/2
        else:
            return np.pi*3/2
    elif sin/cos > 0:
        if sin > 0:
            return math.atan(sin/cos)
        else:
            return math.atan(sin/cos)+np.pi
    else:
        if sin > 0:
            return math.atan(sin/cos)+np.pi
        else:
            return math.atan(sin/cos)+2*np.pi
#计算t0时刻的轨道根数
def calculateroot(r,v):
    rabssqure = 0; vabssqure = 0; sigma = 0
    for i in range(3):
        rabssqure += r[i,0]**2; vabssqure += v[i,0]**2
        sigma += r[i,0]*v[i,0]
    rabs = np.sqrt(rabssqure); vabs=np.sqrt(vabssqure)
    a = abs(1/(2/rabs-vabssqure))
    n = np.sqrt(1/a**3)
    ecosE = 1-rabs/a; esinE = sigma/(a**2*n)
    e = np.sqrt(ecosE**2+esinE**2)
    cosE = ecosE/e; sinE = esinE/e
    E = calculateduesin_cos(sinE,cosE)
    M = E-e*np.sin(E)
    h = chacheng(r,v); habs = np.sqrt(h[0,0]**2+h[1,0]**2+h[2,0]**2)
    i = math.acos(h[2,0]/habs)
    Omega = calculateduesin_cos(h[0,0]/(habs*np.sin(i)),-h[1,0]/(habs*np.sin(i)))
    P = np.cos(E)/rabs*r-np.sin(E)/(a*n)*v; Q = np.sin(E)/rabs/np.sqrt(abs(1-e**2))*r+(np.cos(E)-e)/(a*n*np.sqrt(abs(1-e**2)))*v
    omega = calculateduesin_cos(P[2,0]/np.sin(i),Q[2,0]/np.sin(i))    
    return a,e,i,E,M,Omega,omega
##计算新的F和G
#级数方法
def calculateFG1(tau,r,v):
    F = []; G = []; sigma = 0; rabssqure = 0; vabssqure = 0
    for i in range(3):
        rabssqure += r[i,0]**2; vabssqure += v[i,0]**2
        sigma += r[i,0]*v[i,0]
    rabs = np.sqrt(rabssqure); vabs=np.sqrt(vabssqure)
    u = 1/rabs
    for i in range(len(tau)):
        f = 1-u**3*tau[i]**2/2+u**5*sigma*tau[i]**3/2+1/24*u**5*(3*vabssqure-2*u-15*u**2*sigma**2)*tau[i]**4+1/8*u**7*sigma*(-3*vabssqure+2*u+7*u**2*sigma**2)*tau[i]**5+1/720*u**7*(u**2*sigma**2*(630*vabssqure-420*u-945*u**2*sigma**2)-(22*u**2-66*u*vabssqure+45*vabssqure))*tau[i]**6
        g = tau[i]-1/6*u**3*tau[i]**3+1/4*u**5*sigma*tau[i]**4+1/120*u**5*(9*vabssqure-8*u-45*u**2*sigma**2)*tau[i]**5+1/24*u**7*sigma*(-6*vabssqure+5*u+14*u**2*sigma**2)*tau[i]**6
        F += [f]; G += [g]
    return F,G
#封闭方法
def calculateFG2(tau,r,v):
    a,e,i,E,M,Omega,omega = calculateroot(r,v)    
    n = np.sqrt(1/a**3)
    M0 = M; E0 = E
    F = []; G = []; rabssqure = 0; vabssqure = 0
    for i in range(3):
        rabssqure += r[i,0]**2; vabssqure += v[i,0]**2
    rabs = np.sqrt(rabssqure)
    for i in range(len(tau)):       
        E = solvekeplerequation(n*tau[i]+M0,e,1e-14)   
        deltaE = E-E0
        F += [1-a/rabs*(1-np.cos(deltaE))]
        G += [tau[i]-1/n*(deltaE-np.sin(deltaE))]
    return F,G

#计算星历表
def calculater_v(a,e,i,Omega,omega,M0,t0,t):
    n = np.sqrt(1/a**3)
    E = solvekeplerequation(n*(t-t0)+M0,e,1e-14)    
    P = np.array([[np.cos(Omega)*np.cos(omega)-np.sin(Omega)*np.sin(omega)*np.cos(i)],
                  [np.sin(Omega)*np.cos(omega)+np.cos(Omega)*np.sin(omega)*np.cos(i)],
                  [np.sin(omega)*np.sin(i)]])
    Q=np.array([[-np.cos(i)*np.cos(omega)*np.sin(Omega)-np.cos(Omega)*np.sin(omega)],
                [np.cos(i)*np.cos(omega)*np.cos(Omega)-np.sin(omega)*np.sin(Omega)],
                [np.cos(omega)*np.sin(i)]])
    r = a*(np.cos(E)-e)*P+a*np.sqrt(1-e**2)*np.sin(E)*Q
    rabs = np.sqrt(r[0,0]**2+r[1,0]**2+r[2,0]**2)
    v = -a**2*n/rabs*np.sin(E)*P+a**2*n/rabs*np.sqrt(abs(1-e**2))*np.cos(E)*Q
    return r,v



#######
#main program
###卫星定轨
#指定选取的数据点
num = [1,2,3]

##初始值
lamda=118.82091666/180*np.pi;phai=31.893611111/180*np.pi;Robserve=0.999102
ut1=[21.575128333,21.603110555,21.631419722,21.654491388,21.714094166,21.742875555]    #单位为小时
alphastation=[142.935,157.274166667,171.817916667,183.134166667,208.640416667,219.102083333]
deltastation=[8.521111111,-2.395,-14.508333333,-23.56333333,-40.0202777777,-44.998611111]
alphastationdegree=[142.935,157.274166667,171.817916667,183.134166667,208.640416667,219.102083333]
deltastationdegree=[8.521111111,-2.395,-14.508333333,-23.56333333,-40.0202777777,-44.998611111]
#将角度全部转化为弧度
for i in range(len(ut1)):
    alphastation[i]=alphastation[i]/180*np.pi;deltastation[i]=deltastation[i]/180*np.pi
#将时间单位化并计算恒星时
tunit=[]; t_star=[]
for i in range(len(ut1)):
    tunit += [ut1[i]*3600/807.3033597477413]  #806.8129    808.9862    807.3033597477413
    t_star += [transfer1950(ut1[i])]

t_star_h = []
for i in range(len(t_star)):
    t_star_h += [t_star[i]*180/2/np.pi/15]
#选取t0
t0 = tunit[0]/2+tunit[5]/2

#R为每一个时刻测站的地心赤道坐标  L为每一时刻在测站坐标系下的单位矢量
R = np.zeros([3,len(num)]);L = np.zeros([3,len(num)]);F = [];G = []
for i in range(len(num)):
    r = transfer1(lamda,phai,t_star[num[i]-1],Robserve)
    l = calculateL(alphastation[num[i]-1],deltastation[num[i]-1])
    R[0,i] = r[0,0]; R[1,i] = r[1,0]; R[2,i] = r[2,0]
    L[0,i] = l[0,0]; L[1,i] = l[1,0]; L[2,i] = l[2,0]
    F += [1]; G += [tunit[num[i]-1]-t0]
tau=G

##通过循环计算轨道根数
method = int(input('级数法输入1,封闭法输入2:'))
deltamax = 1
while deltamax > 1e-14:
    r,v = solvenp(G,F,R,L,len(num))
    if method == 1:
        Fnew,Gnew = calculateFG1(tau,r,v)
    else:
        Fnew,Gnew = calculateFG2(tau,r,v)
    deltalist = []
    for i in range(len(tau)):
        deltalist += [Fnew[i]-F[i],Gnew[i]-G[i]]
    deltamax = max(max(deltalist),abs(min(deltalist)))
    F = Fnew;G = Gnew
a,e,i,E,M,Omega,omega = calculateroot(r,v) 
print('a=',a,'AU','\ne=',e,'\ni=',i*180/np.pi,'degree','\n','M=',M*180/np.pi,'degree','\nΩ=',Omega*180/np.pi,'degree','\nω=',omega*180/np.pi,'degree','\n','E=',E*180/np.pi,'degree')

##计算不同时刻的球面坐标
deltanew = [];alphanew = []
for num2 in [1,2,3,4,5,6]:
    t_calculate = tunit[num2-1]
    ##计算在地心赤道坐标系下的星历表
    r,v = calculater_v(a,e,i,Omega,omega,M,t0,t_calculate)   #计算的星历表是在地心赤道坐标系中的



    ##计算在测站坐标系下的星历表
    t_star = transfer1950(ut1[num2-1])
    R = transfer1(lamda,phai,t_star,Robserve)
    r = r-R
    #计算坐标
    delta = math.asin(r[2,0]/np.sqrt(r[2,0]**2+r[1,0]**2+r[0,0]**2))
    alpha = calculateduesin_cos(r[1,0]/np.sqrt(r[0,0]**2+r[1,0]**2),r[0,0]/np.sqrt(r[0,0]**2+r[1,0]**2))
    deltanew += [delta*180/np.pi]; alphanew += [alpha*180/np.pi]
alphadelta = []; deltadelta =[]
for i in range(6):
    alphadelta += [alphastationdegree[i]-alphanew[i]]
    deltadelta += [deltastationdegree[i]-deltanew[i]]
for i in range(6):
    print('Δα'+str(i+1)+'='+str(alphadelta[i]),' ','Δδ'+str(i+1)+'='+str(deltadelta[i]))






#main program
#指定计算的点
#num = [1,11,21,31]
###小行星定轨   t0 = 1998 12 04.0     tunit = 58.1324409d
#J2000历元从平太阳时到格林尼治平恒星时的变换(t为J2000平太阳时儒略世纪数365.25天，86400秒)，单位为小时  起点为2000年1月1日12：00
def SG2000(t):
    S=18.6973746+879000.0513367*t+0.093104/3600*t**2-6.2/3600*10**(-6)*t**3
    return(S)
def calculater_v_branch(a,e,i,Omega,omega,M0,t):
    n = np.sqrt(1/a**3)
    E = solvekeplerequation(n*t+M0,e,1e-14)    
    P = np.array([[np.cos(Omega)*np.cos(omega)-np.sin(Omega)*np.sin(omega)*np.cos(i)],
                  [np.sin(Omega)*np.cos(omega)+np.cos(Omega)*np.sin(omega)*np.cos(i)],
                  [np.sin(omega)*np.sin(i)]])
    Q=np.array([[-np.cos(i)*np.cos(omega)*np.sin(Omega)-np.cos(Omega)*np.sin(omega)],
                [np.cos(i)*np.cos(omega)*np.cos(Omega)-np.sin(omega)*np.sin(Omega)],
                [np.cos(omega)*np.sin(i)]])
    r = a*(np.cos(E)-e)*P+a*np.sqrt(1-e**2)*np.sin(E)*Q
    rabs = np.sqrt(r[0,0]**2+r[1,0]**2+r[2,0]**2)
    v = -a**2*n/rabs*np.sin(E)*P+a**2*n/rabs*np.sqrt(abs(1-e**2))*np.cos(E)*Q
    return r,v
def transfer1_branch(lamda,t_star,Rcos,Rsin):
    R=np.dot(zspin(t_star),np.array([[Rcos*np.cos(lamda)],
                                      [Rcos*np.sin(lamda)],
                                      [Rsin]]))
    return R
##初始值整理
#原始数据
time = [[12.0, 4.13532], [12.0, 4.14534], [12.0, 5.09543], [12.0, 5.09868], [12.0, 3.36964], [12.0, 3.37277], [12.0, 3.38338], [12.0, 3.38649], [12.0, 3.39707], [12.0, 4.37163], [12.0, 4.38105], [12.0, 4.38413], [12.0, 4.38727], [12.0, 4.39042], [12.0, 4.03368], [12.0, 4.03506], [12.0, 4.0361], [12.0, 4.03745], [12.0, 4.4528], [12.0, 4.46571], [12.0, 4.47843], [12.0, 4.49293], [11.0, 25.45604], [11.0, 25.46656], [11.0, 25.47715], [11.0, 25.48772], [12.0, 1.87403], [12.0, 1.88457], [12.0, 1.89575], [12.0, 2.80773], [12.0, 2.81219], [12.0, 2.81639], [12.0, 2.89715], [12.0, 2.90591], [12.0, 2.91478], [12.0, 4.79603], [12.0, 4.79988], [12.0, 4.8038], [12.0, 4.88939], [12.0, 4.89332], [12.0, 4.89734]]
sun_position = [[11.0, 25.0, -0.4554321, -0.8036028, -0.3484062], [11.0, 26.0, -0.4398144, -0.8106968, -0.3514825], [11.0, 27.0, -0.4240627, -0.8175404, -0.3544503], [11.0, 28.0, -0.408182, -0.8241316, -0.3573086], [11.0, 29.0, -0.3921772, -0.8304686, -0.3600567], [11.0, 30.0, -0.3760533, -0.8365496, -0.3626937], [12.0, 1.0, -0.3598152, -0.8423733, -0.3652189], [12.0, 2.0, -0.3434676, -0.8479379, -0.3676317], [12.0, 3.0, -0.3270153, -0.853242, -0.3699313], [12.0, 4.0, -0.3104629, -0.8582842, -0.3721173], [12.0, 5.0, -0.293815, -0.8630629, -0.3741888], [12.0, 6.0, -0.2770762, -0.8675765, -0.3761452]]
sun_position_dic = {'25': [-0.4554321, -0.8036028, -0.3484062], '26': [-0.4398144, -0.8106968, -0.3514825], '27': [-0.4240627, -0.8175404, -0.3544503], '28': [-0.408182, -0.8241316, -0.3573086], '29': [-0.3921772, -0.8304686, -0.3600567], '30': [-0.3760533, -0.8365496, -0.3626937], '1': [-0.3598152, -0.8423733, -0.3652189], '2': [-0.3434676, -0.8479379, -0.3676317], '3': [-0.3270153, -0.853242, -0.3699313], '4': [-0.3104629, -0.8582842, -0.3721173], '5': [-0.293815, -0.8630629, -0.3741888], '6': [-0.2770762, -0.8675765, -0.3761452]}
alpha = [10.96020278, 10.96067222, 11.00633056, 11.00648333, 10.92378611, 10.92394167, 10.92443056, 10.924575, 10.92504167, 10.97163611, 10.972075, 10.9722, 10.97234722, 10.97250833, 10.955475, 10.95553333, 10.95557222, 10.95556444, 10.97548056, 10.97609167, 10.97667222, 10.97736944, 10.55532778, 10.555775, 10.55628889, 10.55672778, 10.85284167, 10.85333333, 10.85384167, 10.89707778, 10.89728333, 10.89747778, 10.90121667, 10.90161944, 10.90203333, 10.99200556, 10.99218889, 10.992375, 10.9964, 10.99658056, 10.99676389]
delta = [19.7, 19.7, 19.783333, 19.783333, 19.625083, 19.625583, 19.726667, 19.627, 19.627806, 19.724611, 19.725611, 19.725944, 19.726361, 19.726583, 19.683333, 19.683333, 19.683333, 19.683333, 19.732056, 19.733333, 19.734694, 19.736, 18.970833, 18.972028, 18.972306, 18.973194, 19.483333, 19.484528, 19.483333, 19.570417, 19.566667, 19.566667, 19.579083, 19.566667, 19.566667, 19.767028, 19.766667, 19.766667, 19.776806, 19.766667, 19.766667]
lamda =  [34.7625, 34.7625, 34.7625, 34.7625, 279.2379, 279.2379, 279.2379, 279.2379, 279.2379, 279.2379, 279.2379, 279.2379, 279.2379, 279.2379, 14.29, 14.29, 14.29, 14.29, 254.9897, 254.9897, 254.9897, 254.9897, 253.34, 253.34, 253.34, 253.34, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575, 117.575]
Rcos =  [0.86165, 0.86165, 0.86165, 0.86165, 0.88044, 0.88044, 0.88044, 0.88044, 0.88044, 0.88044, 0.88044, 0.88044, 0.88044, 0.88044, 0.659, 0.659, 0.659, 0.659, 0.76865, 0.76865, 0.76865, 0.76865, 0.833, 0.833, 0.833, 0.833, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278, 0.76278]
Rsin =  [0.50608, 0.50608, 0.50608, 0.50608, 0.47257, 0.47257, 0.47257, 0.47257, 0.47257, 0.47257, 0.47257, 0.47257, 0.47257, 0.47257, 0.748, 0.748, 0.748, 0.748, 0.63793, 0.63793, 0.63793, 0.63793, 0.544, 0.544, 0.544, 0.544, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447, 0.6447]
'''
timen = []
alphan = []
deltan = []
lamdan = []
Rcosn = []
Rsinn = []
for i in num:
    timen += [time[i]]
    alphan += [alpha[i]]
    deltan += [delta[i]]
    lamdan += [lamda[i]]
    Rcosn += [Rcos[i]]
    Rsinn += [Rsin[i]]
time = timen
alpha =alphan
delta =deltan
lamda =lamdan
Rcos =Rcosn
Rsin =Rsinn
'''



Re_s = 149597870700; Re = 6371393
t0 = 4.0
#将原始数据转化为有用的数据
tunit = 58.1324409
t_century = []
tau = []
for i in range(len(time)):
    if time[i][0] == 12:
        tau += [(time[i][1]-t0)/tunit]
        t_century += [(time[i][1]-31.5-365)/36525]
    else:
        tau += [(time[i][1]-t0-30)/tunit]
        t_century += [(time[i][1]-61.5-365)/36525]
    alpha[i] = alpha[i]/180*np.pi; delta[i] = delta[i]/180*np.pi
for i in range(len(lamda)):
    lamda[i] = lamda[i]/180*np.pi
    Rcos[i] = Rcos[i]*Re/Re_s
    Rsin[i] = Rsin[i]*Re/Re_s
#计算每一个点的平恒星时       
t_star = []
for i in range(len(t_century)):
    t_star += [SG2000(t_century[i])]
    t_star[i] = t_star[i]-int(t_star[i]/360)*360 
#L为每一时刻在测站坐标系下的单位矢量
L = np.zeros([3,len(alpha)]);F = [];G = tau
for i in range(len(alpha)):
    l = calculateL(alpha[i],delta[i])
    L[0,i] = l[0,0]; L[1,i] = l[1,0]; L[2,i] = l[2,0]
    F += [1]
#R为每一个时刻测站的日心赤道坐标
Rearth = np.zeros([3,len(alpha)])    #Rearth为地心在日心赤道坐标中的坐标
for i in range(len(alpha)):
    r =sun_position_dic[str(int(time[i][1]))]
    Rearth[0,i] = -r[0]; Rearth[1,i] = -r[1]; Rearth[2,i] = -r[2]
Robserve_earth = np.zeros([3,len(alpha)])
for i in range(len(alpha)):
    r = transfer1_branch(lamda[i],t_star[i],Rcos[i],Rsin[i])
    Robserve_earth[0,i] = r[0,0];Robserve_earth[1,i] = r[1,0];Robserve_earth[2,i] = r[2,0]
R = Rearth+Robserve_earth
##通过循环计算轨道根数
deltamax = 1
while deltamax > 1e-10:
    r,v = solvenp(G,F,R,L,len(alpha))
    Fnew,Gnew = calculateFG1(tau,r,v)
    deltalist = []
    for i in range(len(tau)):
        deltalist += [Fnew[i]-F[i],Gnew[i]-G[i]]
    deltamax = max(max(deltalist),abs(min(deltalist)))
    F = Fnew;G = Gnew
a,e,i,E,M,Omega,omega = calculateroot(r,v) 
print('a=',a,'e=',e,'i=',i*180/np.pi,'\n','M=',M*180/np.pi,'Omega=',Omega*180/np.pi,'omega=',omega*180/np.pi,'\n','E=',E*180/np.pi)