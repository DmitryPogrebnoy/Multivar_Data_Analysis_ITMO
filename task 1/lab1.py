import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

def PDF(n,t,m):
    from scipy.stats import kde
    density = kde.gaussian_kde(n)
    xgrid = np.linspace(n.min(), n.max(), 100)
    plt.title('PDF')
    plt.hist(n, bins=m, density=True)
    plt.plot(xgrid, density(xgrid), 'r-')
    plt.xlabel(f'Distribution of ' + t)
    plt.show()

def BoxWithWhiskers (n,t):
    median = n.quantile(0.5)
    q1 = n.quantile(0.25)
    q3 = n.quantile(0.75)
    iqr = q3 - q1
    print('Median =',median,'; First quartile =',q1,'; Third quartile =',q3,'; Interquartile range =',iqr)
    plt.title('Box With Whiskers')
    plt.xlabel(f'Distribution of ' + t)
    plt.boxplot(n)
    plt.show()

def MaximumLikelihood (n,t,m,lim):
    from scipy.stats import lognorm
    x = np.linspace(n.min(), n.max(), 100)
    parameters = scipy.stats.lognorm.fit(n)
    plt.hist(n, bins=m, density=True)
    pdf = scipy.stats.lognorm.pdf(x, parameters[0], parameters[1], parameters[2])
    plt.xlabel(f'Distribution of ' + t)
    axes = plt.gca()
    axes.set_ylim([0, lim])
    plt.title('Maximum Likelihood Method. Lognormal')
    plt.plot(x, pdf, 'r--')
    plt.show()

def MaximumLikelihoodNormal (n,t,m,lim):
    from scipy.stats import norm
    x = np.linspace(n.min(), n.max(), 100)
    parameters = scipy.stats.norm.fit(n)
    plt.hist(n, bins=m, density=True)
    pdf = scipy.stats.norm.pdf(x, parameters[0], parameters[1])
    plt.xlabel(f'Distribution of ' + t)
    axes = plt.gca()
    axes.set_ylim([0, lim])
    plt.title('Maximum Likelihood Method. Normal')
    plt.plot(x, pdf, 'r--')
    plt.show()

def log(x, m, s):
    return np.exp(-((np.log(x)-m)**2)/(2*s**2))/(x*s*(2*np.pi)**0.5)

def LeastSquaresMethod (n,t,m,m1,lim):
    from scipy import optimize
    value, binsedge = np.histogram(n, bins=m1, density=True)
    ydata = [value[i] for i in range(len(value)-1)]
    xdata = [(binsedge[i] + binsedge[i + 1]) / 2 for i in range(len(value) - 1)]
    popt, pcov = scipy.optimize.curve_fit(log, xdata, ydata)
    mu, sigma = popt
    x = np.linspace(n.min(), n.max(), 100)
    x1 = x.tolist()
    x2 = [log(x2,mu,sigma) for x2 in x1]
    plt.xlabel(f'Distribution of ' + t)
    axes = plt.gca()
    axes.set_ylim([0, lim])
    plt.title('Least Squares Method. Lognormal')
    plt.plot(x, x2, 'r--')
    plt.hist(n, bins=m, density=True)
    plt.show()

def normal(x, m, s):
    return np.exp(-(((x-m)**2)/(2*(s**2)))/(s*(2*np.pi)**0.5))

def LeastSquaresMethodNormal (n,t,m,m1,lim):
    from scipy import optimize
    value, binsedge = np.histogram(n, bins=m1, density=True)
    ydata = [value[i] for i in range(len(value)-1)]
    xdata = [(binsedge[i] + binsedge[i + 1]) / 2 for i in range(len(value) - 1)]
    popt, pcov = scipy.optimize.curve_fit(normal, xdata, ydata)
    mu, sigma = popt
    x = np.linspace(n.min(), n.max(), 100)
    x1 = x.tolist()
    x2 = [normal(x2,mu,sigma) for x2 in x1]
    plt.xlabel(f'Distribution of ' + t)
    axes = plt.gca()
    axes.set_ylim([0, lim])
    plt.title('Least Squares Method. Normal')
    plt.plot(x, x2, 'r--')
    plt.hist(n, bins=m, density=True)
    plt.show()

def QQPlotLognorm (n, m, a, b,t):
    from scipy.stats import lognorm
    parameters = scipy.stats.lognorm.fit(n)
    percs = np.linspace(0, 100, m)
    qn_first = np.percentile(n, percs)
    qn_lognorm = scipy.stats.lognorm.ppf(percs / 100.0, *parameters)

    plt.figure(figsize=(10, 10))
    plt.plot(qn_first, qn_lognorm, ls="", marker="o", markersize=6)
    x = np.linspace(np.min(n), np.max(n))
    plt.plot(x, x, color="k", ls="--")
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.title(t)
    plt.xlabel(f'Empirical distribution')
    plt.ylabel('Theoretical (lognormal) distribution')
    plt.show()

def QQPlotNorm (n, m, a, b,t):
    from scipy.stats import norm
    parameters = scipy.stats.norm.fit(n)
    percs = np.linspace(0, 100, m)
    qn_first = np.percentile(n, percs)
    qn_norm = scipy.stats.norm.ppf(percs / 100.0, *parameters)

    plt.figure(figsize=(10, 10))
    plt.plot(qn_first, qn_norm, ls="", marker="o", markersize=6)
    x = np.linspace(np.min(n), np.max(n))
    plt.plot(x, x, color="k", ls="--")
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.title(t)
    plt.xlabel(f'Empirical distribution')
    plt.ylabel('Theoretical (lognormal) distribution')
    plt.show()

def TestLognorm (n):
    from scipy.stats import lognorm
    parameters = scipy.stats.lognorm.fit(n)
    ks = scipy.stats.kstest(n, 'lognorm', parameters, N=100)
    cm = scipy.stats.cramervonmises(n, 'lognorm', parameters)
    print(ks)
    print(cm)

def TestNorm (n):
    from scipy.stats import norm
    parameters = scipy.stats.norm.fit(n)
    ks = scipy.stats.kstest(n, 'norm', parameters, N=100)
    cm = scipy.stats.cramervonmises(n, 'norm', parameters)
    print(ks)
    print(cm)

def LSM_lognormal_for_LSM_and_MLE (n,m1):
    from scipy import optimize
    value, binsedge = np.histogram(n, bins=m1, density=True)
    ydata = [value[i] for i in range(len(value)-1)]
    xdata = [(binsedge[i] + binsedge[i + 1]) / 2 for i in range(len(value) - 1)]
    popt, pcov = scipy.optimize.curve_fit(log, xdata, ydata)
    mu, sigma = popt
    x = np.linspace(n.min(), n.max(), 100)
    x1 = x.tolist()
    x2 = [log(x2,mu,sigma) for x2 in x1]
    return x2

def MLE_lognormal_for_LSM_and_MLE (n):
    from scipy.stats import lognorm
    x = np.linspace(n.min(), n.max(), 100)
    parameters = scipy.stats.lognorm.fit(n)
    pdf = scipy.stats.lognorm.pdf(x, parameters[0], parameters[1], parameters[2])
    return pdf

def LSM_normal_for_LSM_and_MLE (n,m1):
    from scipy import optimize
    value, binsedge = np.histogram(n, bins=m1, density=True)
    ydata = [value[i] for i in range(len(value)-1)]
    xdata = [(binsedge[i] + binsedge[i + 1]) / 2 for i in range(len(value) - 1)]
    popt, pcov = scipy.optimize.curve_fit(normal, xdata, ydata)
    mu, sigma = popt
    x = np.linspace(n.min(), n.max(), 100)
    x1 = x.tolist()
    x2 = [normal(x2,mu,sigma) for x2 in x1]
    return x2

def MLE_normal_for_LSM_and_MLE (n):
    from scipy.stats import norm
    x = np.linspace(n.min(), n.max(), 100)
    parameters = scipy.stats.norm.fit(n)
    pdf = scipy.stats.norm.pdf(x, parameters[0], parameters[1])
    return pdf

def MLE_and_LSM_Lognormal(n, t, m, m1, lim):
    x = np.linspace(n.min(), n.max(), 100)

    plt.hist(n, bins=m, density=True)
    plt.plot(x, MLE_lognormal_for_LSM_and_MLE(n), 'r--')
    plt.plot(x, LSM_lognormal_for_LSM_and_MLE(n, m1), 'b--')
    plt.legend(['Maximum Likelihood Method', 'Least Squares Method'])

    plt.xlabel(f'Distribution of ' + t)
    axes = plt.gca()
    axes.set_ylim([0, lim])
    plt.title('Maximum Likelihood Method and Least Squares Method. Lognormal')
    plt.show()

def MLE_and_LSM_Normal(n, t, m, m1, lim):
    x = np.linspace(n.min(), n.max(), 100)

    plt.hist(n, bins=m, density=True)
    plt.plot(x, MLE_normal_for_LSM_and_MLE(n), 'r--')
    plt.plot(x, LSM_normal_for_LSM_and_MLE(n, m1), 'b--')
    plt.legend(['Maximum Likelihood Method', 'Least Squares Method'])

    plt.xlabel(f'Distribution of ' + t)
    axes = plt.gca()
    axes.set_ylim([0, lim])
    plt.title('Maximum Likelihood Method and Least Squares Method. Normal')
    plt.show()

path_to_file = 'avocado.csv'
data = pd.read_csv(r'C:\Users\Ксения\Documents\Учеба\ИТМО\1 семестр\Методы\Лаб 1\Код Лаб 1\avocado.csv')
df = pd.DataFrame(data,columns= ['AveragePrice', 'Total Volume', '4046'])


PDF(df['AveragePrice'],'Average Price', 10)
PDF(df['Total Volume'],'Total Volume', 50)
PDF(df['4046'],'Total number of avocados sold with PLU 4046', 50)

BoxWithWhiskers(df['AveragePrice'],'Average Price')
BoxWithWhiskers(df['Total Volume'],'Total Volume')
BoxWithWhiskers(df['4046'],'Total number of avocados sold with PLU 4046')

MaximumLikelihood(df['AveragePrice'],'Average Price', 10, 1.1)
MaximumLikelihood(df['Total Volume'], 'Total Volume', 50, 0.0000008)
MaximumLikelihood(df['4046'], 'Total number of avocados sold with PLU 4046', 50, 0.0000008)

LeastSquaresMethod(df['AveragePrice'],'Average Price', 10, 10, 1.1)
LeastSquaresMethod(df['Total Volume'], 'Total Volume', 50, 100000, 0.0000008)
LeastSquaresMethod(df['4046'], 'Total number of avocados sold with PLU 4046', 50, 100000, 0.0000008)

QQPlotLognorm(df['AveragePrice'], 21, 0, 4,'Average Price. Lognorm')
QQPlotLognorm(df['Total Volume'], 1000, 0, 10000000, 'Total Volume. Lognorm')
QQPlotLognorm(df['4046'], 100, 0, 50000, 'Total number of avocados sold with PLU 4046. Lognorm')

print('Test for Average Price Lognorm')
TestLognorm(df['AveragePrice'])
print('---------------------------------------------------------------------------')
print('Test for Total Volume Lognorm')
TestLognorm(df['Total Volume'])
print('---------------------------------------------------------------------------')
print('Test for 4046 Lognorm')
TestLognorm(df['4046'])


MaximumLikelihoodNormal(df['AveragePrice'],'Average Price', 10, 1.1)
LeastSquaresMethodNormal(df['AveragePrice'],'Average Price', 10, 10, 1.1)
QQPlotNorm(df['AveragePrice'], 21, 0, 4,'Average Price. Normal')
print('Test for Average Price Normal')
TestNorm(df['AveragePrice'])


MLE_and_LSM_Lognormal(df['Average Price'], 'AveragePrice', 10, 100, 1.1)
MLE_and_LSM_Lognormal(df['Total Volume'], 'Total Volume', 50, 100000, 0.0000008)
MLE_and_LSM_Lognormal(df['4046'], 'Total number of avocados sold with PLU 4046', 50, 100000, 0.0000008)

MLE_and_LSM_Normal(df['Average Price'], 'AveragePrice', 10, 100, 1.1)



