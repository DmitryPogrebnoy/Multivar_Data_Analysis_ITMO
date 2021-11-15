import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns



def PDF(n):
    from scipy.stats import kde
    density = kde.gaussian_kde(df[n])
    xgrid = np.linspace(df[n].min(), df[n].max(), 100)
    plt.hist(df[n], bins=10, density=True)
    plt.plot(xgrid, density(xgrid), 'r-')
    plt.show()


def BoxWithWhiskers (n):
    #sns.boxplot(data=df[n], orient='h')
    plt.boxplot(df[n])
    plt.show()

def MaximumLikelihood (n):
    from scipy.stats import lognorm
    x = np.linspace(df[n].min(), df[n].max(), 100)
    parameters = scipy.stats.lognorm.fit(df[n])
    plt.hist(df[n], bins=10, density=True)
    pdf = scipy.stats.lognorm.pdf(x, parameters[0], parameters[1],parameters[2])
    plt.plot(x, pdf, 'r--')
    plt.show()

def log(x, m, s):
    return np.exp(-((np.log(x)-m)**2)/(2*s**2))/(x*s*(2*np.pi)**0.5)


def LeastSquaresMethod (n):
    value, binsedge, _ = plt.hist(df[n], bins=10, density=True)
    ydata = [value[i] for i in range(len(value)-1)]
    xdata = [(binsedge[i] + binsedge[i + 1]) / 2 for i in range(len(value) - 1)]
    popt, pcov = scipy.optimize.curve_fit(log, xdata, ydata)
    mu, sigma = popt
    #print('mu = ',mu)
    x = np.linspace(df[n].min(), df[n].max(), 100)
    x1 = x.tolist()
    x2 = [log(x2,mu,sigma) for x2 in x1]
    plt.plot(x, x2, 'r--')
    plt.hist(df[n], bins=5, density=True)
    plt.show()

def QQPlotLognorm (n, m, a, b):
    from scipy.stats import lognorm
    # Calculation of quantiles
    parameters = scipy.stats.lognorm.fit(df[n])
    percs = np.linspace(0, 100, m)
    qn_first = np.percentile(df[n], percs)
    qn_lognorm = scipy.stats.lognorm.ppf(percs / 100.0, *parameters)

    # Building a quantile biplot
    plt.figure(figsize=(10, 10))
    plt.plot(qn_first, qn_lognorm, ls="", marker="o", markersize=6)
    x = np.linspace(np.min(df[n]), np.max(df[n]))
    plt.plot(x, x, color="k", ls="--")
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.xlabel(f'Empirical distribution')
    plt.ylabel('Theoretical (lognormal) distribution')
    plt.show()

def QQPlotGamma (n, m, a, b):
    from scipy.stats import gamma
    # Calculation of quantiles
    parameters = scipy.stats.gamma.fit(df[n])
    percs = np.linspace(0, 100, m)
    qn_first = np.percentile(df[n], percs)
    qn_gamma = scipy.stats.gamma.ppf(percs / 100.0, *parameters)

    # Building a quantile biplot
    plt.figure(figsize=(10, 10))
    plt.plot(qn_first, qn_gamma, ls="", marker="o", markersize=6)
    x = np.linspace(np.min(df[n]), np.max(df[n]))
    plt.plot(x, x, color="k", ls="--")
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.xlabel(f'Empirical distribution')
    plt.ylabel('Theoretical (gamma) distribution')
    plt.show()

def QQPlotExp (n, m, a, b):
    from scipy.stats import expon
    # Calculation of quantiles
    parameters = scipy.stats.expon.fit(df[n])
    percs = np.linspace(0, 100, m)
    qn_first = np.percentile(df[n], percs)
    qn_expon = scipy.stats.expon.ppf(percs / 100.0, *parameters)

    # Building a quantile biplot
    plt.figure(figsize=(10, 10))
    plt.plot(qn_first, qn_expon, ls="", marker="o", markersize=6)
    x = np.linspace(np.min(df[n]), np.max(df[n]))
    plt.plot(x, x, color="k", ls="--")
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.xlabel(f'Empirical distribution')
    plt.ylabel('Theoretical (gamma) distribution')
    plt.show()

def QQPlotBeta (n, m, a, b):
    from scipy.stats import beta
    # Calculation of quantiles
    parameters = scipy.stats.beta.fit(df[n])
    percs = np.linspace(0, 100, m)
    qn_first = np.percentile(df[n], percs)
    qn_beta = scipy.stats.beta.ppf(percs / 100.0, *parameters)

    # Building a quantile biplot
    plt.figure(figsize=(10, 10))
    plt.plot(qn_first, qn_beta, ls="", marker="o", markersize=6)
    x = np.linspace(np.min(df[n]), np.max(df[n]))
    plt.plot(x, x, color="k", ls="--")
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.xlabel(f'Empirical distribution')
    plt.ylabel('Theoretical (gamma) distribution')
    plt.show()

def QQPlotPareto (n, m, a, b):
    from scipy.stats import pareto
    # Calculation of quantiles
    parameters = scipy.stats.pareto.fit(df[n])
    percs = np.linspace(0, 100, m)
    qn_first = np.percentile(df[n], percs)
    qn_pareto = scipy.stats.pareto.ppf(percs / 100.0, *parameters)

    # Building a quantile biplot
    plt.figure(figsize=(10, 10))
    plt.plot(qn_first, qn_pareto, ls="", marker="o", markersize=6)
    x = np.linspace(np.min(df[n]), np.max(df[n]))
    plt.plot(x, x, color="k", ls="--")
    plt.xlim(a, b)
    plt.ylim(a, b)
    plt.xlabel(f'Empirical distribution')
    plt.ylabel('Theoretical (Pareto) distribution')
    plt.show()


def TestLognorm (n):
    from scipy.stats import lognorm
    # Calculation of the Kolmogorov-Smirnov test and chi-square
    parameters = scipy.stats.lognorm.fit(df[n])
    ks = scipy.stats.kstest(df[n], 'lognorm', parameters, N=100)
    cm = scipy.stats.cramervonmises(df[n], 'lognorm', parameters)
    print(ks)
    print(cm)

def TestGamma (n):
    # Calculation of the Kolmogorov-Smirnov test and chi-square
    parameters = scipy.stats.gamma.fit(df[n])
    ks = scipy.stats.kstest(df[n], 'gamma', parameters, N=100)
    cm = scipy.stats.cramervonmises(df[n], 'gamma', parameters)
    print(ks)
    print(cm)

def TestExp (n):
    # Calculation of the Kolmogorov-Smirnov test and chi-square
    parameters = scipy.stats.expon.fit(df[n])
    ks = scipy.stats.kstest(df[n], 'expon', parameters, N=100)
    cm = scipy.stats.cramervonmises(df[n], 'expon', parameters)
    print(ks)
    print(cm)

def TestBeta (n):
    from scipy.stats import beta
    # Calculation of the Kolmogorov-Smirnov test and chi-square
    parameters = scipy.stats.beta.fit(df[n])
    ks = scipy.stats.kstest(df[n], 'beta', parameters, N=100)
    cm = scipy.stats.cramervonmises(df[n], 'beta', parameters)
    print(ks)
    print(cm)

def TestPareto (n):
    from scipy.stats import pareto
    parameters = scipy.stats.pareto.fit(df[n])
    ks = scipy.stats.kstest(df[n], 'pareto', parameters, N=100)
    cm = scipy.stats.cramervonmises(df[n], 'pareto', parameters)
    print(ks)
    #print(cm)

def TestVeibulo (n):
    from scipy.stats import exponweib
    parameters = scipy.stats.exponweib.fit(df[n])
    ks = scipy.stats.kstest(df[n], 'exponweib', parameters, N=100)
    cm = scipy.stats.cramervonmises(df[n], 'exponweib', parameters)
    print(ks)
    print(cm)

path_to_file = 'avocado.csv'
data = pd.read_csv(r'C:\Users\Ксения\Documents\Учеба\ИТМО\1 семестр\Методы\Лаб 1\Код Лаб 1\avocado.csv')
df = pd.DataFrame(data,columns= ['AveragePrice', 'Total Volume', '4046'])


#PDF('Total Volume')
#BoxWithWhiskers('Total Volume')
#MaximumLikelihood('4046')
#LeastSquaresMethod('4046')

#QQPlotLognorm('AveragePrice', 21, 0, 4)
#QQPlotLognorm('Total Volume', 1000, 0, 10000000)
#QQPlotLognorm('4046', 100, 0, 10000)

#TestLognorm('AveragePrice')
#TestLognorm('Total Volume')
#TestLognorm('4046')

#QQPlotGamma('AveragePrice', 21, 0, 4)
#QQPlotGamma('Total Volume', 1000, 0, 100000000)
#QQPlotGamma('4046', 100, 0, 100000000)
"""
TestGamma('AveragePrice')
TestGamma('Total Volume')
TestGamma('4046')

QQPlotExp('AveragePrice', 21, 0, 4)
QQPlotExp('Total Volume', 1000, 0, 10000000)
QQPlotExp('4046', 1000, 0, 4000)
TestExp('AveragePrice')
TestExp('Total Volume')
TestExp('4046')

QQPlotBeta('AveragePrice', 21, 0, 4)
QQPlotBeta('Total Volume', 1000, 0, 10000000)
QQPlotBeta('4046', 100, 0, 10000)
"""
#TestBeta('AveragePrice')
#TestBeta('Total Volume')
#TestBeta('4046')

QQPlotPareto('Total Volume', 1000, 0, 10000)
TestPareto('Total Volume')

#TestVeibulo('Total Volume')





