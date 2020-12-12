import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as ex
import plotly.graph_objects as go

def generate_data(N, mean, prec):
    return np.random.normal(mean, np.sqrt(1/prec), N)

def calc_likelihood(d, mu, tau):
    """
    Calculate likelihood p(d | mu, tau)
    """
    n = len(d)
    sum_d = np.sum((d-mu) ** 2)
    likelihood = (tau / (2*np.pi)) ** (n/2) * np.exp((-tau/2) * sum_d)
    return likelihood

def Exp_mu(data, mu_0, mu_n, lmbda_0, lmbda_n):
    """
    Calculates the E[mu] as the equation 10.30 in the Bishop's book.
    """
    exp_mu = 1/lmbda_n + mu_n ** 2
    ss = np.sum((data**2 - 2*data*mu_n) + exp_mu)
    exp_mu = ss / 2 + lmbda_0 * ((mu_0**2 - 2*mu_0*mu_0) + exp_mu)
    return exp_mu

def infer_alpha(a, n):
    """
    Calculates a_n
    """
    return a + (n+1) / 2

def infer_beta(data, mu_0, mu_n, lmbda_0, lmbda_n, b_0):
    """
    Calculates b_n
    """
    b_n = b_0 + Exp_mu(data, mu_0, mu_n, lmbda_0, lmbda_n)
    return b_n

def infer_mu(n, mean, mu_0, lmbda_0):
    """
    Calculates mu_n
    """
    return lmbda_0*mu_0 + n * mean / (lmbda_0+n)

def infer_lmbda(n, lmbda_0, a_n, b_n):
    """
    Calculates lambda_n
    """
    lmbda_n = (lmbda_0 + n) * (a_n / b_n)
    return lmbda_n

def VI(data, ite=1):
    """
    Variational inference algorithm.
    Approximates mu, lambda, alpha and beta for ite iterations.
    """
    mu_0, mu_n = 0, 0
    lmbda_0, lmbda_n = 0, 1
    a_0, a_n = 0, 0
    b_0, b_n = 0, 1

    n = len(data)
    mean = np.mean(data)
    a_n = infer_alpha(a_0, n)
    mu_n = infer_mu(n, mean, mu_0, lmbda_0)

    for _ in range(ite):
        b_n = infer_beta(data, mu_0, mu_n, lmbda_0, lmbda_n, b_0)
        lmbda_n = infer_lmbda(n, lmbda_0, a_n, b_n)

    return mu_n, lmbda_n, a_n, b_n

def q_mu(x, mu, lmbda):
    return stats.norm.pdf(x, mu, np.sqrt(1.0 / lmbda))


def q_tau(tau, a, b):
    return stats.gamma.pdf(tau, a, loc=0, scale=(1.0 / b))


def plot(data, ite, mu, lmbda, a, b, t_mu, t_lmbda, t_a, t_b, t_prec):
    """
    For plotting the comparison graph between the inferred posterior and the true posterior
    """
    mu_vals = np.linspace(t_mu - 0.5, t_mu + 0.5, 100)
    tau_vals = np.linspace(t_prec - 0.75, t_prec + 0.75, 100)
    t_mu_vals = np.linspace(t_mu - 0.5, t_mu + 0.5, 100)
    t_tau_vals = np.linspace(t_prec - 0.75, t_prec + 0.75, 100)
    
    M, T = np.meshgrid(mu_vals, tau_vals, indexing="ij")
    Mtrue, Ttrue = np.meshgrid(t_mu_vals, t_tau_vals, indexing="ij")
    
    Z = np.zeros_like(M)
    Ztrue = np.zeros_like(Mtrue)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i][j] = q_mu(mu_vals[i], mu, lmbda) * q_tau(tau_vals[j], a, b)

    for i in range(Ztrue.shape[0]):
        for j in range(Ztrue.shape[1]):
            Ztrue[i][j] = q_mu(t_mu_vals[i], t_mu, t_lmbda * t_tau_vals[j]) * q_tau(t_tau_vals[j], t_a, t_b) * calc_likelihood(data, t_mu_vals[i], t_tau_vals[j])

    custom_lines = [Line2D([0], [0], color="red", lw=4),
                    Line2D([0], [0], color="green", lw=4)]

    _, ax = plt.subplots()
    ax.legend(custom_lines, ['Approximated', 'True Posterior'])

    plt.contour(M, T, Z, 5, colors="red")
    plt.contour(Mtrue, Ttrue, Ztrue, 5, colors="green")
    plt.xlabel("Mean")
    plt.ylabel("Precision")
    plt.title("Approximated vs True \nN: " + str(len(data)) + ", "
              "Iterations: " + str(ite) + "\n(mu, lambda, a, b) = ("
              + str(t_mu) + ", " + str(t_lmbda) + ", " + str(t_a) + ", " + str(t_b) + ")")
    plt.show()



def main():
    SEED = 2020
    ITE = 5

    N = 100
    t_mu = 3
    t_lmbda = 10
    t_a = 20
    t_b = 5
    t_prec = t_a/t_b
    
    np.random.seed(SEED)
    data = generate_data(N, t_mu, t_prec)

    mu, lmbda, a, b = VI(data, ite=ITE)

    print("True posterior hyperparameters:\n", t_mu, t_lmbda, t_a, t_b)
    print("Inferred posterior hyperparameters:\n", mu, lmbda, a, b)
    print("-"*70)
    plot(data, ITE, mu, lmbda, a, b, t_mu, t_lmbda, t_a, t_b, t_prec)

if __name__ == "__main__":
    main()
        
