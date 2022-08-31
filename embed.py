import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections.abc import Iterable
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go


def plotly_3d(X, title="", mode='markers', size=(500, 500)):

    fig = go.Figure(data=go.Scatter3d(
        x=X[0], y=X[1], z=X[2],
        mode=mode,
        marker=dict(
            size=1,
            color=np.arange(0, len(X[0])),
            colorscale='Viridis',
        ),
    ))
    fig.update_layout(
        width=size[0],
        height=size[1],
        title=title
    )
    fig.show()


def random_walk_coords(n):

    x = np.zeros(n).astype(int)
    y = np.zeros(n).astype(int)

    # filling the coordinates with random variables
    for i in range(1, n):
        val = random.randint(1, 4)
        if val == 1:
            x[i] = x[i - 1] + 1
            y[i] = y[i - 1]
        elif val == 2:
            x[i] = x[i - 1] - 1
            y[i] = y[i - 1]
        elif val == 3:
            x[i] = x[i - 1]
            y[i] = y[i - 1] + 1
        else:
            x[i] = x[i - 1]
            y[i] = y[i - 1] - 1

    return x, y


def rand_sgn():
    return 1 if random.random() < 0.5 else -1


def rand_float(low, high):
    return random.random()*(high-low) + low


def gaussian(N, mu=0, sigma=0.1, normalize=True):
    x = np.linspace(-1, 1, N)
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp(- (x - mu)**2 / (2 * sigma**2))

    if normalize:
        y /= np.max(y)

    return y


# FROM https://github.com/sethhirsh/sHAVOK/blob/master/Figure%201.ipynb
def build_hankel(data, rows, cols=None):
    if cols is None:
        cols = len(data) - rows
    X = np.empty((rows, cols))
    for k in range(rows):
        X[k, :] = data[k:cols + k]
    return X


def multi_build_hankel(X_ts, N):
    Hs = []
    M = len(X_ts)
    L = len(X_ts[0])-N
    for data in X_ts:
        H = build_hankel(data, N, L)
        Hs.append(H)

    multi_H = np.zeros((L, N*M))
    for m in range(M):
        H = Hs[m]
        for n in range(N):
            multi_H[:, m*n] = H[n, :]

    return multi_H


def tanh(t, a=1, lam=1, center=0):
    return a/(1+np.exp(lam*-(t+center)))


def svd_embedding(data, N, D, center=False):

    # bulid time delay hankel matrix
    H = build_hankel(data, N)

    # center if necessary
    if center:
        H -= H[H.shape[0]//2]

    # svd of hankel
    u, s, v = np.linalg.svd(H, full_matrices=False)

    # take first D singular vectors
    H_hat = v[:D]

    return H_hat


def plot_embed(data, Ns, hankel=False, sz=(20, 14), c=None, line=False, center=False, square_plot=False):

    num_sl = len(Ns)
    r = math.ceil(math.sqrt(num_sl))

    fig, axarr = plt.subplots(r, r)
    fig.set_size_inches(sz[0], sz[1])
    if square_plot:
        fig.set_size_inches(sz[1], sz[1])

    if isinstance(axarr, Iterable):
        axs1 = [item for sublist in axarr for item in sublist]
    else:
        axs1 = [axarr]

    for i, N in enumerate(Ns):

        if hankel:
            H = data
        else:
            H = build_hankel(data, N, len(data)-N)

        if center:
            H -= H[H.shape[0]//2]

        u, s, v = np.linalg.svd(H, full_matrices=False)

        H_hat = v[:2]

        # H_hat[0] -= np.mean(H_hat[0])
        # H_hat[1] -= np.mean(H_hat[1])
        # H_hat[0] /= np.max(H_hat[0])
        # H_hat[1] /= np.max(H_hat[1])
        L = len(H_hat[0])

        # plt.plot(s[0:10], '.-',)
        # plt.title("Hankel Singular Values")
        # plt.show()

        # plt.plot(u[:, :3])
        # plt.show()

        # plt.plot(v[0])
        # plt.plot(v[1])
        # plt.plot(trajectory[:,1])
        # plt.plot(trajectory[:,0])
        # plt.show()
        if c is None:
            c = np.arange(0, L)
        c = c[:L]

        scatter = axs1[i].scatter(H_hat[0], H_hat[1], c=c)

        if square_plot:
            mi, mx = np.min(H_hat), np.max(H_hat)
            plt.xlim([mi, mx])
            plt.ylim([mi, mx])

        if line:
            axs1[i].plot(H_hat[0], H_hat[1])
        axs1[i].set_title(f'N={N}')

    if c is not None:
        fig.colorbar(scatter)

    plt.show()


def lorentz(T=20, h=0.01):
    def init_XYZ(m):
        X = np.zeros(m)
        Y = np.zeros(m)
        Z = np.zeros(m)
        X[0] = -5.91652
        Y[0] = -5.52332
        Z[0] = 24.57231
        return (X, Y, Z)

    m = int(T/h)
    (X, Y, Z) = init_XYZ(m)

    for k in range(0, m-1):
        X[k+1] = X[k] + h*10*(Y[k]-X[k])
        Y[k+1] = Y[k] + h*((X[k]*(28-Z[k]))-Y[k])
        Z[k+1] = Z[k] + h*(X[k]*Y[k]-8*Z[k]/3)

    return np.array([X, Y, Z])


def lorentz_signal(T=20, h=0.01, noise=0.0):
    L = lorentz(T, h)

    obs_x = L[0, :]
    obs_y = L[1, :]
    obs_z = L[2, :]

    X = obs_x
    X = X + (noise * np.random.randn(len(X)))

    return X


def exp_f(x, e, a): return a*np.exp(x*e)


def build_exp_series(a_s, e_s, noise=0.0, time=np.arange(0, 3, 0.1)):
    components = []
    X = np.zeros_like(time)
    for i in range(len(e_s)):
        c = exp_f(time, e_s[i], a_s[i])
        c = c.real
        X += c
        components.append(c)

    X *= (1+noise * np.random.randn(len(X)))
    y_i = np.argmax(np.abs(e_s.real))
    Y = components[y_i]

    return X, Y, time, components
