"""Reusable plotting and simulation helpers for the stochastic sampling notebook.

This module turns the original notebook cells into callable functions so the
notebook can keep short, readable cells that delegate the implementation here.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import matplotlib
import numpy as np
from IPython.display import clear_output, display
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.pyplot import *
from numpy import *
from scipy.stats import norm

try:
    import pymc as pm
except ImportError:  # pragma: no cover - optional dependency in this project
    pm = None


def render_figure(fig, *, close_figure=True):
    """Display a matplotlib figure using the notebook-friendly pattern."""
    display(fig)
    if close_figure:
        close(fig)
    return fig


def animated_frame(fig, pause=1.5):
    """Display an animation frame in-place and optionally sleep."""
    clear_output(wait=True)
    display(fig)
    if pause:
        time.sleep(pause)


def get_demo_grid(points=100, xmin=-5, xmax=5):
    """Return a standard x-grid used by the early 1D examples."""
    return linspace(xmin, xmax, points, endpoint=False)


def f(x):
    """Mixture of Gaussians used for integral examples."""
    return exp(-(x - 1) ** 2) + exp(-(x + 2) ** 2) / 2.0


def g(x):
    """Narrow Gaussian used to motivate importance sampling."""
    return exp(-100 * (x - 1) ** 2)


def _plot_bar_samples(ax, sample_points, func, width, facecolor="#aa3333"):
    for sample in sample_points:
        ax.add_patch(
            Rectangle(
                (sample - width / 2.0, 0.0),
                width,
                func(sample),
                facecolor=facecolor,
            )
        )


def plot_integral_function(points=100, xmin=-5, xmax=5):
    """Plot the target function whose integral we want to estimate."""
    x = get_demo_grid(points=points, xmin=xmin, xmax=xmax)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f(x))
    ax.text(-4.5, 0.9, r"$\int_{-\infty}^\infty f(x) \, \mathrm{d}x$", fontsize=12)
    render_figure(fig)
    return fig


def plot_regular_sampling(samples=40, xmin=-5, xmax=5, points=100):
    """Show regular-bin integration for the function f."""
    x = get_demo_grid(points=points, xmin=xmin, xmax=xmax)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f(x))
    ax.text(-4.5, 0.9, r"$\sum_{-5}^5 \,\Delta x f(x_i)$", fontsize=12)
    width = (xmax - xmin) / samples
    sample_points = linspace(xmin, xmax, samples, endpoint=False)
    _plot_bar_samples(ax, sample_points, f, width)
    render_figure(fig)
    approximation = sum(f(sample_points)) * (xmax - xmin) / samples
    print("Value of Integral = ", 1.5 * sqrt(pi))
    print("Approximation = ", approximation)
    return fig, approximation


def plot_uniform_sampling(samples=200, xmin=-5, xmax=5, points=100, seed=None):
    """Show Monte Carlo integration with uniformly distributed samples."""
    if seed is not None:
        random.seed(seed)
    x = get_demo_grid(points=points, xmin=xmin, xmax=xmax)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f(x))
    ax.text(-4.5, 0.8, r"$\frac{\mathrm{range}}{N} \sum_i \,f(x_i)$", fontsize=12)
    width = 0.1
    sample_points = random.uniform(xmin, xmax, samples)
    _plot_bar_samples(ax, sample_points, f, width)
    render_figure(fig)
    approximation = sum(f(sample_points)) * (xmax - xmin) / samples
    print("N = number of samples = ", samples)
    print("Value of Integral = ", 1.5 * sqrt(pi))
    print("Approximation = ", approximation)
    return fig, approximation


def plot_narrow_function_uniform_sampling(samples=200, xmin=-5, xmax=5, points=100, seed=None):
    """Show the failure mode of uniform sampling on a narrow target."""
    if seed is not None:
        random.seed(seed)
    x = get_demo_grid(points=points, xmin=xmin, xmax=xmax)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, g(x))
    ax.text(-4.5, 0.8, r"$\frac{\mathrm{range}}{N}\sum_i \,f(x_i)$", fontsize=12)
    ax.set_xlim((xmin, xmax))
    width = 0.05
    sample_points = random.uniform(xmin, xmax, samples)
    _plot_bar_samples(ax, sample_points, g, width)
    render_figure(fig)
    approximation = sum(g(sample_points)) * (xmax - xmin) / samples
    print("samples =", samples)
    print("Value of Integral = ", sqrt(pi / 100))
    print("Approximation = ", approximation)
    return fig, approximation


def plot_importance_sampling(samples=200, mean=1.0, scale=0.2, points=100, seed=None):
    """Show importance sampling with a Gaussian proposal."""
    if seed is not None:
        random.seed(seed)
    x = get_demo_grid(points=points)
    proposal = norm(loc=mean, scale=scale)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, g(x))
    ax.plot(x, proposal.pdf(x) / 2.0, "r")
    ax.text(-4.5, 0.8, r"$\frac{\mathrm{range}}{N}\sum_i w_if(x_i)$", fontsize=12)
    width = 0.05
    sample_points = proposal.rvs(size=samples)
    _plot_bar_samples(ax, sample_points, g, width)
    render_figure(fig)
    approximation = 1.0 / samples * sum(g(sample_points) / proposal.pdf(sample_points))
    print("samples = ", samples)
    print("Value of Integral = ", sqrt(pi / 100))
    print("Approximation = ", approximation)
    return fig, approximation


def plot_ideal_importance_sampling(samples=1, points=100, seed=None):
    """Show the near-ideal proposal for the narrow Gaussian example."""
    if seed is not None:
        random.seed(seed)
    x = get_demo_grid(points=points)
    proposal = norm(loc=1, scale=sqrt(0.005))
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, g(x))
    ax.plot(x, proposal.pdf(x) / 6, "r")
    ax.text(-4.5, 0.8, r"$\frac{\mathrm{range}}{N}\sum_i w_if(x_i)$", fontsize=12)
    width = 1.0 / samples
    sample_points = proposal.rvs(size=samples)
    _plot_bar_samples(ax, sample_points, g, width)
    render_figure(fig)
    approximation = 1.0 / samples * sum(g(sample_points) / proposal.pdf(sample_points))
    print("samples = ", samples)
    print("Value of Integral = ", sqrt(pi / 100))
    print("Approximation = ", approximation)
    return fig, approximation


def animate_rejection_sampling(samples=50, xmin=-5, xmax=5, seed=None, pause=0.1):
    """Animate rejection sampling on the function f."""
    if seed is not None:
        random.seed(seed)
    x = get_demo_grid()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f(x))
    sample_points = random.uniform(xmin, xmax, samples)
    sample_heights = random.uniform(0, 1.2, samples)

    for index in range(samples):
        facecolor = "red" if sample_heights[index] > f(sample_points[index]) else "green"
        ax.add_patch(
            matplotlib.patches.Ellipse(
                (sample_points[index], sample_heights[index]),
                0.2,
                0.03,
                facecolor=facecolor,
                edgecolor="none",
            )
        )
        animated_frame(fig, pause=pause)

    print("samples = ", samples)
    close(fig)
    return fig


def build_metropolis_patch_collection(sample_points, heights, colors):
    """Build a patch collection for the multi-particle Metropolis demo."""
    dots = []
    for index in range(sample_points.size):
        dots.append(
            matplotlib.patches.Ellipse(
                (sample_points[index], heights[index] * f(sample_points[index])),
                0.2,
                0.03,
                facecolor=colors[index],
                edgecolor="none",
            )
        )
    return PatchCollection(dots, match_original=True)


def metropolis_step(sample_points, target_function=f, proposal_scale=1.0):
    """Single Metropolis update for all sample points."""
    delta = random.normal(0, proposal_scale, sample_points.size)
    proposed = sample_points + delta
    updated = sample_points.copy()
    for index in range(sample_points.size):
        if target_function(proposed[index]) < target_function(sample_points[index]):
            if random.rand() < target_function(proposed[index]) / target_function(sample_points[index]):
                updated[index] = proposed[index]
        else:
            updated[index] = proposed[index]
    return updated


def animate_metropolis_sampling(samples=100, n_frames=10, moves_per_frame=2, seed=None, pause=0.5):
    """Animate the many-particle Metropolis sampler converging on f."""
    if seed is not None:
        random.seed(seed)
    x = get_demo_grid()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f(x))

    sample_points = random.uniform(-5, 5, samples)
    heights = random.uniform(0, 1, samples)
    color_values = random.randint(0, 255, (samples, 3))
    colors = ["#%02X%02X%02X" % tuple(rgb) for rgb in color_values]

    for _ in range(n_frames):
        patch_collection = build_metropolis_patch_collection(sample_points, heights, colors)
        ax.add_collection(patch_collection)
        animated_frame(fig, pause=pause)
        for _ in range(moves_per_frame):
            sample_points = metropolis_step(sample_points)
        patch_collection.remove()

    print("samples = ", samples)
    close(fig)
    return fig


def animate_single_chain_metropolis(samples=1, sigma=1.0, n_steps=10, seed=None, pause=3.0):
    """Animate the single-chain Metropolis proposal/accept cycle."""
    if seed is not None:
        random.seed(seed)
    x = get_demo_grid()
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f(x))

    current = -1.0
    proposal = norm(loc=current, scale=sigma)
    current_patch = matplotlib.patches.Ellipse(
        (current, f(current)), 0.2, 0.03, facecolor="green", edgecolor="none"
    )
    ax.add_patch(current_patch)
    proposal_line = ax.plot(x, proposal.pdf(x), "g")
    animated_frame(fig, pause=2.0)

    for _ in range(n_steps):
        delta = random.normal(0, sigma, samples)
        candidate = current + delta
        candidate_patch = matplotlib.patches.Rectangle(
            (candidate - 0.1, f(candidate) - 0.02),
            0.2,
            0.04,
            facecolor="red",
            edgecolor="none",
        )
        ax.add_patch(current_patch)
        ax.add_patch(candidate_patch)
        animated_frame(fig, pause=pause)

        if f(candidate) < f(current):
            if random.rand() < f(candidate) / f(current):
                current = candidate
        else:
            current = candidate

        proposal_line.pop(0).remove()
        candidate_patch.remove()
        proposal = norm(loc=current, scale=sigma)
        current_patch = matplotlib.patches.Ellipse(
            (current, f(current)), 0.2, 0.03, facecolor="green", edgecolor="none"
        )
        ax.add_patch(current_patch)
        proposal_line = ax.plot(x, proposal.pdf(x), "g")
        animated_frame(fig, pause=pause)

    close(fig)
    return fig


def plot_metropolis_hastings_bias(points=100):
    """Illustrate the asymmetric proposal in the Metropolis-Hastings section."""
    x = get_demo_grid(points=points)
    fig = figure()
    ax = fig.add_subplot(111)
    ax.plot(x, f(x))

    current = 0.0
    proposal = norm(loc=current - 1, scale=1)
    current_patch = matplotlib.patches.Ellipse(
        (current, f(current)), 0.2, 0.03, facecolor="green", edgecolor="none"
    )
    ax.add_patch(current_patch)
    ax.plot(x, proposal.pdf(x), "g")
    fill_x = x[: int(x.size / 2 + 1)].copy()
    fill_y = proposal.pdf(fill_x)
    fill_x[-1] = fill_x[-2]
    fill_y[-1] = 0
    fill(fill_x, fill_y, facecolor="g", alpha=0.5)
    ax.annotate(
        "84%",
        xy=(-1.5, 0.2),
        xytext=(-4, 0.8),
        arrowprops=dict(facecolor="g", shrink=0.05),
        fontsize=16,
    )
    return render_figure(fig)


def load_text_message_data(path="txtdata.csv"):
    """Load the text-message count dataset used in the Poisson examples."""
    count_data = np.loadtxt(path)
    return count_data, len(count_data)


def plot_count_data(path="txtdata.csv", figsize=(12.5, 3.5)):
    """Plot the text-message count time series."""
    count_data, n_count_data = load_text_message_data(path)
    fig = figure()
    fig.set_size_inches(*figsize)
    bar(np.arange(n_count_data), count_data, color="#348ABD")
    xlabel("Time (days)")
    ylabel("count of text-msgs received")
    xlim(0, n_count_data)
    return render_figure(fig), count_data


def require_pymc():
    """Raise a clear error when PyMC examples are called without PyMC."""
    if pm is None:
        raise ImportError("PyMC is required for the Bayesian count-data examples.")


def run_single_change_mcmc(count_data, draws=10000, tune=5000):
    """Fit the one-change-point Poisson model and return posterior samples."""
    require_pymc()
    n_count_data = len(count_data)
    with pm.Model() as model:
        alpha = 1.0 / count_data.mean()
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)
        tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)

        idx = np.arange(n_count_data)
        lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
        pm.Poisson("obs", lambda_, observed=count_data)

        step = pm.Metropolis()
        trace = pm.sample(draws, tune=tune, step=step, return_inferencedata=False)

    return {
        "trace": trace,
        "lambda_1_samples": trace["lambda_1"],
        "lambda_2_samples": trace["lambda_2"],
        "tau_samples": trace["tau"],
        "n_count_data": n_count_data,
    }


def hdi_of_mcmc(sample_vec, cred_mass=0.95):
    """Compute a highest density interval from MCMC samples."""
    assert len(sample_vec), "need points to find HDI"
    sorted_pts = sort(sample_vec)
    ci_idx_inc = int(floor(cred_mass * len(sorted_pts)))
    n_cis = len(sorted_pts) - ci_idx_inc
    ci_width = sorted_pts[ci_idx_inc:] - sorted_pts[:n_cis]
    min_idx = argmin(ci_width)
    hdi_min = sorted_pts[min_idx]
    hdi_max = sorted_pts[min_idx + ci_idx_inc]
    return hdi_min, hdi_max


def plot_hdi(ax, hdi):
    """Draw a highest density interval marker on an axis."""
    hdi_min, hdi_max = hdi
    hdi_line, = ax.plot([hdi_min, hdi_max], [0, 0], lw=5.0, color="k")
    hdi_line.set_clip_on(False)
    ax.text(hdi_min, -0.04, "%.3g" % hdi_min, horizontalalignment="center", verticalalignment="top", color="r")
    ax.text(hdi_max, -0.04, "%.3g" % hdi_max, horizontalalignment="center", verticalalignment="top", color="r")
    ax.text((hdi_min + hdi_max) / 2, 0.08, "95% HDI", horizontalalignment="center", verticalalignment="bottom")


def plot_single_change_posteriors(lambda_1_samples, lambda_2_samples, tau_samples, n_count_data):
    """Plot posterior marginals for the one-change-point Poisson model."""
    fig = figure()
    fig.set_size_inches(10, 6)

    ax = subplot(311)
    ax.set_autoscaley_on(False)
    hist(lambda_1_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label=r"posterior of $\lambda_1$", color="#A60628", density=True)
    plot_hdi(ax, hdi_of_mcmc(lambda_1_samples))
    legend(loc="upper right")
    title(r"""Posterior distributions of the variables
    $\lambda_1, \; \lambda_2, \; \tau$""")
    xlim([15, 30])
    xlabel(r"$\lambda_1$ value")
    ylabel("probability density")

    ax = subplot(312)
    ax.set_autoscaley_on(False)
    hist(lambda_2_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label=r"posterior of $\lambda_2$", color="#7A68A6", density=True)
    plot_hdi(ax, hdi_of_mcmc(lambda_2_samples))
    legend(loc="upper right")
    xlim([15, 30])
    xlabel(r"$\lambda_2$ value")
    ylabel("probability density")

    subplot(313)
    weights = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
    hist(tau_samples, bins=n_count_data, alpha=1, label=r"posterior of $\tau$",
         color="#467821", weights=weights, rwidth=2.0, width=0.4)
    xticks(np.arange(n_count_data))
    legend(loc="upper right")
    ylim([0, 0.75])
    xlim([35, n_count_data - 20])
    xlabel(r"$\tau$ (in days)")
    ylabel("probability")
    render_figure(fig)
    return fig


def run_three_segment_mcmc(count_data, draws=10000, tune=5000, seed=42):
    """Fit the three-rate Poisson change-point model and return posterior samples."""
    require_pymc()
    random.seed(seed)
    n_count_data = len(count_data)
    with pm.Model() as model:
        alpha = 1.0 / count_data.mean()
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)
        lambda_3 = pm.Exponential("lambda_3", alpha)
        tau_1 = pm.DiscreteUniform("tau_1", lower=0, upper=30)
        tau_2 = pm.DiscreteUniform("tau_2", lower=40, upper=n_count_data - 1)

        idx = np.arange(n_count_data)
        lambda__ = pm.math.switch(tau_1 > idx, lambda_1, lambda_2)
        lambda_ = pm.math.switch(tau_2 > idx, lambda__, lambda_3)
        pm.Poisson("obs", lambda_, observed=count_data)

        step = pm.Metropolis()
        trace = pm.sample(draws, tune=tune, step=step, return_inferencedata=False)

    return {
        "trace": trace,
        "lambda_1_samples": trace["lambda_1"],
        "lambda_2_samples": trace["lambda_2"],
        "lambda_3_samples": trace["lambda_3"],
        "tau_1_samples": trace["tau_1"],
        "tau_2_samples": trace["tau_2"],
        "n_count_data": n_count_data,
    }


def plot_three_segment_posteriors(lambda_1_samples, lambda_2_samples, lambda_3_samples,
                                  tau_1_samples, tau_2_samples, n_count_data):
    """Plot posterior marginals for the three-segment Poisson model."""
    fig = figure()
    fig.set_size_inches(10, 10)

    ax = subplot(511)
    ax.set_autoscaley_on(False)
    hist(lambda_1_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label=r"posterior of $\lambda_1$", color="#A60628", density=True)
    plot_hdi(ax, hdi_of_mcmc(lambda_1_samples))
    legend(loc="upper right")
    title(r"""Posterior distributions of the variables
    $\lambda_1, \; \lambda_2, \; \lambda_3, \; \tau_1, \; \tau_2$""")
    xlim([5, 30])
    xlabel(r"$\lambda_1$ value")
    ylabel("probability density")

    ax = subplot(512)
    ax.set_autoscaley_on(False)
    hist(lambda_2_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label=r"posterior of $\lambda_2$", color="#7A68A6", density=True)
    plot_hdi(ax, hdi_of_mcmc(lambda_2_samples))
    legend(loc="upper right")
    xlim([5, 30])
    xlabel(r"$\lambda_2$ value")
    ylabel("probability density")

    ax = subplot(513)
    ax.set_autoscaley_on(False)
    hist(lambda_3_samples, histtype="stepfilled", bins=30, alpha=0.85,
         label=r"posterior of $\lambda_3$", color="#3A68A6", density=True)
    plot_hdi(ax, hdi_of_mcmc(lambda_3_samples))
    legend(loc="upper right")
    xlim([5, 30])
    xlabel(r"$\lambda_3$ value")
    ylabel("probability density")

    subplot(514)
    weights = 1.0 / tau_1_samples.shape[0] * ones_like(tau_1_samples)
    hist(tau_1_samples, bins=n_count_data, alpha=1, label=r"posterior of $\tau_1$",
         color="#467821", weights=weights, rwidth=2.0, width=0.4)
    xticks(arange(n_count_data))
    legend(loc="upper right")
    ylim([0, 1])
    xlim([20, 40])
    xlabel(r"$\tau_1$ (in days)")
    ylabel("probability")

    subplot(515)
    weights = 1.0 / tau_2_samples.shape[0] * ones_like(tau_2_samples)
    hist(tau_2_samples, bins=n_count_data, alpha=1, label=r"posterior of $\tau_2$",
         color="#663821", weights=weights, rwidth=2.0, width=0.4)
    xticks(arange(n_count_data))
    legend(loc="upper right")
    ylim([0, 1])
    xlim([35, 55])
    xlabel(r"$\tau_2$ (in days)")
    ylabel("probability")
    render_figure(fig)
    return fig


def kalman_smoother_up_to(t_end, obs, proc_noise, obs_noise_sd):
    """Kalman filter + RTS smoother on obs[0..t_end]."""
    horizon = t_end + 1
    m = zeros(horizon)
    P = zeros(horizon)
    m_pred = zeros(horizon)
    P_pred = zeros(horizon)

    P_pred[0] = 1.0
    K = P_pred[0] / (P_pred[0] + obs_noise_sd ** 2)
    m[0] = K * obs[0]
    P[0] = (1 - K) * P_pred[0]

    for step in range(1, horizon):
        m_pred[step] = m[step - 1]
        P_pred[step] = P[step - 1] + proc_noise ** 2
        K = P_pred[step] / (P_pred[step] + obs_noise_sd ** 2)
        m[step] = m_pred[step] + K * (obs[step] - m_pred[step])
        P[step] = (1 - K) * P_pred[step]

    ms = m.copy()
    Ps = P.copy()
    for step in range(horizon - 2, -1, -1):
        G = P[step] / P_pred[step + 1]
        ms[step] = m[step] + G * (ms[step + 1] - m_pred[step + 1])
        Ps[step] = P[step] + G ** 2 * (Ps[step + 1] - P_pred[step + 1])

    return ms, sqrt(Ps)


def systematic_resample(weights, rng_module=random):
    """Systematic resampling indices for a vector of normalised weights."""
    cumulative = cumsum(weights)
    n_particles = weights.size
    u0 = rng_module.uniform(0, 1.0 / n_particles)
    u = u0 + arange(n_particles) / n_particles
    return searchsorted(cumulative, u)


def generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, d=1, seed=42):
    """Generate synthetic state-space data for the particle-filter demos."""
    random.seed(seed)
    if d == 1:
        true_x = zeros(n_timesteps)
        observations = zeros(n_timesteps)
        observations[0] = true_x[0] + random.normal(0, obs_noise)
        for step in range(1, n_timesteps):
            true_x[step] = true_x[step - 1] + random.normal(0, process_noise)
            observations[step] = true_x[step] + random.normal(0, obs_noise)
        return true_x, observations

    true_x = zeros((n_timesteps, d))
    observations = zeros((n_timesteps, d))
    observations[0] = true_x[0] + random.normal(0, obs_noise, d)
    for step in range(1, n_timesteps):
        true_x[step] = true_x[step - 1] + random.normal(0, process_noise, d)
        observations[step] = true_x[step] + random.normal(0, obs_noise, d)
    return true_x, observations


def make_particle_colors(n_particles, seed=42):
    """Create repeatable pseudo-random colors for particles."""
    random.seed(seed)
    return array([
        "#%02X%02X%02X" % (random.randint(80, 220), random.randint(80, 220), random.randint(80, 220))
        for _ in range(n_particles)
    ])


@dataclass
class ParticleFilterHistory:
    rmse_history: list
    resample_times: list


def _set_panel_limits(ax, n_timesteps, y_min=-6, y_max=6):
    ax.set_xlim(-0.5, n_timesteps - 0.5)
    ax.set_ylim(y_min, y_max)


def _plot_weight_histogram(ax, weights, n_particles, ess, did_resample=False, title_suffix=""):
    sorted_weights = sort(weights)[::-1]
    bar_colors = ["limegreen" if did_resample else "steelblue"] * n_particles
    ax.bar(range(n_particles), sorted_weights, color=bar_colors, alpha=0.75)
    ax.axhline(1.0 / n_particles, color="orange", linestyle="--", linewidth=1.5, label="Uniform (1/N)")
    ax.set_xlabel("Particle (sorted by weight)")
    ax.set_ylabel("Weight")
    resample_text = "RESAMPLED ✓" if did_resample else "no resample"
    ax.set_title(f"Weight distribution  —  {resample_text}\nESS = {ess:.1f} / {n_particles}{title_suffix}")
    ax.set_ylim(0, 1.0)
    ax.text(
        0.35,
        0.75,
        f"Particles with >1% weight: {int(sum(weights > 0.01))}\nMax weight: {weights.max():.3f}",
        transform=ax.transAxes,
        bbox=dict(
            boxstyle="round,pad=0.3",
            facecolor="lightgreen" if did_resample else "lightblue",
            alpha=0.85,
        ),
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def _plot_rmse_panel(ax, rmse_history, n_timesteps, title, label=None, resample_times=None):
    ax.plot(range(len(rmse_history)), rmse_history, "r-o", markersize=5, linewidth=2, label=label)
    if resample_times:
        for resample_time in resample_times:
            ax.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
    ax.set_xlim(-0.5, n_timesteps - 0.5)
    ax.set_ylim(0, max(rmse_history) * 1.5 + 0.05)
    ax.set_xlabel("Time step  t")
    ax.set_ylabel(label or "RMSE")
    ax.set_title(title)
    if label is not None:
        ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def animate_sis_joint_demo(n_particles=50, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                           seed=42, pause=0.5, figsize=(14, 10)):
    """Animate SIS on the full 1D trajectory posterior without resampling."""
    fig, axes = subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flat
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, seed=seed)
    random.seed(seed)
    trajectories = zeros((n_particles, n_timesteps))
    trajectories[:, 0] = random.normal(0, 1.0, n_particles)
    log_weights = -0.5 * ((observations[0] - trajectories[:, 0]) / obs_noise) ** 2
    log_weights -= log_weights.max()
    weights = exp(log_weights)
    weights /= weights.sum()
    particle_colors = make_particle_colors(n_particles, seed=seed)
    rmse_history = []

    for step in range(n_timesteps):
        if step > 0:
            trajectories[:, step] = trajectories[:, step - 1] + random.normal(0, process_noise, n_particles)
            log_lik_t = -0.5 * ((observations[step] - trajectories[:, step]) / obs_noise) ** 2
            log_weights += log_lik_t
            log_weights -= log_weights.max()
            weights = exp(log_weights)
            weights /= weights.sum()

        ks_mean, ks_std = kalman_smoother_up_to(step, observations, process_noise, obs_noise)
        t_range = arange(step + 1)
        part_mean = array([sum(weights * trajectories[:, past_step]) for past_step in range(step + 1)])
        part_std = array([
            sqrt(sum(weights * (trajectories[:, past_step] - part_mean[past_step]) ** 2))
            for past_step in range(step + 1)
        ])
        rmse_history.append(sqrt(mean((part_mean - ks_mean) ** 2)))

        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
        max_w = weights.max()
        ax1.plot(true_x[: step + 1], "b-", linewidth=3, label="True state", zorder=5)
        ax1.plot(observations[: step + 1], "ro", markersize=6, label="Observations", zorder=5)
        for particle in range(n_particles):
            alpha = float(0.1 + 0.85 * (weights[particle] / max_w))
            linewidth = float(0.3 + 3.0 * (weights[particle] / max_w))
            ax1.plot(range(step + 1), trajectories[particle, : step + 1],
                     color=particle_colors[particle], alpha=alpha, linewidth=linewidth)
            ax1.scatter(
                step,
                trajectories[particle, step],
                s=15 + (weights[particle] / max_w) * 180,
                c=particle_colors[particle],
                zorder=4,
                edgecolors="black",
                linewidth=0.4,
                alpha=0.85,
            )
        ess = 1.0 / sum(weights ** 2)
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State")
        ax1.set_title(
            f"SIS — Joint  $p(x_{{0:{step}}} \\mid y_{{0:{step}}})$  (t = {step})\n"
            f"ESS = {ess:.1f} / {n_particles}  — no resampling"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_range, true_x[: step + 1], "b-", linewidth=3, zorder=5, label="True state")
        ax2.fill_between(t_range, ks_mean - ks_std, ks_mean + ks_std,
                         color="forestgreen", alpha=0.20, label="KS ±1σ  (true posterior)")
        ax2.plot(t_range, ks_mean, color="forestgreen", linewidth=2.5, label="KS mean  (true posterior)")
        ax2.fill_between(t_range, part_mean - part_std, part_mean + part_std,
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, part_mean, color="darkorange", linewidth=2, linestyle="--", label="Particle mean")
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State")
        ax2.set_title(f"Joint estimate quality  (t = {step})\nParticle mean vs True posterior")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=False)
        _plot_rmse_panel(ax4, rmse_history, n_timesteps,
                         "Estimation error over time\nRMSE grows as weight degeneracy increases")
        tight_layout()
        animated_frame(fig, pause=pause)

    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=[])


def animate_sis_optimal_joint_demo(n_particles=50, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                                   seed=42, pause=0.5, figsize=(14, 10)):
    """Animate SIS with the optimal proposal on the full 1D trajectory posterior."""
    fig, axes = subplots(2, 2, figsize=figsize)
    ax1, ax2, ax3, ax4 = axes.flat
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, seed=seed)
    random.seed(seed)
    trajectories = zeros((n_particles, n_timesteps))
    trajectories[:, 0] = random.normal(0, 1.0, n_particles)
    sigma_opt2 = 1.0 / (1.0 / process_noise ** 2 + 1.0 / obs_noise ** 2)
    sigma_opt = sqrt(sigma_opt2)
    sigma_pred2 = process_noise ** 2 + obs_noise ** 2
    log_weights = -0.5 * ((observations[0] - trajectories[:, 0]) / obs_noise) ** 2
    log_weights -= log_weights.max()
    weights = exp(log_weights)
    weights /= weights.sum()
    particle_colors = make_particle_colors(n_particles, seed=seed)
    rmse_history = []

    for step in range(n_timesteps):
        if step > 0:
            x_prev = trajectories[:, step - 1]
            mu_opt = sigma_opt2 * (x_prev / process_noise ** 2 + observations[step] / obs_noise ** 2)
            trajectories[:, step] = random.normal(mu_opt, sigma_opt)
            log_pred = -0.5 * ((observations[step] - x_prev) ** 2 / sigma_pred2)
            log_weights += log_pred
            log_weights -= log_weights.max()
            weights = exp(log_weights)
            weights /= weights.sum()

        ks_mean, ks_std = kalman_smoother_up_to(step, observations, process_noise, obs_noise)
        t_range = arange(step + 1)
        part_mean = array([sum(weights * trajectories[:, past_step]) for past_step in range(step + 1)])
        part_std = array([
            sqrt(sum(weights * (trajectories[:, past_step] - part_mean[past_step]) ** 2))
            for past_step in range(step + 1)
        ])
        rmse_history.append(sqrt(mean((part_mean - ks_mean) ** 2)))

        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
        max_w = weights.max()
        ax1.plot(true_x[: step + 1], "b-", linewidth=3, label="True state", zorder=5)
        ax1.plot(observations[: step + 1], "ro", markersize=6, label="Observations", zorder=5)
        for particle in range(n_particles):
            alpha = float(0.1 + 0.85 * (weights[particle] / max_w))
            linewidth = float(0.3 + 3.0 * (weights[particle] / max_w))
            ax1.plot(range(step + 1), trajectories[particle, : step + 1],
                     color=particle_colors[particle], alpha=alpha, linewidth=linewidth)
            ax1.scatter(
                step,
                trajectories[particle, step],
                s=15 + (weights[particle] / max_w) * 180,
                c=particle_colors[particle],
                zorder=4,
                edgecolors="black",
                linewidth=0.4,
                alpha=0.85,
            )
        ess = 1.0 / sum(weights ** 2)
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State")
        ax1.set_title(f"SIS — Optimal proposal  (t = {step})\nESS = {ess:.1f} / {n_particles}  — no resampling")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_range, true_x[: step + 1], "b-", linewidth=3, zorder=5, label="True state")
        ax2.fill_between(t_range, ks_mean - ks_std, ks_mean + ks_std,
                         color="forestgreen", alpha=0.20, label="KS ±1σ  (true posterior)")
        ax2.plot(t_range, ks_mean, color="forestgreen", linewidth=2.5, label="KS mean  (true posterior)")
        ax2.fill_between(t_range, part_mean - part_std, part_mean + part_std,
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, part_mean, color="darkorange", linewidth=2, linestyle="--", label="Particle mean")
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State")
        ax2.set_title(f"Joint estimate quality  (t = {step})\nParticle mean vs True posterior")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=False)
        _plot_rmse_panel(ax4, rmse_history, n_timesteps, "Estimation error over time\nRMSE with optimal proposal")
        tight_layout()
        animated_frame(fig, pause=pause)

    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=[])


def _configure_pf_figure(figsize):
    fig, axes = subplots(2, 2, figsize=figsize)
    return fig, axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]


def animate_optimal_resampling_demo(n_particles=50, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                                    seed=42, pause=0.5, figsize=(14, 10)):
    """Animate the 1D optimal-proposal filter with systematic resampling."""
    fig, ax1, ax2, ax3, ax4 = _configure_pf_figure(figsize)
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, seed=seed)
    random.seed(seed)
    lineage = zeros((n_particles, n_timesteps))
    particles = random.normal(0, 1.0, n_particles)
    lineage[:, 0] = particles.copy()
    sigma_opt2 = 1.0 / (1.0 / process_noise ** 2 + 1.0 / obs_noise ** 2)
    sigma_opt = sqrt(sigma_opt2)
    sigma_pred2 = process_noise ** 2 + obs_noise ** 2
    log_w = -0.5 * ((observations[0] - particles) / obs_noise) ** 2
    log_w -= log_w.max()
    weights = exp(log_w)
    weights /= weights.sum()
    colors = make_particle_colors(n_particles, seed=seed)
    resample_times = []
    rmse_history = []
    resample_thresh = n_particles / 2

    for step in range(n_timesteps):
        did_resample = False
        if step > 0:
            x_prev = particles.copy()
            ess = 1.0 / sum(weights ** 2)
            if ess < resample_thresh:
                idx = systematic_resample(weights)
                lineage[:, :step] = lineage[idx, :step]
                x_prev = x_prev[idx]
                colors = colors[idx]
                weights = ones(n_particles) / n_particles
                did_resample = True
                resample_times.append(step)
            mu_opt = sigma_opt2 * (x_prev / process_noise ** 2 + observations[step] / obs_noise ** 2)
            particles = random.normal(mu_opt, sigma_opt)
            lineage[:, step] = particles
            log_incr = -0.5 * ((observations[step] - x_prev) ** 2 / sigma_pred2)
            log_incr -= log_incr.max()
            weights = weights * exp(log_incr)
            weights /= weights.sum()

        ks_mean, ks_std = kalman_smoother_up_to(step, observations, process_noise, obs_noise)
        t_range = arange(step + 1)
        part_mean = array([sum(weights * lineage[:, past_step]) for past_step in range(step + 1)])
        part_std = array([
            sqrt(sum(weights * (lineage[:, past_step] - part_mean[past_step]) ** 2))
            for past_step in range(step + 1)
        ])
        rmse_history.append(sqrt(mean((part_mean - ks_mean) ** 2)))

        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
        max_w = weights.max()
        ax1.plot(true_x[: step + 1], "b-", linewidth=3, label="True state", zorder=5)
        ax1.plot(observations[: step + 1], "ro", markersize=6, label="Observations", zorder=5)
        for particle in range(n_particles):
            alpha = float(0.08 + 0.87 * (weights[particle] / max_w))
            linewidth = float(0.3 + 2.7 * (weights[particle] / max_w))
            ax1.plot(range(step + 1), lineage[particle, : step + 1],
                     color=colors[particle], alpha=alpha, linewidth=linewidth)
            ax1.scatter(step, particles[particle], s=15 + (weights[particle] / max_w) * 170,
                        c=colors[particle], zorder=4, edgecolors="black", linewidth=0.4, alpha=0.9)
        for resample_time in resample_times:
            ax1.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        if resample_times:
            ax1.axvline(resample_times[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.8,
                        label="Resampling event")
        ess = 1.0 / sum(weights ** 2)
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State")
        ax1.set_title(
            f"SIS — Optimal proposal + resampling  (t = {step})\n"
            f"ESS = {ess:.1f} / {n_particles}  — threshold = N/2 = {int(resample_thresh)}"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_range, true_x[: step + 1], "b-", linewidth=3, zorder=5, label="True state")
        ax2.fill_between(t_range, ks_mean - ks_std, ks_mean + ks_std,
                         color="forestgreen", alpha=0.20, label="KS ±1σ  (true posterior)")
        ax2.plot(t_range, ks_mean, color="forestgreen", linewidth=2.5, label="KS mean  (true posterior)")
        ax2.fill_between(t_range, part_mean - part_std, part_mean + part_std,
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, part_mean, color="darkorange", linewidth=2, linestyle="--", label="Particle mean")
        for resample_time in resample_times:
            ax2.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State")
        ax2.set_title(f"Joint estimate quality  (t = {step})\nParticle mean vs True posterior")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=did_resample)
        _plot_rmse_panel(ax4, rmse_history, n_timesteps,
                         "Estimation error over time\nRMSE with optimal proposal + resampling",
                         resample_times=resample_times)
        tight_layout()
        animated_frame(fig, pause=pause)

    print(f"\nResampling occurred at timesteps: {resample_times}")
    print(f"Final ESS: {1.0 / sum(weights ** 2):.1f} / {n_particles}")
    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=resample_times)


def animate_prior_resampling_demo(n_particles=50, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                                  seed=42, pause=0.5, figsize=(14, 10)):
    """Animate the 1D prior-proposal filter with systematic resampling."""
    fig, ax1, ax2, ax3, ax4 = _configure_pf_figure(figsize)
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, seed=seed)
    random.seed(seed)
    lineage = zeros((n_particles, n_timesteps))
    particles = random.normal(0, 1.0, n_particles)
    lineage[:, 0] = particles.copy()
    log_w = -0.5 * ((observations[0] - particles) / obs_noise) ** 2
    log_w -= log_w.max()
    weights = exp(log_w)
    weights /= weights.sum()
    colors = make_particle_colors(n_particles, seed=seed)
    resample_times = []
    rmse_history = []
    resample_thresh = n_particles / 2

    for step in range(n_timesteps):
        did_resample = False
        if step > 0:
            x_prev = particles.copy()
            ess = 1.0 / sum(weights ** 2)
            if ess < resample_thresh:
                idx = systematic_resample(weights)
                lineage[:, :step] = lineage[idx, :step]
                x_prev = x_prev[idx]
                colors = colors[idx]
                weights = ones(n_particles) / n_particles
                did_resample = True
                resample_times.append(step)
            particles = x_prev + random.normal(0, process_noise, n_particles)
            lineage[:, step] = particles
            log_lik_t = -0.5 * ((observations[step] - particles) / obs_noise) ** 2
            log_lik_t -= log_lik_t.max()
            weights = weights * exp(log_lik_t)
            weights /= weights.sum()

        ks_mean, ks_std = kalman_smoother_up_to(step, observations, process_noise, obs_noise)
        t_range = arange(step + 1)
        part_mean = array([sum(weights * lineage[:, past_step]) for past_step in range(step + 1)])
        part_std = array([
            sqrt(sum(weights * (lineage[:, past_step] - part_mean[past_step]) ** 2))
            for past_step in range(step + 1)
        ])
        rmse_history.append(sqrt(mean((part_mean - ks_mean) ** 2)))

        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()
        max_w = weights.max()
        ax1.plot(true_x[: step + 1], "b-", linewidth=3, label="True state", zorder=5)
        ax1.plot(observations[: step + 1], "ro", markersize=6, label="Observations", zorder=5)
        for particle in range(n_particles):
            alpha = float(0.08 + 0.87 * (weights[particle] / max_w))
            linewidth = float(0.3 + 2.7 * (weights[particle] / max_w))
            ax1.plot(range(step + 1), lineage[particle, : step + 1],
                     color=colors[particle], alpha=alpha, linewidth=linewidth)
            ax1.scatter(step, particles[particle], s=15 + (weights[particle] / max_w) * 170,
                        c=colors[particle], zorder=4, edgecolors="black", linewidth=0.4, alpha=0.9)
        for resample_time in resample_times:
            ax1.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        if resample_times:
            ax1.axvline(resample_times[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.8,
                        label="Resampling event")
        ess = 1.0 / sum(weights ** 2)
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State")
        ax1.set_title(
            f"SIS — Prior proposal + resampling  (t = {step})\n"
            f"ESS = {ess:.1f} / {n_particles}  — threshold = N/2 = {int(resample_thresh)}"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_range, true_x[: step + 1], "b-", linewidth=3, zorder=5, label="True state")
        ax2.fill_between(t_range, ks_mean - ks_std, ks_mean + ks_std,
                         color="forestgreen", alpha=0.20, label="KS ±1σ  (true posterior)")
        ax2.plot(t_range, ks_mean, color="forestgreen", linewidth=2.5, label="KS mean  (true posterior)")
        ax2.fill_between(t_range, part_mean - part_std, part_mean + part_std,
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, part_mean, color="darkorange", linewidth=2, linestyle="--", label="Particle mean")
        for resample_time in resample_times:
            ax2.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State")
        ax2.set_title(f"Joint estimate quality  (t = {step})\nParticle mean vs True posterior")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=did_resample)
        _plot_rmse_panel(ax4, rmse_history, n_timesteps,
                         "Estimation error over time\nRMSE with prior proposal + resampling",
                         resample_times=resample_times)
        tight_layout()
        animated_frame(fig, pause=pause)

    print(f"\nResampling occurred at timesteps: {resample_times}")
    print(f"Final ESS: {1.0 / sum(weights ** 2):.1f} / {n_particles}")
    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=resample_times)


def run_particle_filter_dimension_experiment(d, n_particles, n_timesteps, process_noise, obs_noise,
                                             resample_thresh_frac=0.5, seed=0, do_resample=True):
    """Run the high-dimensional optimal-proposal filter used in the dimension study."""
    rng = random.default_rng(seed)
    true_x = zeros((n_timesteps, d))
    for step in range(1, n_timesteps):
        true_x[step] = true_x[step - 1] + rng.normal(0, process_noise, d)
    obs = true_x + rng.normal(0, obs_noise, (n_timesteps, d))
    particles = rng.normal(0, 1.0, (n_particles, d))
    sigma_opt2 = 1.0 / (1.0 / process_noise ** 2 + 1.0 / obs_noise ** 2)
    sigma_opt = sqrt(sigma_opt2)
    sigma_pred2 = process_noise ** 2 + obs_noise ** 2
    log_w = -0.5 * sum(((obs[0] - particles) / obs_noise) ** 2, axis=1)
    log_w -= log_w.max()
    weights = exp(log_w)
    weights /= weights.sum()
    ess_history = [float(1.0 / sum(weights ** 2))]
    n_resamp = 0
    resample_threshold = resample_thresh_frac * n_particles

    for step in range(1, n_timesteps):
        x_prev = particles.copy()
        ess = 1.0 / sum(weights ** 2)
        if do_resample and ess < resample_threshold:
            idx = systematic_resample(weights, rng_module=rng)
            x_prev = x_prev[idx]
            weights = ones(n_particles) / n_particles
            n_resamp += 1
        mu_opt = sigma_opt2 * (x_prev / process_noise ** 2 + obs[step] / obs_noise ** 2)
        particles = rng.normal(mu_opt, sigma_opt)
        log_incr = -0.5 * sum(((obs[step] - x_prev) ** 2 / sigma_pred2), axis=1)
        log_incr -= log_incr.max()
        weights = weights * exp(log_incr)
        weights /= weights.sum()
        ess_history.append(float(1.0 / sum(weights ** 2)))

    return array(ess_history), n_resamp


def plot_curse_of_dimensionality_summary(n_particles=500, n_timesteps=20, n_runs=20,
                                         process_noise=0.5, obs_noise=1.0,
                                         dimensions=(1, 2, 4, 8, 16, 32),
                                         resample_thresh_frac=0.5,
                                         figsize=(10, 8)):
    """Summarise ESS collapse across state dimensions."""
    results = {}
    for dimension in dimensions:
        ess_runs = []
        resamp_runs = []
        for run in range(n_runs):
            ess_history, n_resamp = run_particle_filter_dimension_experiment(
                d=dimension,
                n_particles=n_particles,
                n_timesteps=n_timesteps,
                process_noise=process_noise,
                obs_noise=obs_noise,
                resample_thresh_frac=resample_thresh_frac,
                seed=run * 97 + dimension,
                do_resample=True,
            )
            ess_runs.append(ess_history)
            resamp_runs.append(n_resamp)
        results[dimension] = {
            "ess_mean": mean(ess_runs, axis=0),
            "ess_std": std(ess_runs, axis=0),
            "resamp_mean": mean(resamp_runs),
            "resamp_std": std(resamp_runs),
            "final_ess_mean": mean([ess[-1] for ess in ess_runs]),
            "final_ess_std": std([ess[-1] for ess in ess_runs]),
        }

    results_no_resample = {}
    for dimension in dimensions:
        ess_runs = []
        for run in range(n_runs):
            ess_history, _ = run_particle_filter_dimension_experiment(
                d=dimension,
                n_particles=n_particles,
                n_timesteps=n_timesteps,
                process_noise=process_noise,
                obs_noise=obs_noise,
                resample_thresh_frac=resample_thresh_frac,
                seed=run * 97 + dimension,
                do_resample=False,
            )
            ess_runs.append(ess_history)
        results_no_resample[dimension] = {
            "ess_mean": mean(ess_runs, axis=0),
            "ess_std": std(ess_runs, axis=0),
        }

    cmap = get_cmap("plasma")
    colors = {dimension: cmap(index / (len(dimensions) - 1)) for index, dimension in enumerate(dimensions)}
    t_range = arange(n_timesteps)
    fig, axes = subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    for dimension in dimensions:
        mean_ess = results_no_resample[dimension]["ess_mean"]
        std_ess = results_no_resample[dimension]["ess_std"]
        ax.plot(t_range, mean_ess, color=colors[dimension], linewidth=2, label=f"d = {dimension}")
        ax.fill_between(t_range, mean_ess - std_ess, mean_ess + std_ess, color=colors[dimension], alpha=0.15)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Effective Sample Size")
    ax.set_title(f"ESS over time — NO resampling (N = {n_particles})\nShaded = ±1 std over {n_runs} runs")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_timesteps - 1)
    ax.set_ylim(0, n_particles + 20)

    ax = axes[0, 1]
    for dimension in dimensions:
        mean_ess = results[dimension]["ess_mean"]
        std_ess = results[dimension]["ess_std"]
        ax.plot(t_range, mean_ess, color=colors[dimension], linewidth=2, label=f"d = {dimension}")
        ax.fill_between(t_range, mean_ess - std_ess, mean_ess + std_ess, color=colors[dimension], alpha=0.15)
    ax.axhline(resample_thresh_frac * n_particles, color="red", linestyle="--", linewidth=1.5,
               label="Resample threshold (N/2)")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Effective Sample Size")
    ax.set_title(f"ESS over time — WITH resampling (N = {n_particles})\nShaded = ±1 std over {n_runs} runs")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, n_timesteps - 1)
    ax.set_ylim(0, n_particles + 20)

    ax = axes[1, 0]
    final_means = array([results[dimension]["final_ess_mean"] for dimension in dimensions])
    final_stds = array([results[dimension]["final_ess_std"] for dimension in dimensions])
    ax.errorbar(dimensions, final_means, yerr=final_stds, fmt="o-", color="steelblue", linewidth=2,
                markersize=8, capsize=5, label="Final ESS (mean ± std)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(dimensions)
    ax.set_xticklabels([str(dimension) for dimension in dimensions])
    ax.set_ylim(0, n_particles + 20)
    ax.axhline(resample_thresh_frac * n_particles, color="red", linestyle="--", linewidth=1.5,
               label="Resample threshold")
    ax.axhline(1, color="gray", linestyle=":", linewidth=1.5, label="Complete degeneracy (ESS=1)")
    ax.set_xlabel("State dimension d  (log₂ scale)")
    ax.set_ylabel("Final Effective Sample Size")
    ax.set_title("Final ESS vs dimension (with resampling)\nExponential collapse as d grows")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    resamp_means = array([results[dimension]["resamp_mean"] for dimension in dimensions])
    resamp_stds = array([results[dimension]["resamp_std"] for dimension in dimensions])
    bars = ax.bar(range(len(dimensions)), resamp_means, color=[colors[dimension] for dimension in dimensions],
                  alpha=0.8, edgecolor="black", linewidth=0.7)
    ax.errorbar(range(len(dimensions)), resamp_means, yerr=resamp_stds, fmt="none", color="black", capsize=5,
                linewidth=1.5)
    ax.axhline(n_timesteps - 1, color="red", linestyle="--", linewidth=1.5,
               label=f"Max possible ({n_timesteps - 1})")
    ax.set_xticks(range(len(dimensions)))
    ax.set_xticklabels([f"d={dimension}" for dimension in dimensions])
    ax.set_xlabel("State dimension")
    ax.set_ylabel("Resampling events per run")
    ax.set_title(f"Resampling frequency vs dimension\n(over {n_timesteps} steps, N={n_particles})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, n_timesteps + 1)
    for bar, dimension in zip(bars, dimensions):
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.3,
                f"{results[dimension]['resamp_mean']:.1f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    tight_layout()
    render_figure(fig)
    print("\n=== CURSE OF DIMENSIONALITY SUMMARY ===")
    print(f"{'d':>4}  {'Final ESS':>10}  {'ESS/N (%)':>10}  {'Resamplings':>12}")
    print("-" * 42)
    for dimension in dimensions:
        final_ess = results[dimension]["final_ess_mean"]
        print(f"{dimension:>4}  {final_ess:>10.1f}  {100 * final_ess / n_particles:>9.1f}%  "
              f"{results[dimension]['resamp_mean']:>11.1f}")
    return results


def animate_high_dim_prior_resampling(d=4, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                                      particle_multiplier=50, seed=42, pause=0,
                                      figsize=(14, 10)):
    """Animate the high-dimensional prior-proposal filter with resampling."""
    n_particles = particle_multiplier * d
    resample_thresh = n_particles / 2
    fig, ax1, ax2, ax3, ax4 = _configure_pf_figure(figsize)
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, d=d, seed=seed)
    random.seed(seed)
    lineage = zeros((n_particles, n_timesteps, d))
    particles = random.normal(0, 1.0, (n_particles, d))
    lineage[:, 0, :] = particles.copy()
    log_w = sum(-0.5 * ((observations[0] - particles) / obs_noise) ** 2, axis=1)
    log_w -= log_w.max()
    weights = exp(log_w)
    weights /= weights.sum()
    colors = make_particle_colors(n_particles, seed=seed)
    resample_times = []
    rmse_history = []

    for step in range(n_timesteps):
        did_resample = False
        if step > 0:
            x_prev = particles.copy()
            ess = 1.0 / sum(weights ** 2)
            if ess < resample_thresh:
                idx = systematic_resample(weights)
                lineage[:, :step, :] = lineage[idx, :step, :]
                x_prev = x_prev[idx]
                colors = colors[idx]
                weights = ones(n_particles) / n_particles
                did_resample = True
                resample_times.append(step)
            particles = x_prev + random.normal(0, process_noise, (n_particles, d))
            lineage[:, step, :] = particles
            log_lik_t = sum(-0.5 * ((observations[step] - particles) / obs_noise) ** 2, axis=1)
            log_lik_t -= log_lik_t.max()
            weights = weights * exp(log_lik_t)
            weights /= weights.sum()

        ks_means = zeros((step + 1, d))
        ks_stds = zeros((step + 1, d))
        for dim in range(d):
            ks_means[:, dim], ks_stds[:, dim] = kalman_smoother_up_to(step, observations[:, dim], process_noise, obs_noise)

        weighted_lineage = lineage[:, : step + 1, :]
        part_mean = sum(weights[:, newaxis, newaxis] * weighted_lineage, axis=0)
        part_std = sqrt(sum(weights[:, newaxis, newaxis] *
                            (weighted_lineage - part_mean[newaxis]) ** 2, axis=0))
        rmse_per_dim = sqrt(mean((part_mean - ks_means) ** 2, axis=0))
        rmse_history.append(mean(rmse_per_dim))

        ess = 1.0 / sum(weights ** 2)
        max_w = weights.max()
        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()

        ax1.plot(true_x[: step + 1, 0], "b-", linewidth=3, label="True state (dim 0)", zorder=5)
        ax1.plot(observations[: step + 1, 0], "ro", markersize=6, label="Obs (dim 0)", zorder=5)
        for particle in range(n_particles):
            alpha = float(0.03 + 0.70 * (weights[particle] / max_w))
            linewidth = float(0.2 + 2.0 * (weights[particle] / max_w))
            ax1.plot(range(step + 1), lineage[particle, : step + 1, 0],
                     color=colors[particle], alpha=alpha, linewidth=linewidth)
            ax1.scatter(step, particles[particle, 0], s=8 + (weights[particle] / max_w) * 100,
                        c=colors[particle], zorder=4, edgecolors="black", linewidth=0.3, alpha=0.85)
        for resample_time in resample_times:
            ax1.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        if resample_times:
            ax1.axvline(resample_times[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.8,
                        label="Resampling event")
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State (dim 0)")
        ax1.set_title(
            f"d={d} — Prior proposal + resampling  (t = {step})\n"
            f"ESS = {ess:.1f} / {n_particles}  — threshold = N/2 = {int(resample_thresh)}"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        t_range = arange(step + 1)
        ax2.plot(t_range, true_x[: step + 1, 0], "b-", linewidth=3, zorder=5, label="True state (dim 0)")
        ax2.fill_between(t_range, ks_means[:, 0] - ks_stds[:, 0], ks_means[:, 0] + ks_stds[:, 0],
                         color="forestgreen", alpha=0.20, label="KS ±1σ  (true posterior)")
        ax2.plot(t_range, ks_means[:, 0], color="forestgreen", linewidth=2.5, label="KS mean  (true posterior)")
        ax2.fill_between(t_range, part_mean[:, 0] - part_std[:, 0], part_mean[:, 0] + part_std[:, 0],
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, part_mean[:, 0], color="darkorange", linewidth=2, linestyle="--", label="Particle mean")
        for resample_time in resample_times:
            ax2.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State (dim 0)")
        ax2.set_title(f"Dim 0 estimate quality  (t = {step})\nParticle mean vs True posterior")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=did_resample,
                               title_suffix=f"  (d={d} dims)")
        _plot_rmse_panel(ax4, rmse_history, n_timesteps,
                         f"Estimation error over time\nPrior proposal + resampling  (d={d} dims,  N={n_particles})",
                         label=f"Mean RMSE  (d={d} dims)",
                         resample_times=resample_times)
        tight_layout()
        animated_frame(fig, pause=pause)

    print(f"\nd={d}, N={n_particles}: Resampling at timesteps: {resample_times}")
    print(f"Total resampling events: {len(resample_times)}")
    print(f"Final ESS: {1.0 / sum(weights ** 2):.1f} / {n_particles}")
    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=resample_times)


def animate_high_dim_optimal_resampling(d=4, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                                        particle_multiplier=50, seed=42, pause=0,
                                        figsize=(14, 10)):
    """Animate the high-dimensional optimal-proposal filter with resampling."""
    n_particles = particle_multiplier * d
    resample_thresh = n_particles / 2
    sigma_opt2 = 1.0 / (1.0 / process_noise ** 2 + 1.0 / obs_noise ** 2)
    sigma_opt = sqrt(sigma_opt2)
    sigma_pred2 = process_noise ** 2 + obs_noise ** 2
    fig, ax1, ax2, ax3, ax4 = _configure_pf_figure(figsize)
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, d=d, seed=seed)
    random.seed(seed)
    lineage = zeros((n_particles, n_timesteps, d))
    particles = random.normal(0, 1.0, (n_particles, d))
    lineage[:, 0, :] = particles.copy()
    log_w = sum(-0.5 * ((observations[0] - particles) / obs_noise) ** 2, axis=1)
    log_w -= log_w.max()
    weights = exp(log_w)
    weights /= weights.sum()
    colors = make_particle_colors(n_particles, seed=seed)
    resample_times = []
    rmse_history = []

    for step in range(n_timesteps):
        did_resample = False
        if step > 0:
            x_prev = particles.copy()
            ess = 1.0 / sum(weights ** 2)
            if ess < resample_thresh:
                idx = systematic_resample(weights)
                lineage[:, :step, :] = lineage[idx, :step, :]
                x_prev = x_prev[idx]
                colors = colors[idx]
                weights = ones(n_particles) / n_particles
                did_resample = True
                resample_times.append(step)
            mu_opt = sigma_opt2 * (x_prev / process_noise ** 2 + observations[step] / obs_noise ** 2)
            particles = random.normal(mu_opt, sigma_opt)
            lineage[:, step, :] = particles
            log_pred = sum(-0.5 * ((observations[step] - x_prev) ** 2 / sigma_pred2), axis=1)
            log_pred -= log_pred.max()
            weights = weights * exp(log_pred)
            weights /= weights.sum()

        ks_means = zeros((step + 1, d))
        ks_stds = zeros((step + 1, d))
        for dim in range(d):
            ks_means[:, dim], ks_stds[:, dim] = kalman_smoother_up_to(step, observations[:, dim], process_noise, obs_noise)

        weighted_lineage = lineage[:, : step + 1, :]
        part_mean = sum(weights[:, newaxis, newaxis] * weighted_lineage, axis=0)
        part_std = sqrt(sum(weights[:, newaxis, newaxis] *
                            (weighted_lineage - part_mean[newaxis]) ** 2, axis=0))
        rmse_per_dim = sqrt(mean((part_mean - ks_means) ** 2, axis=0))
        rmse_history.append(mean(rmse_per_dim))

        ess = 1.0 / sum(weights ** 2)
        max_w = weights.max()
        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()

        ax1.plot(true_x[: step + 1, 0], "b-", linewidth=3, label="True state (dim 0)", zorder=5)
        ax1.plot(observations[: step + 1, 0], "ro", markersize=6, label="Obs (dim 0)", zorder=5)
        for particle in range(n_particles):
            alpha = float(0.03 + 0.70 * (weights[particle] / max_w))
            linewidth = float(0.2 + 2.0 * (weights[particle] / max_w))
            ax1.plot(range(step + 1), lineage[particle, : step + 1, 0],
                     color=colors[particle], alpha=alpha, linewidth=linewidth)
            ax1.scatter(step, particles[particle, 0], s=8 + (weights[particle] / max_w) * 100,
                        c=colors[particle], zorder=4, edgecolors="black", linewidth=0.3, alpha=0.85)
        for resample_time in resample_times:
            ax1.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        if resample_times:
            ax1.axvline(resample_times[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.8,
                        label="Resampling event")
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State (dim 0)")
        ax1.set_title(
            f"d={d} — Optimal proposal + resampling  (t = {step})\n"
            f"ESS = {ess:.1f} / {n_particles}  — threshold = N/2 = {int(resample_thresh)}"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        t_range = arange(step + 1)
        ax2.plot(t_range, true_x[: step + 1, 0], "b-", linewidth=3, zorder=5, label="True state (dim 0)")
        ax2.fill_between(t_range, ks_means[:, 0] - ks_stds[:, 0], ks_means[:, 0] + ks_stds[:, 0],
                         color="forestgreen", alpha=0.20, label="KS ±1σ  (true posterior)")
        ax2.plot(t_range, ks_means[:, 0], color="forestgreen", linewidth=2.5, label="KS mean  (true posterior)")
        ax2.fill_between(t_range, part_mean[:, 0] - part_std[:, 0], part_mean[:, 0] + part_std[:, 0],
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, part_mean[:, 0], color="darkorange", linewidth=2, linestyle="--", label="Particle mean")
        for resample_time in resample_times:
            ax2.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State (dim 0)")
        ax2.set_title(f"Dim 0 estimate quality  (t = {step})\nParticle mean vs True posterior")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=did_resample,
                               title_suffix=f"  (optimal proposal, d={d} dims)")
        _plot_rmse_panel(ax4, rmse_history, n_timesteps,
                         f"Estimation error over time\nOptimal proposal + resampling  (d={d} dims,  N={n_particles})",
                         label=f"Mean RMSE  (optimal proposal, d={d})",
                         resample_times=resample_times)
        tight_layout()
        animated_frame(fig, pause=pause)

    print(f"\nd={d}, N={n_particles}: Resampling at timesteps: {resample_times}")
    print(f"Total resampling events: {len(resample_times)}")
    print(f"Final ESS: {1.0 / sum(weights ** 2):.1f} / {n_particles}")
    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=resample_times)


def animate_high_dim_filtering_prior(d=4, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                                     particle_multiplier=50, seed=42, pause=0,
                                     figsize=(14, 10)):
    """Animate the high-dimensional filtering-distribution demo with the prior proposal."""
    n_particles = particle_multiplier * d
    resample_thresh = n_particles / 2
    fig, ax1, ax2, ax3, ax4 = _configure_pf_figure(figsize)
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, d=d, seed=seed)
    random.seed(seed)
    particles = random.normal(0, 1.0, (n_particles, d))
    log_w = sum(-0.5 * ((observations[0] - particles) / obs_noise) ** 2, axis=1)
    log_w -= log_w.max()
    weights = exp(log_w)
    weights /= weights.sum()
    colors = make_particle_colors(n_particles, seed=seed)
    resample_times = []
    rmse_history = []
    filter_mean_history = zeros((n_timesteps, d))
    filter_std_history = zeros((n_timesteps, d))
    kf_mean_history = zeros((n_timesteps, d))
    kf_std_history = zeros((n_timesteps, d))

    for step in range(n_timesteps):
        did_resample = False
        if step > 0:
            x_prev = particles.copy()
            ess = 1.0 / sum(weights ** 2)
            if ess < resample_thresh:
                idx = systematic_resample(weights)
                x_prev = x_prev[idx]
                colors = colors[idx]
                weights = ones(n_particles) / n_particles
                did_resample = True
                resample_times.append(step)
            particles = x_prev + random.normal(0, process_noise, (n_particles, d))
            log_lik_t = sum(-0.5 * ((observations[step] - particles) / obs_noise) ** 2, axis=1)
            log_lik_t -= log_lik_t.max()
            weights = weights * exp(log_lik_t)
            weights /= weights.sum()

        for dim in range(d):
            ks_mean, ks_std = kalman_smoother_up_to(step, observations[:, dim], process_noise, obs_noise)
            kf_mean_history[step, dim] = ks_mean[-1]
            kf_std_history[step, dim] = ks_std[-1]

        filter_mean_history[step] = sum(weights[:, newaxis] * particles, axis=0)
        filter_std_history[step] = sqrt(sum(weights[:, newaxis] * (particles - filter_mean_history[step]) ** 2, axis=0))
        rmse_history.append(float(sqrt(mean((filter_mean_history[step] - kf_mean_history[step]) ** 2))))

        ess = 1.0 / sum(weights ** 2)
        max_w = weights.max()
        t_range = arange(step + 1)
        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()

        ax1.plot(range(step + 1), true_x[: step + 1, 0], "b-", linewidth=3, label="True state (dim 0)", zorder=5)
        ax1.plot(range(step + 1), observations[: step + 1, 0], "ro", markersize=6, label="Obs (dim 0)", zorder=5)
        marker_sizes = 8 + (weights / max_w) * 100
        ax1.scatter([step] * n_particles, particles[:, 0], s=marker_sizes, c=list(colors), zorder=4,
                    edgecolors="black", linewidth=0.3, alpha=0.85)
        ax1.errorbar(step, filter_mean_history[step, 0], yerr=filter_std_history[step, 0], fmt="D",
                     color="darkorange", markersize=9, linewidth=2, capsize=6, zorder=6,
                     label="Particle filter mean ±1σ")
        ax1.errorbar(step, kf_mean_history[step, 0], yerr=kf_std_history[step, 0], fmt="s",
                     color="forestgreen", markersize=9, linewidth=2, capsize=6, zorder=7,
                     label="KF mean ±1σ  (exact)")
        for resample_time in resample_times:
            ax1.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        if resample_times:
            ax1.axvline(resample_times[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.8,
                        label="Resampling event")
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State (dim 0)")
        ax1.set_title(
            f"d={d} — Filtering distribution  (t = {step})\n"
            f"ESS = {ess:.1f} / {n_particles}  —  threshold = N/2 = {int(resample_thresh)}"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_range, true_x[: step + 1, 0], "b-", linewidth=3, zorder=5, label="True state (dim 0)")
        ax2.fill_between(t_range,
                         kf_mean_history[: step + 1, 0] - kf_std_history[: step + 1, 0],
                         kf_mean_history[: step + 1, 0] + kf_std_history[: step + 1, 0],
                         color="forestgreen", alpha=0.20, label="KF ±1σ  (exact filter)")
        ax2.plot(t_range, kf_mean_history[: step + 1, 0], color="forestgreen", linewidth=2.5,
                 label="KF mean  (exact filter)")
        ax2.fill_between(t_range,
                         filter_mean_history[: step + 1, 0] - filter_std_history[: step + 1, 0],
                         filter_mean_history[: step + 1, 0] + filter_std_history[: step + 1, 0],
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, filter_mean_history[: step + 1, 0], color="darkorange", linewidth=2,
                 linestyle="--", label="Particle filtering mean")
        for resample_time in resample_times:
            ax2.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State (dim 0)")
        ax2.set_title(f"Filtering mean history  (t = {step})\nParticle filter mean vs Kalman filter (KF)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=did_resample,
                               title_suffix=f"  (d={d} dims)")
        _plot_rmse_panel(ax4, rmse_history, n_timesteps,
                         f"Filtering estimation error over time\nPrior proposal + resampling  (d={d} dims,  N={n_particles})",
                         label=f"Filter RMSE  (d={d} dims)",
                         resample_times=resample_times)
        tight_layout()
        animated_frame(fig, pause=pause)

    print(f"\nd={d}, N={n_particles}: Resampling at timesteps: {resample_times}")
    print(f"Total resampling events: {len(resample_times)}")
    print(f"Final ESS: {1.0 / sum(weights ** 2):.1f} / {n_particles}")
    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=resample_times)


def animate_high_dim_filtering_optimal(d=4, n_timesteps=25, process_noise=0.5, obs_noise=1.0,
                                       particle_multiplier=50, seed=42, pause=0,
                                       figsize=(14, 10)):
    """Animate the high-dimensional filtering-distribution demo with the optimal proposal."""
    n_particles = particle_multiplier * d
    resample_thresh = n_particles / 2
    sigma_opt2 = 1.0 / (1.0 / process_noise ** 2 + 1.0 / obs_noise ** 2)
    sigma_opt = sqrt(sigma_opt2)
    sigma_pred2 = process_noise ** 2 + obs_noise ** 2
    fig, ax1, ax2, ax3, ax4 = _configure_pf_figure(figsize)
    true_x, observations = generate_linear_gaussian_data(n_timesteps, process_noise, obs_noise, d=d, seed=seed)
    random.seed(seed)
    particles = random.normal(0, 1.0, (n_particles, d))
    log_w = sum(-0.5 * ((observations[0] - particles) / obs_noise) ** 2, axis=1)
    log_w -= log_w.max()
    weights = exp(log_w)
    weights /= weights.sum()
    colors = make_particle_colors(n_particles, seed=seed)
    resample_times = []
    rmse_history = []
    filter_mean_history = zeros((n_timesteps, d))
    filter_std_history = zeros((n_timesteps, d))
    kf_mean_history = zeros((n_timesteps, d))
    kf_std_history = zeros((n_timesteps, d))

    for step in range(n_timesteps):
        did_resample = False
        if step > 0:
            x_prev = particles.copy()
            ess = 1.0 / sum(weights ** 2)
            if ess < resample_thresh:
                idx = systematic_resample(weights)
                x_prev = x_prev[idx]
                colors = colors[idx]
                weights = ones(n_particles) / n_particles
                did_resample = True
                resample_times.append(step)
            mu_opt = sigma_opt2 * (x_prev / process_noise ** 2 + observations[step] / obs_noise ** 2)
            particles = random.normal(mu_opt, sigma_opt)
            log_pred = sum(-0.5 * ((observations[step] - x_prev) ** 2 / sigma_pred2), axis=1)
            log_pred -= log_pred.max()
            weights = weights * exp(log_pred)
            weights /= weights.sum()

        for dim in range(d):
            ks_mean, ks_std = kalman_smoother_up_to(step, observations[:, dim], process_noise, obs_noise)
            kf_mean_history[step, dim] = ks_mean[-1]
            kf_std_history[step, dim] = ks_std[-1]

        filter_mean_history[step] = sum(weights[:, newaxis] * particles, axis=0)
        filter_std_history[step] = sqrt(sum(weights[:, newaxis] * (particles - filter_mean_history[step]) ** 2, axis=0))
        rmse_history.append(float(sqrt(mean((filter_mean_history[step] - kf_mean_history[step]) ** 2))))

        ess = 1.0 / sum(weights ** 2)
        max_w = weights.max()
        t_range = arange(step + 1)
        ax1.clear(); ax2.clear(); ax3.clear(); ax4.clear()

        ax1.plot(range(step + 1), true_x[: step + 1, 0], "b-", linewidth=3, label="True state (dim 0)", zorder=5)
        ax1.plot(range(step + 1), observations[: step + 1, 0], "ro", markersize=6, label="Obs (dim 0)", zorder=5)
        marker_sizes = 8 + (weights / max_w) * 100
        ax1.scatter([step] * n_particles, particles[:, 0], s=marker_sizes, c=list(colors), zorder=4,
                    edgecolors="black", linewidth=0.3, alpha=0.85)
        ax1.errorbar(step, filter_mean_history[step, 0], yerr=filter_std_history[step, 0], fmt="D",
                     color="darkorange", markersize=9, linewidth=2, capsize=6, zorder=6,
                     label="Particle filter mean ±1σ")
        ax1.errorbar(step, kf_mean_history[step, 0], yerr=kf_std_history[step, 0], fmt="s",
                     color="forestgreen", markersize=9, linewidth=2, capsize=6, zorder=7,
                     label="KF mean ±1σ  (exact)")
        for resample_time in resample_times:
            ax1.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        if resample_times:
            ax1.axvline(resample_times[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.8,
                        label="Resampling event")
        _set_panel_limits(ax1, n_timesteps)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("State (dim 0)")
        ax1.set_title(
            f"d={d} — Optimal proposal filtering  (t = {step})\n"
            f"ESS = {ess:.1f} / {n_particles}  —  threshold = N/2 = {int(resample_thresh)}"
        )
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        ax2.plot(t_range, true_x[: step + 1, 0], "b-", linewidth=3, zorder=5, label="True state (dim 0)")
        ax2.fill_between(t_range,
                         kf_mean_history[: step + 1, 0] - kf_std_history[: step + 1, 0],
                         kf_mean_history[: step + 1, 0] + kf_std_history[: step + 1, 0],
                         color="forestgreen", alpha=0.20, label="KF ±1σ  (exact filter)")
        ax2.plot(t_range, kf_mean_history[: step + 1, 0], color="forestgreen", linewidth=2.5,
                 label="KF mean  (exact filter)")
        ax2.fill_between(t_range,
                         filter_mean_history[: step + 1, 0] - filter_std_history[: step + 1, 0],
                         filter_mean_history[: step + 1, 0] + filter_std_history[: step + 1, 0],
                         color="darkorange", alpha=0.20, label="Particle ±1σ")
        ax2.plot(t_range, filter_mean_history[: step + 1, 0], color="darkorange", linewidth=2,
                 linestyle="--", label="Particle filtering mean")
        for resample_time in resample_times:
            ax2.axvline(resample_time, color="limegreen", linewidth=2, linestyle="--", alpha=0.8)
        _set_panel_limits(ax2, n_timesteps)
        ax2.set_xlabel("Time")
        ax2.set_ylabel("State (dim 0)")
        ax2.set_title(f"Filtering mean history  (t = {step})\nParticle filter mean vs Kalman filter (KF)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        _plot_weight_histogram(ax3, weights, n_particles, ess, did_resample=did_resample,
                               title_suffix=f"  (optimal proposal, d={d} dims)")
        _plot_rmse_panel(ax4, rmse_history, n_timesteps,
                         f"Filtering estimation error over time\nOptimal proposal + resampling  (d={d} dims,  N={n_particles})",
                         label=f"Filter RMSE  (optimal proposal, d={d})",
                         resample_times=resample_times)
        tight_layout()
        animated_frame(fig, pause=pause)

    print(f"\nd={d}, N={n_particles}: Resampling at timesteps: {resample_times}")
    print(f"Total resampling events: {len(resample_times)}")
    print(f"Final ESS: {1.0 / sum(weights ** 2):.1f} / {n_particles}")
    close(fig)
    return ParticleFilterHistory(rmse_history=rmse_history, resample_times=resample_times)


def kf_step(m, P, y, sp, so):
    """One-step Kalman filter update used by the SMC² demo."""
    P_pred = P + sp ** 2
    S = P_pred + so ** 2
    innov = y - m
    log_p = -0.5 * (log(2 * pi * S) + innov ** 2 / S)
    K = P_pred / S
    return m + K * innov, (1 - K) * P_pred, log_p


def kf_full_loglik(obs, sp, so):
    """Full Kalman-filter log likelihood used in SMC² rejuvenation."""
    m = 0.0
    P = 1.0
    ll = 0.0
    for y in obs:
        m, P, lp = kf_step(m, P, y, sp, so)
        ll += lp
    return ll, m, P


def run_smc2_demo(n_timesteps=250, sigma_proc_true=0.5, sigma_obs_true=1.0,
                  M=200, n_mcmc=5, seed=42, figsize=(14, 10)):
    """Run the SMC² demo for joint state and parameter filtering in 1D."""
    resample_thresh_outer = M / 2
    random.seed(seed)
    true_x_1d, observations_1d = generate_linear_gaussian_data(
        n_timesteps, sigma_proc_true, sigma_obs_true, d=1, seed=seed
    )
    log_sp_mu = 0.0
    log_sp_std = 0.5
    log_so_mu = 0.0
    log_so_std = 0.5
    random.seed(123)
    log_sp = random.normal(log_sp_mu, log_sp_std, M)
    log_so = random.normal(log_so_mu, log_so_std, M)
    kf_m_all = zeros(M)
    kf_P_all = ones(M)
    cumloglik = zeros(M)
    outer_log_w = zeros(M)
    theta_weights = ones(M) / M
    theta_mean_sp = zeros(n_timesteps)
    theta_mean_so = zeros(n_timesteps)
    theta_std_sp = zeros(n_timesteps)
    theta_std_so = zeros(n_timesteps)
    outer_ess_hist = zeros(n_timesteps)
    smc2_filt_m = zeros(n_timesteps)
    smc2_filt_s = zeros(n_timesteps)
    resample_events = []

    for step in range(n_timesteps):
        y = observations_1d[step]
        sp = exp(log_sp)
        so = exp(log_so)
        log_incr = zeros(M)
        for particle in range(M):
            new_m, new_P, lp = kf_step(kf_m_all[particle], kf_P_all[particle], y, sp[particle], so[particle])
            kf_m_all[particle] = new_m
            kf_P_all[particle] = new_P
            cumloglik[particle] += lp
            log_incr[particle] = lp

        outer_log_w += log_incr - log_incr.max()
        outer_log_w -= outer_log_w.max()
        theta_weights = exp(outer_log_w)
        theta_weights /= theta_weights.sum()
        ess = 1.0 / sum(theta_weights ** 2)
        outer_ess_hist[step] = ess
        smc2_filt_m[step] = sum(theta_weights * kf_m_all)
        smc2_filt_s[step] = sqrt(sum(theta_weights * (kf_P_all + (kf_m_all - smc2_filt_m[step]) ** 2)))

        sp = exp(log_sp)
        so = exp(log_so)
        theta_mean_sp[step] = sum(theta_weights * sp)
        theta_mean_so[step] = sum(theta_weights * so)
        theta_std_sp[step] = sqrt(sum(theta_weights * (sp - theta_mean_sp[step]) ** 2))
        theta_std_so[step] = sqrt(sum(theta_weights * (so - theta_mean_so[step]) ** 2))

        if ess < resample_thresh_outer:
            resample_events.append(step)
            idx = systematic_resample(theta_weights)
            log_sp = log_sp[idx].copy()
            log_so = log_so[idx].copy()
            kf_m_all = kf_m_all[idx].copy()
            kf_P_all = kf_P_all[idx].copy()
            cumloglik = cumloglik[idx].copy()
            outer_log_w = zeros(M)
            theta_weights = ones(M) / M
            obs_so_far = observations_1d[: step + 1]
            mh_step = 0.1
            for _ in range(n_mcmc):
                for particle in range(M):
                    prop_lsp = log_sp[particle] + random.normal(0, mh_step)
                    prop_lso = log_so[particle] + random.normal(0, mh_step)
                    ll_prop, pm_state, pP_state = kf_full_loglik(obs_so_far, exp(prop_lsp), exp(prop_lso))
                    lp_prop = (
                        -0.5 * ((prop_lsp - log_sp_mu) / log_sp_std) ** 2
                        - 0.5 * ((prop_lso - log_so_mu) / log_so_std) ** 2
                    )
                    lp_curr = (
                        -0.5 * ((log_sp[particle] - log_sp_mu) / log_sp_std) ** 2
                        - 0.5 * ((log_so[particle] - log_so_mu) / log_so_std) ** 2
                    )
                    log_accept = (ll_prop + lp_prop) - (cumloglik[particle] + lp_curr)
                    if log(random.uniform()) < log_accept:
                        log_sp[particle] = prop_lsp
                        log_so[particle] = prop_lso
                        cumloglik[particle] = ll_prop
                        kf_m_all[particle] = pm_state
                        kf_P_all[particle] = pP_state

    kf_m_true = 0.0
    kf_P_true = 1.0
    kf_means_true = zeros(n_timesteps)
    kf_stds_true = zeros(n_timesteps)
    for step in range(n_timesteps):
        kf_m_true, kf_P_true, _ = kf_step(kf_m_true, kf_P_true, observations_1d[step], sigma_proc_true, sigma_obs_true)
        kf_means_true[step] = kf_m_true
        kf_stds_true[step] = sqrt(kf_P_true)

    t_range = arange(n_timesteps)
    fig, axes = subplots(2, 2, figsize=figsize)

    ax = axes[0, 0]
    ax.plot(t_range, true_x_1d, "b-", linewidth=3, label="True state", zorder=5)
    ax.plot(t_range, observations_1d, "ro", markersize=5, label="Observations", zorder=5)
    ax.fill_between(t_range, kf_means_true - kf_stds_true, kf_means_true + kf_stds_true, color="forestgreen", alpha=0.20)
    ax.plot(t_range, kf_means_true, color="forestgreen", linewidth=2.5, label="KF ±1σ  (true θ known)")
    ax.fill_between(t_range, smc2_filt_m - smc2_filt_s, smc2_filt_m + smc2_filt_s, color="darkorange", alpha=0.20)
    ax.plot(t_range, smc2_filt_m, color="darkorange", linewidth=2, linestyle="--", label="SMC² ±1σ  (unknown θ)")
    for event in resample_events:
        ax.axvline(event, color="limegreen", linewidth=2, linestyle="--", alpha=0.7)
    if resample_events:
        ax.axvline(resample_events[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.7,
                   label="Outer resampling")
    ax.set_xlim(-0.5, n_timesteps - 0.5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("Time")
    ax.set_ylabel("State")
    ax.set_title(f"State filtering: SMC² (unknown θ) vs KF (known θ)\n(M={M} θ-particles, {n_mcmc} MH steps/rejuvenation)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.fill_between(t_range, theta_mean_sp - theta_std_sp, theta_mean_sp + theta_std_sp, color="steelblue", alpha=0.25)
    ax.plot(t_range, theta_mean_sp, color="steelblue", linewidth=2.5, label="σ_proc estimate ±1σ")
    ax.axhline(sigma_proc_true, color="steelblue", linewidth=1.5, linestyle=":", label=f"True σ_proc = {sigma_proc_true}")
    ax.fill_between(t_range, theta_mean_so - theta_std_so, theta_mean_so + theta_std_so, color="tomato", alpha=0.25)
    ax.plot(t_range, theta_mean_so, color="tomato", linewidth=2.5, label="σ_obs estimate ±1σ")
    ax.axhline(sigma_obs_true, color="tomato", linewidth=1.5, linestyle=":", label=f"True σ_obs = {sigma_obs_true}")
    for event in resample_events:
        ax.axvline(event, color="limegreen", linewidth=2, linestyle="--", alpha=0.7)
    ax.set_xlim(-0.5, n_timesteps - 0.5)
    ax.set_ylim(0, 2.5)
    ax.set_xlabel("Time")
    ax.set_ylabel("Parameter estimate")
    ax.set_title("Online parameter learning\n(θ-weighted mean and ±1σ of outer particles)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t_range, outer_ess_hist, "r-o", markersize=5, linewidth=2, label="Outer θ-particle ESS")
    ax.axhline(resample_thresh_outer, color="red", linestyle="--", linewidth=1.5,
               label=f"Resample threshold  (M/2 = {int(resample_thresh_outer)})")
    ax.axhline(M, color="gray", linestyle=":", linewidth=1, label=f"Max  (M = {M})")
    for event in resample_events:
        ax.axvline(event, color="limegreen", linewidth=2, linestyle="--", alpha=0.7)
    if resample_events:
        ax.axvline(resample_events[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.7,
                   label="Resampling + rejuvenation")
    ax.set_xlim(-0.5, n_timesteps - 0.5)
    ax.set_ylim(0, M + 20)
    ax.set_xlabel("Time step  t")
    ax.set_ylabel("ESS  (θ-particles)")
    ax.set_title("Outer (parameter) ESS over time\n(ESS resets after resampling + MH rejuvenation)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    scatter = ax.scatter(exp(log_sp), exp(log_so), c=theta_weights, cmap="plasma", alpha=0.85,
                         edgecolors="black", linewidth=0.3, s=20 + theta_weights / theta_weights.max() * 120)
    colorbar(scatter, ax=ax, label="θ-particle weight")
    ax.axvline(sigma_proc_true, color="steelblue", linewidth=2, linestyle="--", label=f"True σ_proc = {sigma_proc_true}")
    ax.axhline(sigma_obs_true, color="tomato", linewidth=2, linestyle="--", label=f"True σ_obs = {sigma_obs_true}")
    ax.set_xlabel("σ_proc  (process noise)")
    ax.set_ylabel("σ_obs  (observation noise)")
    ax.set_title(f"Final θ-particle cloud  (t = {n_timesteps - 1})\nColour / size ∝ weight")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    tight_layout()
    render_figure(fig)
    print("\n=== SMC² RESULTS ===")
    print("Final parameter estimates (θ-weighted mean ± std):")
    print(f"  σ_proc: {theta_mean_sp[-1]:.3f} ± {theta_std_sp[-1]:.3f}  (true: {sigma_proc_true})")
    print(f"  σ_obs:  {theta_mean_so[-1]:.3f} ± {theta_std_so[-1]:.3f}  (true: {sigma_obs_true})")
    print(f"Outer resampling + rejuvenation events at t = {resample_events}")
    print(f"Total rejuvenation events: {len(resample_events)}")
    return {
        "theta_mean_sp": theta_mean_sp,
        "theta_mean_so": theta_mean_so,
        "theta_std_sp": theta_std_sp,
        "theta_std_so": theta_std_so,
        "outer_ess_hist": outer_ess_hist,
        "resample_events": resample_events,
        "smc2_filt_m": smc2_filt_m,
        "smc2_filt_s": smc2_filt_s,
        "kf_means_true": kf_means_true,
        "kf_stds_true": kf_stds_true,
    }


def animate_smc2_filtering_demo(n_timesteps=200, sigma_proc_true=0.5, sigma_obs_true=1.0,
                                M=200, n_mcmc=5, seed=42, pause=0.0, figsize=(14, 10)):
    """Animate SMC² for online filtering of both the state and unknown parameters."""
    resample_thresh_outer = M / 2
    random.seed(seed)
    true_x_1d, observations_1d = generate_linear_gaussian_data(
        n_timesteps, sigma_proc_true, sigma_obs_true, d=1, seed=seed
    )
    log_sp_mu = 0.0
    log_sp_std = 0.5
    log_so_mu = 0.0
    log_so_std = 0.5
    random.seed(123)
    log_sp = random.normal(log_sp_mu, log_sp_std, M)
    log_so = random.normal(log_so_mu, log_so_std, M)
    kf_m_all = zeros(M)
    kf_P_all = ones(M)
    cumloglik = zeros(M)
    outer_log_w = zeros(M)
    theta_weights = ones(M) / M
    theta_mean_sp = zeros(n_timesteps)
    theta_mean_so = zeros(n_timesteps)
    theta_std_sp = zeros(n_timesteps)
    theta_std_so = zeros(n_timesteps)
    outer_ess_hist = zeros(n_timesteps)
    smc2_filt_m = zeros(n_timesteps)
    smc2_filt_s = zeros(n_timesteps)
    kf_means_true = zeros(n_timesteps)
    kf_stds_true = zeros(n_timesteps)
    resample_events = []

    fig, axes = subplots(2, 2, figsize=figsize)
    kf_m_true = 0.0
    kf_P_true = 1.0

    for step in range(n_timesteps):
        y = observations_1d[step]
        sp = exp(log_sp)
        so = exp(log_so)
        log_incr = zeros(M)
        for particle in range(M):
            new_m, new_P, lp = kf_step(kf_m_all[particle], kf_P_all[particle], y, sp[particle], so[particle])
            kf_m_all[particle] = new_m
            kf_P_all[particle] = new_P
            cumloglik[particle] += lp
            log_incr[particle] = lp

        outer_log_w += log_incr - log_incr.max()
        outer_log_w -= outer_log_w.max()
        theta_weights = exp(outer_log_w)
        theta_weights /= theta_weights.sum()
        ess = 1.0 / sum(theta_weights ** 2)
        outer_ess_hist[step] = ess
        smc2_filt_m[step] = sum(theta_weights * kf_m_all)
        smc2_filt_s[step] = sqrt(sum(theta_weights * (kf_P_all + (kf_m_all - smc2_filt_m[step]) ** 2)))

        sp = exp(log_sp)
        so = exp(log_so)
        theta_mean_sp[step] = sum(theta_weights * sp)
        theta_mean_so[step] = sum(theta_weights * so)
        theta_std_sp[step] = sqrt(sum(theta_weights * (sp - theta_mean_sp[step]) ** 2))
        theta_std_so[step] = sqrt(sum(theta_weights * (so - theta_mean_so[step]) ** 2))

        kf_m_true, kf_P_true, _ = kf_step(kf_m_true, kf_P_true, y, sigma_proc_true, sigma_obs_true)
        kf_means_true[step] = kf_m_true
        kf_stds_true[step] = sqrt(kf_P_true)

        did_resample = False
        if ess < resample_thresh_outer:
            did_resample = True
            resample_events.append(step)
            idx = systematic_resample(theta_weights)
            log_sp = log_sp[idx].copy()
            log_so = log_so[idx].copy()
            kf_m_all = kf_m_all[idx].copy()
            kf_P_all = kf_P_all[idx].copy()
            cumloglik = cumloglik[idx].copy()
            outer_log_w = zeros(M)
            theta_weights = ones(M) / M
            obs_so_far = observations_1d[: step + 1]
            mh_step = 0.1
            for _ in range(n_mcmc):
                for particle in range(M):
                    prop_lsp = log_sp[particle] + random.normal(0, mh_step)
                    prop_lso = log_so[particle] + random.normal(0, mh_step)
                    ll_prop, pm_state, pP_state = kf_full_loglik(obs_so_far, exp(prop_lsp), exp(prop_lso))
                    lp_prop = (
                        -0.5 * ((prop_lsp - log_sp_mu) / log_sp_std) ** 2
                        - 0.5 * ((prop_lso - log_so_mu) / log_so_std) ** 2
                    )
                    lp_curr = (
                        -0.5 * ((log_sp[particle] - log_sp_mu) / log_sp_std) ** 2
                        - 0.5 * ((log_so[particle] - log_so_mu) / log_so_std) ** 2
                    )
                    log_accept = (ll_prop + lp_prop) - (cumloglik[particle] + lp_curr)
                    if log(random.uniform()) < log_accept:
                        log_sp[particle] = prop_lsp
                        log_so[particle] = prop_lso
                        cumloglik[particle] = ll_prop
                        kf_m_all[particle] = pm_state
                        kf_P_all[particle] = pP_state

        t_range = arange(step + 1)
        current_sp = exp(log_sp)
        current_so = exp(log_so)
        max_weight = theta_weights.max()
        state_samples = kf_m_all + random.normal(0, sqrt(kf_P_all))

        ax_state = axes[0, 0]
        ax_param = axes[0, 1]
        ax_ess = axes[1, 0]
        ax_cloud = axes[1, 1]
        ax_state.clear(); ax_param.clear(); ax_ess.clear(); ax_cloud.clear()

        marker_sizes = 18 + theta_weights / max_weight * 140
        ax_state.plot(t_range, true_x_1d[: step + 1], "b-", linewidth=3, label="True state", zorder=5)
        ax_state.plot(t_range, observations_1d[: step + 1], "ro", markersize=5, label="Observations", zorder=5)
        ax_state.scatter(
            full(M, step),
            state_samples,
            c=theta_weights,
            cmap="plasma",
            s=marker_sizes,
            zorder=4,
            edgecolors="black",
            linewidth=0.3,
            alpha=0.85,
            label="Current inner-state samples",
        )
        ax_state.errorbar(
            step,
            smc2_filt_m[step],
            yerr=smc2_filt_s[step],
            fmt="D",
            color="darkorange",
            markersize=9,
            linewidth=2,
            capsize=6,
            zorder=6,
            label="SMC² mean ±1σ",
        )
        ax_state.errorbar(
            step,
            kf_means_true[step],
            yerr=kf_stds_true[step],
            fmt="s",
            color="forestgreen",
            markersize=9,
            linewidth=2,
            capsize=6,
            zorder=7,
            label="KF mean ±1σ  (true posterior)",
        )
        for event in resample_events:
            ax_state.axvline(event, color="limegreen", linewidth=2, linestyle="--", alpha=0.7)
        if resample_events:
            ax_state.axvline(resample_events[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.7,
                             label="Outer resampling")
        ax_state.set_xlim(-0.5, n_timesteps - 0.5)
        ax_state.set_ylim(-5, 5)
        ax_state.set_xlabel("Time")
        ax_state.set_ylabel("State")
        title_suffix = "resampled + rejuvenated" if did_resample else f"ESS = {ess:.1f} / {M}"
        ax_state.set_title(f"SMC² filtering  (t = {step})\nCurrent inner-state samples  — {title_suffix}")
        ax_state.legend(fontsize=8)
        ax_state.grid(True, alpha=0.3)

        ax_param.plot(t_range, true_x_1d[: step + 1], "b-", linewidth=3, zorder=5, label="True state")
        ax_param.fill_between(
            t_range,
            kf_means_true[: step + 1] - kf_stds_true[: step + 1],
            kf_means_true[: step + 1] + kf_stds_true[: step + 1],
            color="forestgreen",
            alpha=0.20,
            label="KF ±1σ  (true posterior)",
        )
        ax_param.plot(t_range, kf_means_true[: step + 1], color="forestgreen", linewidth=2.5,
                      label="KF mean  (true posterior)")
        ax_param.fill_between(
            t_range,
            smc2_filt_m[: step + 1] - smc2_filt_s[: step + 1],
            smc2_filt_m[: step + 1] + smc2_filt_s[: step + 1],
            color="darkorange",
            alpha=0.20,
            label="SMC² ±1σ",
        )
        ax_param.plot(t_range, smc2_filt_m[: step + 1], color="darkorange", linewidth=2,
                      linestyle="--", label="SMC² filtering mean")
        for event in resample_events:
            ax_param.axvline(event, color="limegreen", linewidth=2, linestyle="--", alpha=0.7)
        ax_param.set_xlim(-0.5, n_timesteps - 0.5)
        ax_param.set_ylim(-5, 5)
        ax_param.set_xlabel("Time")
        ax_param.set_ylabel("State")
        ax_param.set_title(f"Filtering mean history  (t = {step})\nSMC² mean vs Kalman filter (KF)")
        ax_param.legend(fontsize=8)
        ax_param.grid(True, alpha=0.3)

        ax_ess.plot(t_range, outer_ess_hist[: step + 1], "r-o", markersize=4, linewidth=2,
                    label="Outer θ-particle ESS")
        ax_ess.axhline(resample_thresh_outer, color="red", linestyle="--", linewidth=1.5,
                       label=f"Resample threshold  (M/2 = {int(resample_thresh_outer)})")
        ax_ess.axhline(M, color="gray", linestyle=":", linewidth=1, label=f"Max  (M = {M})")
        for event in resample_events:
            ax_ess.axvline(event, color="limegreen", linewidth=2, linestyle="--", alpha=0.7)
        if resample_events:
            ax_ess.axvline(resample_events[0], color="limegreen", linewidth=2, linestyle="--", alpha=0.7,
                           label="Resampling + rejuvenation")
        ax_ess.set_xlim(-0.5, n_timesteps - 0.5)
        ax_ess.set_ylim(0, M + 20)
        ax_ess.set_xlabel("Time step  t")
        ax_ess.set_ylabel("ESS  (θ-particles)")
        ax_ess.set_title("Outer ESS over time")
        ax_ess.legend(fontsize=8)
        ax_ess.grid(True, alpha=0.3)

        ax_cloud.fill_between(
            t_range,
            theta_mean_sp[: step + 1] - theta_std_sp[: step + 1],
            theta_mean_sp[: step + 1] + theta_std_sp[: step + 1],
            color="steelblue",
            alpha=0.25,
        )
        ax_cloud.plot(t_range, theta_mean_sp[: step + 1], color="steelblue", linewidth=2.5,
                      label="σ_proc estimate ±1σ")
        ax_cloud.axhline(sigma_proc_true, color="steelblue", linewidth=1.5, linestyle=":",
                         label=f"True σ_proc = {sigma_proc_true}")
        ax_cloud.fill_between(
            t_range,
            theta_mean_so[: step + 1] - theta_std_so[: step + 1],
            theta_mean_so[: step + 1] + theta_std_so[: step + 1],
            color="tomato",
            alpha=0.25,
        )
        ax_cloud.plot(t_range, theta_mean_so[: step + 1], color="tomato", linewidth=2.5,
                      label="σ_obs estimate ±1σ")
        ax_cloud.axhline(sigma_obs_true, color="tomato", linewidth=1.5, linestyle=":",
                         label=f"True σ_obs = {sigma_obs_true}")
        for event in resample_events:
            ax_cloud.axvline(event, color="limegreen", linewidth=2, linestyle="--", alpha=0.7)
        ax_cloud.set_xlim(-0.5, n_timesteps - 0.5)
        ax_cloud.set_ylim(0, 2.5)
        ax_cloud.set_xlabel("Time")
        ax_cloud.set_ylabel("Parameter estimate")
        ax_cloud.set_title("Online parameter filtering\nθ-weighted mean and ±1σ")
        ax_cloud.legend(fontsize=8)
        ax_cloud.grid(True, alpha=0.3)

        tight_layout()
        animated_frame(fig, pause=pause)

    close(fig)
    print("\n=== ANIMATED SMC² RESULTS ===")
    print("Final parameter estimates (θ-weighted mean ± std):")
    print(f"  σ_proc: {theta_mean_sp[-1]:.3f} ± {theta_std_sp[-1]:.3f}  (true: {sigma_proc_true})")
    print(f"  σ_obs:  {theta_mean_so[-1]:.3f} ± {theta_std_so[-1]:.3f}  (true: {sigma_obs_true})")
    print(f"Outer resampling + rejuvenation events at t = {resample_events}")
    print(f"Total rejuvenation events: {len(resample_events)}")
    return {
        "theta_mean_sp": theta_mean_sp,
        "theta_mean_so": theta_mean_so,
        "theta_std_sp": theta_std_sp,
        "theta_std_so": theta_std_so,
        "outer_ess_hist": outer_ess_hist,
        "resample_events": resample_events,
        "smc2_filt_m": smc2_filt_m,
        "smc2_filt_s": smc2_filt_s,
        "kf_means_true": kf_means_true,
        "kf_stds_true": kf_stds_true,
        "true_x": true_x_1d,
        "observations": observations_1d,
    }


__all__ = [
    "f",
    "g",
    "get_demo_grid",
    "plot_integral_function",
    "plot_regular_sampling",
    "plot_uniform_sampling",
    "plot_narrow_function_uniform_sampling",
    "plot_importance_sampling",
    "plot_ideal_importance_sampling",
    "animate_rejection_sampling",
    "build_metropolis_patch_collection",
    "metropolis_step",
    "animate_metropolis_sampling",
    "animate_single_chain_metropolis",
    "plot_metropolis_hastings_bias",
    "load_text_message_data",
    "plot_count_data",
    "run_single_change_mcmc",
    "hdi_of_mcmc",
    "plot_hdi",
    "plot_single_change_posteriors",
    "run_three_segment_mcmc",
    "plot_three_segment_posteriors",
    "kalman_smoother_up_to",
    "systematic_resample",
    "generate_linear_gaussian_data",
    "animate_sis_joint_demo",
    "animate_sis_optimal_joint_demo",
    "animate_optimal_resampling_demo",
    "animate_prior_resampling_demo",
    "run_particle_filter_dimension_experiment",
    "plot_curse_of_dimensionality_summary",
    "animate_high_dim_prior_resampling",
    "animate_high_dim_optimal_resampling",
    "animate_high_dim_filtering_prior",
    "animate_high_dim_filtering_optimal",
    "kf_step",
    "kf_full_loglik",
    "animate_smc2_filtering_demo",
    "run_smc2_demo",
]
