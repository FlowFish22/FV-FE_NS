import matplotlib.pyplot as plt
# line cyclers for quality plots
from cycler import cycler
line_cycler   = (cycler(color=["red", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["-", "--", "-.", ":", ":", "-", "-."]))
marker_cycler = (cycler(color=["red", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))

# matplotlib's standard cycler
standard_cycler = cycler("color", ["red", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])

plt.rc("axes", prop_cycle=line_cycler)

plt.rc("text", usetex=True)
plt.rc("text.latex", preamble=r"\usepackage{newpxtext}\usepackage{newpxmath}\usepackage{commath}\usepackage{mathtools}") 
plt.rc("font", family="serif", size=18.)
plt.rc("savefig", dpi=200)
plt.rc("legend", loc="Best", fontsize="small", fancybox=True, framealpha=0.25)
plt.rc("lines", linewidth=2.5, markersize=1, markeredgewidth=2.5)
plt.rcParams["text.usetex"] =True


# plots
plt.close("all")

import numpy as np
data = np.loadtxt('sol_exact.dat')
x = data[:, 0]
y = data[:, 1]
#x = sorted(x)
data1 = np.loadtxt('sol.dat')
x1 = data1[:, 0]
y1 = data1[:, 1]
#x1 = sorted(x1)
#data2 = np.loadtxt('cross_section_vorticity_T1p67eps0p001.dat')
#x2 = data2[:, 0]
#y2 = data2[:, 2]
#x2 = sorted(x2)
#data3 = np.loadtxt('cross_section_vorticity_T1p67eps0p0001.dat')
#x3 = data3[:, 0]
#y3 = data3[:, 2]
#x3 = sorted(x3)
#data4 = np.loadtxt('crsn_vorticity_t1p256_eps0p001.dat')
#x4 = data1[:, 0]
#y4 = data1[:, 2]
#x4 = sorted(x4)

#x6 = np.linspace(0., 1., 200)

plt.rc("axes", prop_cycle=line_cycler)
plt.figure()
plt.plot(x, y, marker= "o", color= "r", label="Exact soln")
plt.plot(x1, y1, label="Explicit scheme")
#plt.plot(x1, y1, label="_nolegend_")
#plt.plot(x2, y2, label="$\epsilon = 0.001$")
#plt.plot(x3, y3, label="$\epsilon = 0.0001$")
#plt.plot(x4, y4, label="$T=1 p$")

#plt.plot(x6, 3*np.sin(np.pi*2*x), label="$3 \sin(2\pi x)$")
#plt.plot(x, 3*np.cos(np.pi*2*x), label="$3 \cos(2\pi x)$")
plt.minorticks_on()
plt.legend()
plt.xlabel("$ x $")
#plt.xlim(x[0], x[-1])
#plt.yscale("log")
#plt.xscale("log")
#plt.xlim(1.0e-20, 1.0e-12)
#plt.ylim(0.0, 1.0)
#plt.yticks(color='w')
#plt.yticks([])
plt.ylabel(r'$ u $')
plt.title("Exact vs numerical soln $T = 1.5$")
#plt.show ()
plt.savefig("comparison_num_exact", bbox_inches="tight")

