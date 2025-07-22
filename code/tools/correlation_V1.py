import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import statsmodels.api as sm
def correlation(x,y,x_lab,y_lab,path,cut_x=None,cut_y=None,marker="^",c="black",size=13):

    fig, ax = plt.subplots()
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    slope,intercept, r_value,p_value = plot_part(cut_x, cut_y, x, x_lab, y, y_lab,marker=marker,c=c,size=size)

    # plt.legend([f'slop: {round(slope,2)}',f"r-value:{round(r_value,2)}; p-value:{tmp}"],fontsize=size)
    # plt.legend([f'slop: {r4(slope,3)}',f"r-value:{r4(r_value,3)}"],fontsize=size,loc=2)
    # plt.legend()
    # plt.legend([f'slop: {r4(slope,3)}', f'r-value:{r4(r_value,3)}'],fontsize=size,loc=2,handletextpad=0.0, handlelength=0)


    # plt.legend(markerscale=0)
    # plt.annotate(0.1, 0.1, "sss")
    # plt.show()
    fig.savefig(path,dpi=400,bbox_inches='tight', pad_inches=0.03)
    plt.show()


def plot_part(cut_x, cut_y, x, x_lab, y, y_lab,marker,c,x_tick=None,size=13):
    area = np.pi * 15
    tmp_x = []
    tmp_y = []
    outlier_x = []
    outlier_y = []
    for a, b in zip(x, y):
        if cut_x is not None and cut_y is not None:
            if a < cut_x or b < cut_y:
                outlier_x.append(a)
                outlier_y.append(b)
            else:
                tmp_x.append(a)
                tmp_y.append(b)
        elif cut_x is None and cut_y is not None:
            if b < cut_y:
                outlier_x.append(a)
                outlier_y.append(b)
            else:
                tmp_x.append(a)
                tmp_y.append(b)
        elif cut_x is not None and cut_y is None:
            if b < cut_x:
                outlier_x.append(a)
                outlier_y.append(b)
            else:
                tmp_x.append(a)
                tmp_y.append(b)
        else:
            tmp_x.append(a)
            tmp_y.append(b)


    np_tmp_x=np.array(tmp_x)
    np_tmp_y=np.array(tmp_y)

    # another method to calcuclate relationship
    A = np.vstack([np_tmp_x, np.ones(len(np_tmp_x))]).T
    m, b = np.linalg.lstsq(A, np_tmp_y)[0]
    print(f"slop: {m},{b}")
    R_and_P = stats.pearsonr(np_tmp_x, np_tmp_y)
    print(f"R_and_P {R_and_P[0]} {R_and_P[1]}")
    # another method to calcuclate relationship


    slope, intercept, r_value, p_value, std_err = stats.linregress(tmp_x, tmp_y)

    print(f" R: {r_value} pvalue: {p_value}")
    print(f" slop: {slope} intercept: {intercept}")


    plt.scatter(x, y, s=area,  alpha=1, marker=marker,linewidths=1.5,c=c)


    if x_tick==None:
        pass
    else:
        x=x_tick
    yfit = [intercept + slope * xi for xi in x]

    if p_value<0.01:
        p_value="P<0.01"
    else:
        # p_value=f"P={r4(p_value,2)}"
        p_value=f"P={round(p_value,2)}"

    plt.plot(x, yfit,c=c,label=f'R: {"%.2f" %r_value}, {p_value}')  # darkgreen, midnightblue
    # plt.plot(x, yfit,c=c,label=f'y={r4(slope,2)}x+{r4(intercept,2)}\nR:{r4(r_value,2)}, {p_value}')  # darkgreen, midnightblue
    # plt.plot(x, yfit,c=c,label=f'xxxx\nR:{r4(r_value,2)}, {p_value}')  # darkgreen, midnightblue

    plt.xlabel(x_lab, fontsize=size)
    plt.ylabel(y_lab, fontsize=size)

    plt.legend(fontsize=size, loc=2, handletextpad=0, handlelength=0, markerscale=0)

    return slope,intercept, r_value,p_value

