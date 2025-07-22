import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy import stats
from scipy.stats import pearsonr
from tools.excel import r4
import matplotlib.pyplot as plt
import math
size=13
# size=26
from matplotlib import rcParams
# from simple_colors import *
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')

import statsmodels.api as sm
def correlation(x,y,x_lab,y_lab,path,cut_x=None,cut_y=None,marker="^",c="blue",size=13):

    fig, ax = plt.subplots()
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    r_value, slope = plot_part(cut_x, cut_y, x, x_lab, y, y_lab,marker=marker,c=c)

    # plt.legend([f'slop: {round(slope,2)}',f"r-value:{round(r_value,2)}; p-value:{tmp}"],fontsize=size)
    plt.legend([f'slop: {r4(slope,3)}',f"r-value:{r4(r_value,3)}"],fontsize=size,loc=2)
    # plt.show()
    fig.savefig(path,dpi=400,bbox_inches='tight', pad_inches=0.03)




def bland_altman(data1, data2,path,x_lab='Mean',y_lab='Difference',size=13,locator=None):
    fig, ax = plt.subplots()
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    x_features=np.array(data1)
    y_features=np.array(data2)
    #

    if locator:
        x = MultipleLocator(locator[0])  # x轴每10一个刻度
        y = MultipleLocator(locator[1])  # y轴每15一个刻度
        # 设置刻度间隔
        ax = plt.gca()
        ax.xaxis.set_major_locator(x)
        ax.yaxis.set_major_locator(y)


    # sm.graphics.mean_diff_plot(data1, data2, ax=ax)


    mean = np.mean([x_features, y_features], axis=0)
    diff = y_features - x_features
    md = np.mean(diff)
    sd = np.std(diff, axis=0)
    CI_low = md - 1.96 * sd
    CI_high = md + 1.96 * sd

    with np.errstate(divide='ignore', invalid='ignore'):
        percentage_error = (diff / x_features) * 100

    with np.errstate(invalid='ignore'):
        percentage_error[percentage_error > 50] = 50
        percentage_error[percentage_error < -50] = -50

    # plt.scatter(mean, diff, c=percentage_error, edgecolors='face', cmap='jet')
    plt.scatter(mean, diff, c='black')
    plt.axhline(md, color='gray', linestyle='-')
    plt.axhline(md + 1.96 * sd, color='gray', linestyle='--')
    plt.axhline(md - 1.96 * sd, color='gray', linestyle='--')
    plt.ylabel(y_lab,fontsize=size)
    plt.xlabel(x_lab,fontsize=size)
    # plt.title(feature)
    x_range = np.max(mean) - np.min(mean)
    y_range = np.max(diff) - np.min(diff)



    axis_extra_range = 0.1
    # plt.xlim([np.min(mean) - (x_range * axis_extra_range),
    #           np.max(mean) + (x_range * axis_extra_range)])
    # plt.ylim([np.min(diff) - (y_range * axis_extra_range),
    #           np.max(diff) + (y_range * axis_extra_range)])

    # xOutPlot = np.min(mean) + (np.max(mean) - np.min(mean)) * 1.14
    xOutPlot = np.min(mean) + (np.max(mean) - np.min(mean)) *0.8

    plt.text(xOutPlot, md - 1.75 * sd,
             r'-1.96SD: ' + "" + "%.2f" % CI_low,
             ha="center",
             va="center",fontsize=18
             )
    plt.text(xOutPlot, md + 1.75 * sd,
             r'+1.96SD: ' + "" + "%.2f" % CI_high,
             ha="center",
             va="center",fontsize=18
             )
    plt.text(xOutPlot, md+ 0.15 * sd,
             r'Mean: ' + "" + "%.2f" % md,
             ha="center",
             va="center",fontsize=18
             )

    # plt.colorbar()
    fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.03)
    plt.show()
    # plt.savefig('{}/{}-{}-Bland-Altman.png'.format(visualisation_path, algorithm_name, feature))
    # plt.clf()


# from scipy.stats import linregress
# import numpy as np
# import plotly.graph_objects as go
# def bland_altman(data1, data2,path, data1_name='A', data2_name='B', subgroups=None, plotly_template='none', annotation_offset=0.05, plot_trendline=True, n_sd=1.96,*args, **kwargs):
#     data1 = np.array(data1)
#     data2 = np.array(data2)
#
#     data1 = np.asarray( data1 )
#     data2 = np.asarray( data2 )
#     mean = np.mean( [data1, data2], axis=0 )
#     diff = data1 - data2  # Difference between data1 and data2
#     md = np.mean( diff )  # Mean of the difference
#     sd = np.std( diff, axis=0 )  # Standard deviation of the difference
#
#
#     fig = go.Figure()
#
#     if plot_trendline:
#         slope, intercept, r_value, p_value, std_err = linregress(mean, diff)
#         trendline_x = np.linspace(mean.min(), mean.max(), 10)
#         fig.add_trace(go.Scatter(x=trendline_x, y=slope*trendline_x + intercept,
#                                  name='Trendline',
#                                  mode='lines',
#                                  line=dict(
#                                         width=4,
#                                         dash='dot')))
#     if subgroups is None:
#         fig.add_trace( go.Scatter( x=mean, y=diff, mode='markers', **kwargs))
#     else:
#         for group_name in np.unique(subgroups):
#             group_mask = np.where(np.array(subgroups) == group_name)
#             fig.add_trace( go.Scatter(x=mean[group_mask], y=diff[group_mask], mode='markers', name=str(group_name), **kwargs))
#
#
#
#     fig.add_shape(
#         # Line Horizontal
#         type="line",
#         xref="paper",
#         x0=0,
#         y0=md,
#         x1=1,
#         y1=md,
#         line=dict(
#             # color="Black",
#             width=6,
#             dash="dashdot",
#         ),
#         name=f'Mean {round( md, 2 )}',
#     )
#     fig.add_shape(
#         # borderless Rectangle
#         type="rect",
#         xref="paper",
#         x0=0,
#         y0=md - n_sd * sd,
#         x1=1,
#         y1=md + n_sd * sd,
#         line=dict(
#             color="SeaGreen",
#             width=2,
#         ),
#         fillcolor="LightSkyBlue",
#         opacity=0.4,
#         name=f'±{n_sd} Standard Deviations'
#     )
#
#     # Edit the layout
#     fig.update_layout( title=f'Bland-Altman Plot for {data1_name} and {data2_name}',
#                        xaxis_title=f'Average of {data1_name} and {data2_name}',
#                        yaxis_title=f'{data1_name} Minus {data2_name}',
#                        template=plotly_template,
#                        annotations=[dict(
#                                         x=1,
#                                         y=md,
#                                         xref="paper",
#                                         yref="y",
#                                         text=f"Mean {round(md,2)}",
#                                         showarrow=True,
#                                         arrowhead=7,
#                                         ax=50,
#                                         ay=0
#                                     ),
#                                    dict(
#                                        x=1,
#                                        y=n_sd*sd + md + annotation_offset,
#                                        xref="paper",
#                                        yref="y",
#                                        text=f"+{n_sd} SD",
#                                        showarrow=False,
#                                        arrowhead=0,
#                                        ax=0,
#                                        ay=-20
#                                    ),
#                                    dict(
#                                        x=1,
#                                        y=md - n_sd *sd + annotation_offset,
#                                        xref="paper",
#                                        yref="y",
#                                        text=f"-{n_sd} SD",
#                                        showarrow=False,
#                                        arrowhead=0,
#                                        ax=0,
#                                        ay=20
#                                    ),
#                                    dict(
#                                        x=1,
#                                        y=md + n_sd * sd - annotation_offset,
#                                        xref="paper",
#                                        yref="y",
#                                        text=f"{round(md + n_sd*sd, 2)}",
#                                        showarrow=False,
#                                        arrowhead=0,
#                                        ax=0,
#                                        ay=20
#                                    ),
#                                    dict(
#                                        x=1,
#                                        y=md - n_sd * sd - annotation_offset,
#                                        xref="paper",
#                                        yref="y",
#                                        text=f"{round(md - n_sd*sd, 2)}",
#                                        showarrow=False,
#                                        arrowhead=0,
#                                        ax=0,
#                                        ay=20
#                                    )
#                                ])
#     # fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.03)
#     fig.show()
#
#     return fig
#
# def bland_altmanV2(data1, data2,path):
#     data1=np.array(data1)
#     data2=np.array(data2)
#     fig, ax = plt.subplots()
#     plt.xticks(fontsize=size)
#     plt.yticks(fontsize=size)
#     _plot_bland_altman(data1,data2)
#     plt.title('Bland-Altman Plot')
#     fig.savefig(path, dpi=400, bbox_inches='tight', pad_inches=0.03)
#
# def _plot_bland_altman(data1, data2, *args, **kwargs):
#     data1     = np.asarray(data1)
#     data2     = np.asarray(data2)
#     mean      = np.mean([data1, data2], axis=0)
#     diff      = data1 - data2                   # Difference between data1 and data2
#     md        = np.mean(diff)                   # Mean of the difference
#     sd        = np.std(diff, axis=0)            # Standard deviation of the difference
#
#     plt.scatter(mean, diff, *args, **kwargs)
#     plt.axhline(md,           color='gray', linestyle='--')
#     plt.axhline(md + 1.96*sd, color='gray', linestyle='--')
#     plt.axhline(md - 1.96*sd, color='gray', linestyle='--')

def plot_part(cut_x, cut_y, x, x_lab, y, y_lab,marker,c,x_tick=None):
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


    #Calculate the least squares fit of the data. m refers to the slope
    #of the line and b refers to the y-intercept.
    # m,b = np.linalg.lstsq(A,Xdata)[0]
    #Calculate the Pearson correlation coefficient and the 2-tailed p-value).
    np_tmp_x=np.array(tmp_x)
    np_tmp_y=np.array(tmp_y)

    A = np.vstack([np_tmp_x, np.ones(len(np_tmp_x))]).T
    # Calculate the least squares fit of the data. m refers to the slope
    # of the line and b refers to the y-intercept.
    m, b = np.linalg.lstsq(A, np_tmp_y)[0]
    print(f"slop: {m},{b}")
    R_and_P = stats.pearsonr(np_tmp_x, np_tmp_y)
    print(f"R_and_P {R_and_P[0]} {R_and_P[1]}")

    slope, intercept, r_value, p_value, std_err = stats.linregress(tmp_x, tmp_y)
    # print(f" R-squared: {r_value ** 2} pvalue: {p_value}")
    print(f" R: {r_value} pvalue: {p_value}")
    print(f" slop: {slope} intercept: {intercept}")

    # r_value1,p_value1=pearsonr(x,y)
    # print(f"pearsonr: {r_value1}  p_valeu:{p_value1}")
    # print(f" R: {r_value}")
    # plt.scatter(x, y, s=area)
    tmp1 = plt.scatter(x, y, s=area,  alpha=1, marker=marker, edgecolors='k', linewidths=1.5,c=c)
    # tmp2=plt.scatter(outlier_x, outlier_y, s=area, c='red', alpha=1, marker="o", edgecolors='k', linewidths=1.5)
    if x_tick==None:
        pass
    else:
        x=x_tick
    yfit = [intercept + slope * xi for xi in x]
    tmp3 = plt.plot(x, yfit,c=c,label=str())  # darkgreen, midnightblue
    # slope, intercept, r value, p value, std err = stats.linregress(x1, y)# print("R-squared: of" 8 r value**2)
    # plt.scatter(x1, y, s=area, c='’, alpha=1, marker='v’,edgecolors='darkgrey', linewidths=1.5)# yfit = [intercept + slope * xi for xi in x11
    # plt.plot(x1, yfit, c='darkorange’) #darkgreen, midnightblue
    plt.xlabel(x_lab, fontsize=size)
    plt.ylabel(y_lab, fontsize=size)
    tmp = r4(p_value, 3)
    r_value=r_value ** 2
    if r_value<0.001:
        r_value=0.00
    return r_value, slope





def correlation2X(y, x1_type, x1, x2_type, x2, f_y_lab, f_x_lab, path, cut_y=None, cut_x1=None, cut_x2=None, x1_color='red', x2_color='blue',lenged_loc=4):
    # config = {
    #     "font.family": 'serif',
    #     # "font.size": 12,  # 相当于小四大小
    #     "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    #     "font.serif": ['SimSun'],  # 宋体
    #     'axes.unicode_minus': False  # 处理负号，即-号
    # }
    # rcParams.update(config)

    fig, ax = plt.subplots()
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    r_value1, slope1=plot_part( cut_x1,cut_y, x1, f_x_lab, y, f_y_lab, marker="^", c=x1_color,x_tick=x1+x2)
    r_value2, slope2=plot_part(cut_x2,cut_y, x2, f_x_lab, y, f_y_lab,  marker="o", c=x2_color,x_tick=x1+x2)
    # plt.legend([f'slop: {round(slope,2)}',f"r-value:{round(r_value,2)}; p-value:{tmp}"],fontsize=size)


    # plt.legend([f'slop: {r4(slope1,3)}',f'slop: {r4(slope2,3)}',f"r²:{round(r_value1,3)} ({y1_type})",f"r²:{round(r_value2,3)} ({y2_type})"],fontsize=size,loc=4)
    plt.legend([f"R²: {round(r_value1,3)} ({x1_type})", f"R²: {round(r_value2, 3)} ({x2_type})"], fontsize=size-4, loc=lenged_loc,handlelength=1)
    # plt.show()

    fig.savefig(path,dpi=400,bbox_inches='tight', pad_inches=0.03)






def correlation2Y(x,y1_type, y1,y2_type, y2, x_lab, y_lab, path, cut_x=None, cut_y1=None,cut_y2=None,y1_color='red',y2_color='blue',lenged_loc=4):
    # config = {
    #     "font.family": 'serif',
    #     # "font.size": 12,  # 相当于小四大小
    #     "mathtext.fontset": 'stix',  # matplotlib渲染数学字体时使用的字体，和Times New Roman差别不大
    #     "font.serif": ['SimSun'],  # 宋体
    #     'axes.unicode_minus': False  # 处理负号，即-号
    # }
    # rcParams.update(config)

    fig, ax = plt.subplots()
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    r_value1, slope1=plot_part(cut_x, cut_y1, x, x_lab, y1, y_lab,marker="^",c=y1_color)
    r_value2, slope2=plot_part(cut_x, cut_y2, x, x_lab, y2, y_lab,marker="o",c=y2_color)
    # plt.legend([f'slop: {round(slope,2)}',f"r-value:{round(r_value,2)}; p-value:{tmp}"],fontsize=size)


    # plt.legend([f'slop: {r4(slope1,3)}',f'slop: {r4(slope2,3)}',f"r²:{round(r_value1,3)} ({y1_type})",f"r²:{round(r_value2,3)} ({y2_type})"],fontsize=size,loc=4)
    plt.legend([f"R²: {round(r_value1,3)} ({y1_type})",f"R²: {round(r_value2,3)} ({y2_type})"],fontsize=size-4,loc=lenged_loc,handlelength=1)
    # plt.show()

    fig.savefig(path,dpi=400,bbox_inches='tight', pad_inches=0.03)