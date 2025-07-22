
from scipy import stats
'''
https://vimsky.com/examples/detail/python-method-scipy.stats.ttest_rel.html
https://towardsdatascience.com/one-tailed-or-two-tailed-test-that-is-the-question-1283387f631c
'''

def ttest_alt(a, b, alternative='two-sided'):
    tt, tp = stats.ttest_rel(a, b)

    if alternative == 'greater':
        if tt > 0:
            tp = 1 - (1-tp)/2
        else:
            tp /= 2
    elif alternative == 'less':
        if tt <= 0:
            tp /= 2
        else:
            tp = 1 - (1-tp)/2

    return tt, tp


