from utils import *


def list_distances_for_histogram_in_general(selected_points, only_use_following_points, distance_mode):
    update_point_list_with_neighbours(selected_points, only_use_following_points)
    list_of_distances = []
    for point in selected_points:
        if point.distance_to_nearest_neighbour is not None:
            list_of_distances.append(point.distance_to_nearest_neighbour)
    return list_of_distances





dist_1 = GaussianDistribution(3, 1)
dist_2 = GaussianDistribution(3, 1)
dist_3 = add_gaussians(dist_1, dist_2)
ax1 = plt.subplot(411)
dist_1.plot_point(color='r', linestyle='-', linewidth=1.)
ax2 = plt.subplot(412, sharex = ax1)
dist_2.plot_point(color='g', linestyle='-', linewidth=1.)
ax3 = plt.subplot(413, sharex= ax1)
dist_1.plot_point(color='r', linestyle='-', linewidth=1.)
dist_2.plot_point(color='g', linestyle='-', linewidth=1.)
ax4 = plt.subplot(414, sharex=ax1)
dist_3.plot_point(color='b', linestyle='-', linewidth=1.)
plt.show()


# plot moved points
deltas = np.linspace(1, 10, NUM_OF_DELTAS)
colors = np.linspace(0, 1, NUM_OF_DELTAS)

for delta, color in zip(deltas, colors):
    right_raw = []
    moved_right = []
    for point in right_points:
        right_raw.append(point.x)
        moved_right.append(point.x + delta)

    x = [color for _ in range(len(right_raw))]
    plt.plot(moved_right, x, marker='o', ls=' ', mfc=plt.cm.jet(color), label=str(delta))

x = [-1 for _ in range(len(right_raw))]
plt.plot(right_raw, x, 'rx', label='original')
plt.legend()
plt.show()


# durbin-watson-test
colors = np.linspace(0, 1, NUM_OF_DELTAS)
ergebnisse_durbin_watson_test = []
for delta, colour in zip(deltas, colors):
    summe_nenner = 0
    summe_zaehler = 0
    for point in right_points:
        point.x = delta + point.x

    moved_gaussian_pointlist = convert_point_x_list_to_gaussian(right_points, VARIANCE)
    plot_list_of_gaussian_points(list_of_points=moved_gaussian_pointlist, color=plt.cm.jet(colour), linestyle='-',
                                 linewidth=1.)
    for gaussian_point, moved_gaussian_point in zip(gaussian_point_list_right, moved_gaussian_pointlist):
        summe_zaehler += (gaussian_point.mean - moved_gaussian_point.mean)**2
        summe_nenner += gaussian_point.mean**2
    ergebnisse_durbin_watson_test.append(summe_zaehler/summe_nenner)

print(ergebnisse_durbin_watson_test)
plt.show()


# autocorrelation

def autocorr1(x,lags):
    '''numpy.corrcoef, partial'''

    corr=[1. if l==0 else np.corrcoef(x[l:],x[:-l])[0][1] for l in lags]
    return np.array(corr)


def autocorr2(x,lags):
    '''manualy compute, non partial'''

    mean=np.mean(x)
    var=np.var(x)
    xp=x-mean
    corr=[1. if l==0 else np.sum(xp[l:]*xp[:-l])/len(x)/var for l in lags]

    return np.array(corr)


def autocorr3(x,lags):
    '''fft, pad 0s, non partial'''

    n=len(x)
    # pad 0s to 2n-1
    ext_size=2*n-1
    # nearest power of 2
    fsize=2**np.ceil(np.log2(ext_size)).astype('int')

    xp=x-np.mean(x)
    var=np.var(x)

    # do fft and ifft
    cf=np.fft.fft(xp,fsize)
    sf=cf.conjugate()*cf
    corr=np.fft.ifft(sf).real
    corr=corr/var/n

    return corr[:len(lags)]


def autocorr4(x,lags):
    '''fft, don't pad 0s, non partial'''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean

    cf=np.fft.fft(xp)
    sf=cf.conjugate()*cf
    corr=np.fft.ifft(sf).real/var/len(x)

    return corr[:len(lags)]


def autocorr5(x,lags):
    '''numpy.correlate, non partial'''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]

autocor_1 = autocorr1(mean_array_right, range(14))
print('autocorr_1 = {}'.format(autocor_1))
autocor_2 = autocorr2(mean_array_right, range(14))
print('autocorr_2 = {}'.format(autocor_2))
autocor_3 = autocorr3(mean_array_right, range(14))
print('autocorr_3 = {}'.format(autocor_3))
autocor_4 = autocorr4(mean_array_right, range(14))
print('autocorr_4 = {}'.format(autocor_4))
autocor_5 = autocorr5(mean_array_right, range(14))
print('autocorr_5 = {}'.format(autocor_5))


"""
sci_corr_same = signal.correlate(mean_array_right, mean_array_right, 'same')
sci_corr_valid = signal.correlate(mean_array_right, mean_array_right, 'valid')
sci_corr_full = signal.correlate(mean_array_right, mean_array_right, 'full')

print('sci_corr = {}'.format(sci_corr_same))
print('sci_corr = {}'.format(sci_corr_valid))
print('sci_corr = {}'.format(sci_corr_full))
"""



