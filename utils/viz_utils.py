import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

result_folder = 'results'
if not os.path.exists(result_folder):
	os.makedirs(result_folder)

def contourplot(x, y, label, **kwargs):
    def euclidean(a, b): return np.sqrt(a ** 2 + b ** 2)
    xx, yy = np.meshgrid(x, y)
    zz = np.sqrt(xx**2+yy**2)  # np.linalg.norm((xx,yy))
    fig, ax = plt.subplots(1, 1)

    # ax = plt.axes(projection='3d')
    # cp = ax.contourf(xx, yy, zz, label=label)
    # if 'bnds' in kwargs.keys():
    # 	bnds = kwargs['bnds'] / np.linalg.norm(kwargs['bnds']) * 10
    # 	cp = ax.contour(xx, yy, bnds, label=label)
    # else :
    # cp = ax.contour(xx, yy, zz, label=label)
    cp = ax.contour(xx, yy, zz)
    # cp = ax.contour3D(xx, yy, zz, 10))

    # ax.set_zlim(0,150)
    fig.colorbar(cp)

    _num_samples = kwargs['cfg']['num_samples']
    x_pre, candidate_x_pre = kwargs['x_pre'][:
                                             _num_samples], kwargs['x_pre'][_num_samples:]
    y_pre, candidate_y_pre = kwargs['y_pre'][:
                                             _num_samples], kwargs['y_pre'][_num_samples:]
    x_post, candidate_x_post = kwargs['x_post'][:
                                                _num_samples], kwargs['x_post'][_num_samples:]
    y_post, candidate_y_post = kwargs['y_post'][:
                                                _num_samples], kwargs['y_post'][_num_samples:]
    ax.scatter(x_pre, y_pre,  # euclidean(x_pre, y_pre),
               label='pre')
    ax.scatter(x_post, y_post,  # euclidean(x_post, y_post),
               label='post')
    if len(kwargs['x_pre']) > _num_samples:
        ax.scatter(candidate_x_pre, candidate_y_pre,  # euclidean(candidate_x_pre, candidate_y_pre),
                   label='candidate_pre')
        ax.scatter(candidate_x_post, candidate_y_post,  # euclidean(candidate_x_post, candidate_y_post),
                   label='candidate_post')
        plt.xlim([-120, 120])
        plt.ylim([-120, 120])
        ax.set_title('contour plot - '+label)
        ax.set_xlabel('x')
    ax.set_xlabel('y')
    ax.legend(fontsize=12, loc='upper left')  # legend position
    plt.savefig('results/contour plot_%s.png' % label)
    # plt.show()
    # plt.close()


def draw_graphs(train_data, BO_data, x_post, y_post, cfg, num_iter):
    sns.distplot(train_data[:, 0], label='Train data x',
                 color='blue', norm_hist=True)
    sns.distplot(BO_data[:, 0], label='Candidate x',
                 color='red', norm_hist=True)
    plt.title('PreX histogram(iter %d)' % num_iter)
    plt.xlabel('Offset')
    plt.ylabel('Frequency')
    plt.ylim([0, 0.06])
    plt.xlim([-200, 200])
    plt.legend(fontsize=12, loc='upper left')  # legend position
    plt.savefig('results/PreX histogram_%d.png' % num_iter)
    # plt.show()
    plt.close()

    sns.distplot(train_data[:, 1], label='Train data y',
                 color='blue', norm_hist=True)
    sns.distplot(BO_data[:, 1], label='Candidate y',
                 color='red', norm_hist=True)
    plt.title('PreY histogram(iter %d)' % num_iter)
    plt.xlabel('Offset')
    plt.ylabel('Frequency')
    plt.ylim([0, 0.06])
    plt.xlim([-200, 200])
    plt.legend(fontsize=12, loc='upper left')  # legend position
    plt.savefig('results/PreY histogram_%d.png' % num_iter)
    # plt.show()
    plt.close()

    sns.distplot(x_post[:cfg['num_samples']],
                 label='Train data x', color='blue', norm_hist=False)
    sns.distplot(x_post[cfg['num_samples']:],
                 label='Candidate x', color='red', norm_hist=False)
    plt.title('PostX histogram(iter %d)' % num_iter)
    plt.xlabel('Offset')
    plt.ylabel('Frequency')
    plt.ylim([0, 0.06])
    plt.xlim([-200, 200])
    plt.legend(fontsize=12, loc='upper left')  # legend position
    plt.savefig('results/PostX histogram_%d.png' % num_iter)
    # plt.show()
    plt.close()

    sns.distplot(y_post[:cfg['num_samples']],
                 label='Train data y', color='blue', norm_hist=False)
    sns.distplot(x_post[cfg['num_samples']:],
                 label='Candidate y', color='red', norm_hist=False)
    plt.title('PostY histogram(iter %d)' % num_iter)
    plt.xlabel('Offset')
    plt.ylabel('Frequency')
    plt.ylim([0, 0.06])
    plt.xlim([-200, 200])
    plt.legend(fontsize=12, loc='upper left')  # legend position
    plt.savefig('results/PostY histogram_%d.png' % num_iter)
    # plt.show()
    plt.close()

    # Pre vs Post
    plt.scatter(train_data[:, 0], x_post[:cfg['num_samples']], label='train')
    plt.scatter(BO_data[:, 0], x_post[cfg['num_samples']:], label='candidate')
    plt.title('Pre-Post X(iter %d)' % num_iter)
    plt.xlabel('PreX')
    plt.ylabel('PostX')
    plt.legend(fontsize=12, loc='upper left')  # legend position
    plt.savefig('results/Pre-Post X plot_%d.png' % num_iter)
    # plt.show()
    plt.close()

    plt.scatter(train_data[:, 1], y_post[:cfg['num_samples']], label='train')
    plt.scatter(BO_data[:, 1], y_post[cfg['num_samples']:], label='candidate')
    plt.title('Pre-Post Y(iter %d)' % num_iter)
    plt.xlabel('PreY')
    plt.ylabel('PostY')
    plt.legend(fontsize=12, loc='upper left')  # legend position
    plt.savefig('results/Pre-Post Y plot_%d.png' % num_iter)
    # plt.show()
    plt.close()


# def plot(X, y, fig):
#     # scatter plot of inputs and real objective function
#     plt.scatter(X, y)
#     # line plot of surrogate function across domain
#     Xsamples = asarray(arange(0, 1, 0.001))
#     Xsamples = Xsamples.reshape(len(Xsamples), 1)
#     ysamples, _ = surrogate(model, Xsamples)
#     plt.plot(Xsamples, ysamples)
#     # show the plot
#     plt.show()

# def simpleplot(X, y, ax, title='points', xlabel='x', ylabel='y', legend='', label=''):
#     indices_to_order_by = X.squeeze().argsort() # dim squeeze: 10x1 -> 10
#     x_ordered = X[indices_to_order_by]
#     y_ordered = y[indices_to_order_by]
#     ax.plot(x_ordered, y_ordered, marker="o", markerfacecolor="r", label=label)
#     # ax.scatter(X, y, label=legend)
#     ax.legend(loc='best')
#     ax.set_title(title)
#     ax.set_xlabel(xlabel)
#     ax.set_ylabel(ylabel)

# def contourplot(x, y, label):
#     xx, yy = np.meshgrid(x, y)
#     zz = np.sqrt(xx**2+yy**2) #np.linalg.norm((xx,yy))
#     fig, ax = plt.subplots(1,1)
#     cp = ax.contourf(xx, yy, zz, label=label)
#     fig.colorbar(cp)
#     ax.set_title('contour plot - '+label)
#     ax.set_xlabel('x')
#     ax.set_xlabel('y')
#     plt.show()
