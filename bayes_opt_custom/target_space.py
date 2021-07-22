import sys
import numpy as np
from .util import ensure_rng
import data as custom # ../data
import torch
import time
from matplotlib import pyplot as plt

def _hashable(x):
    """ ensure that an point is hashable by a python dict """
    return tuple(map(float, x))


class TargetSpace(object):
    """
    Holds the param-space coordinates (X) and target values (Y)
    Allows for constant-time appends while ensuring no duplicates are added

    Example
    -------
    >>> def target_func(p1, p2):
    >>>     return p1 + p2
    >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
    >>> space = TargetSpace(target_func, pbounds, random_state=0)
    >>> x = space.random_points(1)[0]
    >>> y = space.register_point(x)
    >>> assert self.max_point()['max_val'] == y
    """
    def __init__(self, target_func, pbounds, random_state=None, cfg=None):
        """
        Parameters
        ----------
        target_func : function
            Function to be maximized.

        pbounds : dict
            Dictionary with parameters names as keys and a tuple with minimum
            and maximum values.

        random_state : int, RandomState, or None
            optionally specify a seed for a random number generator
        """
        self.random_state = ensure_rng(random_state)
        
        self.cfg = cfg
        
        num_samples = self.cfg['train_cfg']['num_samples']
        num_workers = self.cfg['train_cfg']['num_workers']
        data_cfg = self.cfg['data_cfg']
        
        # start = time.time()
        self.dataset = custom.CustomData(input_dim=2, num_samples=num_samples, type='toy', cfg=data_cfg)
        self.index = 0
        # self.dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=1, shuffle=True, num_workers=num_workers)
        # end = time.time()
        # print('loading toy data (size: {}): {:.3f} sec'.format(len(self.dataloader.dataset), (end-start)))

        # The function to be optimized
        self.target_func = target_func

        # Get the name of the parameters
        self._keys = sorted(pbounds)
        # Create an array with parameters bounds
        self._bounds = np.array(
            [item[1] for item in sorted(pbounds.items(), key=lambda x: x[0])],
            dtype=np.float
        )

        # preallocated memory for X and Y points
        self._params = np.empty(shape=(0, self.dim))
        self._target = np.empty(shape=(0))

        # keep track of unique points we have seen so far
        self._cache = {}

    def __contains__(self, x):
        return _hashable(x) in self._cache

    def __len__(self):
        assert len(self._params) == len(self._target)
        return len(self._target)

    @property
    def empty(self):
        return len(self) == 0

    @property
    def params(self):
        return self._params

    @property
    def target(self):
        return self._target

    @property
    def dim(self):
        return len(self._keys)

    @property
    def keys(self):
        return self._keys

    @property
    def bounds(self):
        return self._bounds

    def params_to_array(self, params):
        try:
            assert set(params) == set(self.keys)
        except AssertionError:
            raise ValueError(
                "Parameters' keys ({}) do ".format(sorted(params)) +
                "not match the expected set of keys ({}).".format(self.keys)
            )
        return np.asarray([params[key] for key in self.keys])

    def array_to_params(self, x):
        try:
            assert len(x) == len(self.keys)
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        
        # print('x:', x, ' length:', len(x))
        # print('self.keys:', self.keys)
        return dict(zip(self.keys, x))

    def _as_array(self, x):
        try:
            x = np.asarray(x, dtype=float)
        except TypeError:
            x = self.params_to_array(x)

        x = x.ravel()
        try:
            assert x.size == self.dim
        except AssertionError:
            raise ValueError(
                "Size of array ({}) is different than the ".format(len(x)) +
                "expected number of parameters ({}).".format(len(self.keys))
            )
        return x

    def register(self, params, target):
        """
        Append a point and its target value to the known data.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        y : float
            target function value

        Raises
        ------
        KeyError:
            if the point is not unique

        Notes
        -----
        runs in ammortized constant time

        Example
        -------
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(lambda p1, p2: p1 + p2, pbounds)
        >>> len(space)
        0
        >>> x = np.array([0, 0])
        >>> y = 1
        >>> space.add_observation(x, y)
        >>> len(space)
        1
        """
        x = self._as_array(params)
        if x in self:
            raise KeyError('Data point {} is not unique'.format(x))

        # Insert data into unique dictionary
        self._cache[_hashable(x.ravel())] = target

        self._params = np.concatenate([self._params, x.reshape(1, -1)])
        self._target = np.concatenate([self._target, [target]])

    def probe(self, params):
        """
        Evaulates a single point x, to obtain the value y and then records them
        as observations.

        Notes
        -----
        If x has been previously seen returns a cached value of y.

        Parameters
        ----------
        x : ndarray
            a single point, with len(x) == self.dim

        Returns
        -------
        y : float
            target function value.
        """
        sample_type = 'initial'
        if isinstance(params, dict):
            sample_type = 'candidate'
        
        x = self._as_array(params)
        # params (x) not yet registered in target space (self._cache)
        try:
            target = self._cache[_hashable(x)]
        except KeyError: 
            # if same key, value appeared in self._cache (overlap), do not register in target space
            params = dict(zip(self._keys, x))
            target = self.target_func(**params)
            self.register(x, target)
            
            # plot candidate
            if sample_type == 'candidate':
                ax = plt.gca(projection='3d')
                ax.plot(params['x1'], params['x2'], -target[0][0].numpy(), 'r.', markersize=5, label=f'{-target[0][0].numpy():.3f}')
                ax.legend(loc='best')
                ax.set_xlabel('x1')
                ax.set_ylabel('x2')
                ax.set_zlabel('y')
                ax.grid('off')
                # ax.set_xlim([-10,10])
                # ax.set_ylim([-10,10])
                # ax.set_zlim([0,30])
                ax.view_init(elev=15, azim=-45)
                plt.pause(0.00001)
            
            # print('registered target in search space:', x, target)
        return target

    def random_sample(self):
        """
        Creates random points within the bounds of the space.

        Returns
        ----------
        data: ndarray
            [num x dim] array points with dimensions corresponding to `self._keys`

        Example
        -------
        >>> target_func = lambda p1, p2: p1 + p2
        >>> pbounds = {'p1': (0, 1), 'p2': (1, 100)}
        >>> space = TargetSpace(target_func, pbounds, random_state=0)
        >>> space.random_points(1)
        array([[ 55.33253689,   0.54488318]])
        """
        # TODO: support integer, category, and basic scipy.optimize constraints
        data = np.empty((1, self.dim))
        for col, (lower, upper) in enumerate(self._bounds):
            data.T[col] = self.random_state.uniform(lower, upper, size=1)
        return data.ravel()
    
    # ==========================================================================
    # TODO: sample from toy data pool
    # custom
    def sample_single(self):
        data = np.empty((1, self.dim))
        # for col, (lower, upper) in enumerate(self._bounds):
        #     data.T[col] = self.random_state.uniform(lower, upper, size=1)
        # print(data, data.shape)                 # [[value, value]] with shape (1,2)
        # print(data.ravel(), data.ravel().shape) # [value, value] with shape (2, )
        
        # sample_batch = next(iter(self.dataloader))
        # data, _ = sample_batch[0].numpy(), sample_batch[1].numpy() # list type
        
        data = self.dataset[self.index][0]
        self.index += 1
        
        return data.ravel()
    
    # custom
    def sample_multiple(self, init_points):
        # points = []
        # for _ in range(init_points):
        #     # data = np.empty((1, self.dim))
        #     # for col, (lower, upper) in enumerate(self._bounds):
        #     #     data.T[col] = self.random_state.uniform(lower, upper, size=1)
        #     # points.append(data.ravel())
        #     s = time.time()
        #     points.append(self.sample_single())
        #     e = time.time()
            # print('dataloader sample single time: {:.3f}'.format(e-s))
        
        points = self.dataset[self.index:self.index+init_points][0].tolist()
        self.index += init_points
        
        return points
    # ==========================================================================
    
    def max(self):
        """Get maximum target value found and corresponding parameters."""
        try:
            res = {
                'target': self.target.max(),
                'params': dict(
                    zip(self.keys, self.params[self.target.argmax()])
                )
            }
        except ValueError:
            res = {}
        return res

    def res(self):
        """Get all target values found and corresponding parameters."""
        params = [dict(zip(self.keys, p)) for p in self.params]

        return [
            {"target": target, "params": param}
            for target, param in zip(self.target, params)
        ]

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        for row, key in enumerate(self.keys):
            if key in new_bounds:
                self._bounds[row] = new_bounds[key]