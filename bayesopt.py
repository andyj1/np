import sys
import warnings
import time
from bayesopt_custom.target_space import TargetSpace
from bayesopt_custom.event import Events, DEFAULT_EVENTS
from bayesopt_custom.logger import _get_default_logger
from bayesopt_custom.util import UtilityFunction, acq_max, ensure_rng

class Queue:
    def __init__(self):
        self._queue = []

    @property
    def empty(self):
        return len(self) == 0

    def __len__(self):
        return len(self._queue)

    def __next__(self):
        if self.empty:
            raise StopIteration("Queue is empty, no more objects to retrieve.")
        obj = self._queue[0]
        self._queue = self._queue[1:]
        return obj

    def next(self):
        return self.__next__()

    def add(self, obj):
        """Add object to end of queue."""
        # print('adding', obj, 'to queue...')
        self._queue.append(obj)
        

class Observable(object):
    """

    Inspired/Taken from
        https://www.protechtraining.com/blog/post/879#simple-observer
    """
    def __init__(self, events):
        # maps event names to subscribers
        # str -> dict
        self._events = {event: dict() for event in events}

    def get_subscribers(self, event):
        return self._events[event]

    def subscribe(self, event, subscriber, callback=None):
        if callback is None:
            callback = getattr(subscriber, 'update')
        self.get_subscribers(event)[subscriber] = callback

    def unsubscribe(self, event, subscriber):
        del self.get_subscribers(event)[subscriber]

    def dispatch(self, event):
        for _, callback in self.get_subscribers(event).items():
            callback(event, self)


class BayesianOptimization(Observable):
    def __init__(self, f, pbounds, random_state=None, verbose=2,
                 bounds_transformer=None, surrogate=None, cfg=None):
        self._using_gp = False
        self.cfg = cfg # entire config
        self.surrogate = surrogate
        self._regressor = surrogate.model      
        self.model_type = self.cfg['train_cfg']['model_type']
        if self.model_type == 'gp': self._using_gp = True
        self._random_state = ensure_rng(random_state)
        self._verbose = verbose
        self._bounds_transformer = bounds_transformer

        # Data structure containing the function to be optimized, the bounds of
        # its domain, and a record of the evaluations we have done so far
        self._space = TargetSpace(f, pbounds, random_state, cfg=self.cfg)
        
        if self._bounds_transformer:
            self._bounds_transformer.initialize(self._space)

        # queue
        self._queue = Queue()
        
        self.num_candidates_sampled = 0
        self.num_candidates_to_sample = self.cfg['train_cfg']['num_candidates_per_train']
        
        # losses for candidates
        self.MAElist = []
        self.RMSElist = []
        self.MSElist = []
        
        super(BayesianOptimization, self).__init__(events=DEFAULT_EVENTS)
        
    @property
    def space(self):
        return self._space

    @property
    def max(self):
        return self._space.max()

    @property
    def res(self):
        return self._space.res()

    def register(self, params, target):
        """Expect observation with known target"""
        # called in utils.load_logs
        self._space.register(params, target) # target space::register
        self.dispatch(Events.OPTIMIZATION_STEP)

    def probe(self, params, lazy=True):
        """Probe target of x"""
        if lazy:
            self._queue.add(params)
            print('adding:', params)
        else:
            target = self._space.probe(params) # registers suggested point
            self.dispatch(Events.OPTIMIZATION_STEP)
        
    # TODO:
    def suggest(self, utility_function):
        """Most promising point to probe next"""
        # optimizing acquisition
        if len(self._space) == 0:
            # return self._space.array_to_params(self._space.random_sample())
            return self._space.array_to_params(self._space.sample_single())

        # sklearn's GP throws a large number of warnings at times, but
        # we don't really need to see them here.
        # print('regressor to be fit with:\n\ttarget space params', self._space.params, 
        #       '\n\ttarget space targets', self._space.target)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            if (self.num_candidates_sampled % self.num_candidates_to_sample == 0):
                
                if self.num_candidates_sampled > 0:
                    # compute loss on candidates up to current iteration
                    # print(self._space._candidate_targets)
                    self.compute_loss_candidate()
                
                # print('training model...', end='')
                start = time.time()
                if self._using_gp:
                    self.surrogate.model.fit(self._space.params, self._space.target)
                else:
                    train_loader, test_loader = self.surrogate.preprocess(self._space.params, self._space.target)
                    self.surrogate.train(train_loader, test_loader)
                    
                    # clear from memory
                    del train_loader
                    del test_loader
                end = time.time()
                print(f'training model took {(end-start):.4f} sec')
            
        # print('candidates:', self.num_candidates_sampled)

        # Finding argmax of the acquisition function.
        suggestion = acq_max(
            acq_fcn=utility_function.utility,
            model=self._regressor,
            y_max=self._space.target.max(),
            bounds=self._space.bounds,
            random_state=self._random_state
        )
        
        self.num_candidates_sampled += 1
    
        return self._space.array_to_params(suggestion)

    def _prime_queue(self, init_points):
        """Make sure there's something in the queue at the very beginning."""
        if self._queue.empty and self._space.empty:
            init_points = max(init_points, 1)

        # for _ in range(init_points):
        #     self._queue.add(self._space.random_sample())
        
        import time
        start = time.time()
        points = self._space.sample_multiple(init_points)
        end = time.time()
        print('DATA SAMPLING: {:.3f}'.format(end-start))

        start = time.time()
        for _, point in enumerate(points):
            self._queue.add(point)
        end= time.time()
        print('ADDING TO QUEUE: {:.3f}'.format(end-start))
        
        # print('DATA POINTS: TOTAL', len(points))
        print('QUEUE: TOTAL', len(self._queue))
        # print('CURRENT PARAMS, TARGETS: TOTAL', len(self._space.params))
        
    def _prime_subscriptions(self):
        if not any([len(subs) for subs in self._events.values()]):
            _logger = _get_default_logger(self._verbose)
            self.subscribe(Events.OPTIMIZATION_START, _logger)
            self.subscribe(Events.OPTIMIZATION_STEP, _logger)
            self.subscribe(Events.OPTIMIZATION_END, _logger)

    def maximize(self,
                 init_points=5,
                 n_iter=25,
                 acq='ucb',
                 kappa=2.576,
                 kappa_decay=1,
                 kappa_decay_delay=0,
                 xi=0.0,
                 **gp_params):
        
        """Mazimize your function"""
        self._prime_subscriptions()
        self.dispatch(Events.OPTIMIZATION_START)
        self._prime_queue(init_points)
        
        # at this point, queue is filled with 100 initial samples
        self._space.visualize_objective_space(self._queue._queue)
        
        if self._using_gp:
            self.set_gp_params(**gp_params) # sets params in set_params in BaseEstimator in _gpr

        util = UtilityFunction(kind=acq,
                               kappa=kappa,
                               xi=xi,
                               kappa_decay=kappa_decay,
                               kappa_decay_delay=kappa_decay_delay)
        
        iteration = 0
        while not self._queue.empty or iteration < n_iter:
            try:
                x_probe = next(self._queue)
                # print('probing in queue:', x_probe)
            except StopIteration:
                util.update_params()
                x_probe = self.suggest(util)
                iteration += 1
                # print('new point suggested:', x_probe)
            
            # print('QUEUE: TOTAL', len(self._queue))
            # print('CURRENT PARAMS, TARGETS: TOTAL', len(self._space.params))
        
            # x_probe shape: [1, 2] for 2-dimensional point 
            self.probe(x_probe, lazy=False) # sample candidate

            if self._bounds_transformer:
                self.set_bounds(
                    self._bounds_transformer.transform(self._space))


        # ANP: 1000/200, 20x5 candidates, 100 initial
        
        # compute loss (MSE, MAE)
        # print(self._space._candidate_targets)
        # print(self._space._original_data_targets)
        
        # self.compute_loss_candidate()
        # self.compute_loss_original()
        
        self.dispatch(Events.OPTIMIZATION_END)

    # compute loss on candidates only
    def compute_loss_candidate(self):
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error([0]*len(self._space._candidate_targets), self._space._candidate_targets)
        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error([0]*len(self._space._candidate_targets), self._space._candidate_targets, squared=False)
        mse = mean_squared_error([0]*len(self._space._candidate_targets), self._space._candidate_targets)
        
        print(f'MAE:{mae:.4f}')
        print(f'RMSE:{rmse:.4f}')
        print(f'MSE:{mse:.4f}')
        
        self.MAElist.append(round(mae, 4))
        self.RMSElist.append(round(rmse, 4))
        self.MSElist.append(round(mse,4))
        return mae, rmse, mse
        
    # compute loss on original data (initial)
    def compute_loss_original(self):
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error([0]*len(self._space._original_data_targets), self._space._original_data_targets)
        from sklearn.metrics import mean_squared_error
        rmse = mean_squared_error([0]*len(self._space._original_data_targets), self._space._original_data_targets, squared=False)
        mse = mean_squared_error([0]*len(self._space._original_data_targets), self._space._original_data_targets)
        
        # print(f'MAE:{mae:.4f}')
        # print(f'RMSE:{rmse:.4f}')
        # print(f'MSE:{mse:.4f}')
        return mae, rmse, mse

    def set_bounds(self, new_bounds):
        """
        A method that allows changing the lower and upper searching bounds

        Parameters
        ----------
        new_bounds : dict
            A dictionary with the parameter name and its new bounds
        """
        self._space.set_bounds(new_bounds)

    def set_gp_params(self, **params):
        # self._gp.set_params(**params)
        self._regressor.set_params(**params)
