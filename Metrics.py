class Metrics():
    """
    This class re-implements known metrics. 
    However, in this way it is easier to perform the 
    experiments at hand of predicting and evaluating diagnoses.    
    
    Available functions:
    - recall
    - precision
    
    Inputs:
    - golden: list of golden labels
    - retrieved: list of retrieved labels
    - k (optional): compute metric@k
    
    Example
    -------
    a = [1,2,3]
    b = [0,1]
    
    metrics.recall(a,b)
    >>> 0.3333
    metrics.precision(a,b,k=1)
    >>> 0.0
    """
        
    tp = lambda golden,retrieved: len([e for e in retrieved if e in golden])
    fn = lambda golden,retrieved: len([e for e in golden if e not in retrieved])
    fp = lambda golden,retrieved: len([e for e in retrieved if e not in golden])
    cut = lambda retrieved,k: retrieved if k is None else retrieved[:k]
    
    @classmethod
    def compute_all(cls,golden,retrieved):
        return {'tp':cls.tp(golden,retrieved),
                'fn':cls.fn(golden,retrieved),
                'fp':cls.fp(golden,retrieved)
               }
    @classmethod
    def recall(cls,golden,retrieved,k=None):
        if len(retrieved) == 0:
            return 0
        cls.check_inconsistency(golden,retrieved)
        retrieved = cls.cut(retrieved,k)
        metrics = cls.compute_all(golden,retrieved)
        
        return metrics['tp'] / (metrics['tp'] + metrics['fn'])
    @classmethod
    def precision(cls,golden,retrieved,k=None):
        if len(retrieved) == 0:
            return 0
        cls.check_inconsistency(golden,retrieved)
        retrieved = cls.cut(retrieved,k)
        metrics = cls.compute_all(golden,retrieved)

        return metrics['tp'] / (metrics['tp'] + metrics['fp'])
    
    @classmethod
    def get_metrics(cls):
        return ['precision','recall']
    
    @classmethod
    def compute_metric(cls,metric: str,golden,retrieved,k):
        if metric not in cls.get_metrics():
            raise ValueError(f'Expecting one of the followin metrics {cls.get_metrics}. Got {metric}')
        if metric == 'recall':
            return cls.recall(golden,retrieved,k)
        if metric == 'precision':
            return cls.precision(golden,retrieved,k)
    
    @classmethod
    def check_inconsistency(cls,golden,retrieved):
        """
        neither lists should have repeats inside
        """
        assert len(golden) == len(set(golden)), 'Must not have repeats'
        assert len(retrieved) == len(set(retrieved)), 'Must not have repeats'

