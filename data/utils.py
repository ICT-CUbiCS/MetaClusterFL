import numpy as np

def samples_per_class_dirichlet(n_classes,c_clients,alpha,n=None,debug=False):
    """
    
    Returns the number of samples the nth client must sample from each class
    according to the Dirichlet distribution with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    Parameters
    ----------
    n_classes : int
        The number of classes/categories.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the *number* of elements of y that the i-th client must sample.

    debug : boolean, optional
        Whether to perform extra checks (which can be slow) 
    
    Returns
    -------
    A numpy array of shape(c,k) matrix where A[i,j] denotes
    the amount of instances of class j the client i must draw.

    """
    assert alpha>0
    
    # Sample from Dirichelts Dist.
    # proportions[i][j] indicates the proportion of class j that client i must draw
    proportions=np.random.dirichlet(alpha*np.ones(n_classes),c_clients)
    
    # Multiply by n and cast as int
    for client,client_i_n in enumerate(n):
        proportions[client,:]*=client_i_n

    out=proportions.astype('int')
    
    # Correct errors caused by truncation
    missing_by_client=n-out.sum(axis=1)
    assert all(missing_by_client>=0),'Possible overflow'
    for client,n_missed_by_client in enumerate(missing_by_client):
        where_to_add=np.random.choice(n_classes,size=n_missed_by_client)
        np.add.at(out[client,:],where_to_add,1)
    
    if debug:
        # Total of output must equal total of input
        assert out.sum()==sum(n)
    
    return out

def dirichlet_partition(y,c_clients,alpha,n=None,debug=False):
    """
    Randomly partitions an array of labels y into a # c_clients of clients
    according to Dirichelet sampling with concentration parameter alpha.

    Unless the proportion of samples the i-th client must draw is specified in n[i], 
    n is set such that the number of samples are distributed uniformly
    (equivalent to setting n[i] = y.size / c_clients).

    To guarantee that every 0 < alpha <= 1 can be met the total number of samples that can
    be sampled is set to the number of labels with the minimum frequency in y
    ('n_max_all_alphas' in the code). This may be too conservative but it's the
    easiest way to guarantee that samples_per_class_dirichlet doesn't over-assign a class
    (returning a matrix with a sum of column 0 that is greater than the # of instances of
    class 0, for example).

    alpha --> 0 implies very uneven sampling while alpha --> inf approaches uniform sampling.  

    Parameters
    ----------
    y : numpy array
        The numpy array of labels to be partitioned, assumed to be of integers 0 to
        # of classes -1.

    c_clients : int
        The number of clients or number of segments to partition y among.

    alpha : float
        Dirichlet sampling's concentration parameter (0 < alpha <= 1)

    n : numpy array or None, optional
        n[i] specifies the proportion of elements of y that the i-th client must sample.
        Therefore 

    debug : boolean, optional
        Whether to perform extra checks (which can be slow) 
        
    Returns
    -------
    The partition as a dictionary: client id (int) -> array of indices (np.array).

    """
    assert isinstance(c_clients,int) and c_clients>0
    assert alpha>0

    # The number of classes if y is assumed to be pandas' categorical codes.
    classes,counts_y=np.unique(y,return_counts=True)
    n_classes=len(counts_y)

    # Max n such that all alphas can be guaranteed
    # The worst case that can occur is if one client is assigned
    n_max_all_alphas=counts_y.min()

    # If n is None we distribute equally
    if n is None:
        n=[n_max_all_alphas//c_clients]*c_clients

    # else:
    #     assert sum(n)==1 and all([0<=i<=1 for i in n])
    #     n=[int(n_max_all_alphas*n_prop) for n_prop in n]    
    
    # Given how many examples each client must sample from each class
    how_many=samples_per_class_dirichlet(
        n_classes=n_classes,
        c_clients=c_clients,
        alpha=alpha,
        n=n,
        debug=debug
    )

    # Assert we have enough instances from each class
    # assert all(counts_y-how_many.sum(axis=0)>=0),'Not enough instances from each class to compy with how_many'

    # Find indices for each class and shuffle them
    wheres={}
    for class_i in classes:
        w=np.where(y==class_i)[0]
        np.random.shuffle(w)
        wheres[class_i]=list(w)

    # Client -> list of indices
    partition={c:[] for c in range(c_clients)}

    # For every class
    for i,class_i in enumerate(classes):
        # We distribute the corresponding indices to the clients
        prev=0
        for client,ni in enumerate(how_many[:,i]):
            # partition[client].extend(wheres[class_i][prev:prev+ni])
            partition[client].extend(list(np.random.choice(wheres[class_i], ni, replace=False)))
            added=len(wheres[class_i][prev:prev+ni])

            if debug:
                assert added==ni,f'added: {added} ni:{ni}'

            prev+=ni 

    return partition
        