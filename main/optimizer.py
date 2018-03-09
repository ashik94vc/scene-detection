import numpy as np

def backtracking_line_search( func, x, direction, alpha=0.4, beta=0.9, maximum_iterations=65536 ):
    """
    Backtracking linesearch
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    x:                  the current iterate
    direction:          the direction along which to perform the linesearch
    alpha:              the alpha parameter to backtracking linesearch
    beta:               the beta parameter to backtracking linesearch
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    """

    if alpha <= 0:
        raise ValueError("Alpha must be positive")
    if alpha >= 0.5:
        raise ValueError("Alpha must be less than 0.5")
    if beta <= 0:
        raise ValueError("Beta must be positive")
    if beta >= 1:
        raise ValueError("Beta must be less than 1")

    x = np.matrix( x )
    value_0, gradient_0 = func(x, 1)
    value_0 = np.double( value_0 )
    gradient_0 = np.matrix( gradient_0 )

    t = 1
    iterations = 0
    while True:

        # if (TODO: TERMINATION CRITERION): break
        if func(x+t*direction,0) <= value_0 + alpha*t*gradient_0.T*direction:
            break
        t = beta * t
        # t = TODO: BACKTRACKING LINE SEARCH

        iterations += 1
        if iterations >= maximum_iterations:
            break

    return t

def bisection( one_d_fun, MIN, MAX, eps=1e-5, maximum_iterations=65536 ):

  # counting the number of iterations
  iterations = 0

  if eps <= 0:
      raise ValueError("Epsilon must be positive")

  while True:

    MID = ( MAX + MIN ) / 2

    # Oracle access to the function value and derivative
    value, derivative = one_d_fun( MID, 1 )

    # if (TODO: TERMINATION CRITERION): break
    if abs(derivative)*(MAX - MID) <= eps:
        break
    if derivative < 0:
        MIN = MID
    # if derivative... (TODO: LINE SEARCH)
    elif derivative > 0:
        MAX = MID
    else:
        break
    iterations += 1
    if iterations>=maximum_iterations:
        break

  return MID

def newton( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=bisection, *linesearch_args  ):
    """
    Newton's Method
    func:               the function to optimize It is called as "value, gradient, hessian = func( x, 2 )
    initial_x:          the starting point
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """

    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix( initial_x )

    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0

    # newton's method updates
    while True:

        value, gradient, hessian = func( x , 2 )
        value = np.double( value )
        gradient = np.matrix( gradient )
        hessian = np.matrix( hessian )

        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # direction = (TODO)
        direction = -hessian.I * gradient

        # if (TODO: TERMINATION CRITERION): break
        if gradient.T*hessian.I*gradient < eps:
            break

        t = backtracking_line_search( func, x, direction )

        # x = (TODO: UPDATE x)
        x = x + t*direction

        iterations += 1
        if iterations >= maximum_iterations:
            break

    return (x, values, runtimes, xs)

def gradient_descent( func, initial_x, eps=1e-5, maximum_iterations=65536, linesearch=exact_line_search, *linesearch_args ):
    """
    Gradient Descent
    func:               the function to optimize It is called as "value, gradient = func( x, 1 )
    initial_x:          the starting point, should be a float
    eps:                the maximum allowed error in the resulting stepsize t
    maximum_iterations: the maximum allowed number of iterations
    linesearch:         the linesearch routine
    *linesearch_args:   the extra arguments of linesearch routine
    """

    if eps <= 0:
        raise ValueError("Epsilon must be positive")
    x = np.matrix(initial_x)

    # initialization
    values = []
    runtimes = []
    xs = []
    start_time = time.time()
    iterations = 0

    # gradient updates
    while True:

        value, gradient = func( x , 1 )
        value = np.double( value )
        gradient = np.matrix( gradient )

        # updating the logs
        values.append( value )
        runtimes.append( time.time() - start_time )
        xs.append( x.copy() )

        # direction= (TODO)
        direction = -gradient

        # if (TODO: TERMINATION CRITERION): break
        if np.linalg.norm(gradient)**2 < eps:
            break

        t = linesearch( func, x, direction, *linesearch_args )

        # x= (TODO: UPDATE x)
        x = x + t*direction

        iterations += 1
        if iterations >= maximum_iterations:
            break

    return (x, values, runtimes, xs)
