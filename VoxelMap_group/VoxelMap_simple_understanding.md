# 1 
A is [ d(d) / d(R), d(d) / d(t) ]
d = n^T(Rxp+t)
d(d) / d(R) = d(d) / d(p_world) * d(p_world) / d(R) 
d(p_world) / d(R) = p_crose x R 

# 2	change the R is just change the weight of the least square
t_new = Rq + t
dt_new/dR = -R*q^ 	dt_new/dt = I 
this is d(vector) / d(vector) = matrix
	d(scalar) / d(vector) = vector
