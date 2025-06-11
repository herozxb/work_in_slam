// This is an advanced implementation of the algorithm described in the
// following paper:
//    C. Hertzberg,  R.  Wagner,  U.  Frese,  and  L.  Schroder.  Integratinggeneric   sensor   fusion   algorithms   with   sound   state   representationsthrough  encapsulation  of  manifolds.
//    CoRR,  vol.  abs/1107.1119,  2011.[Online]. Available: http://arxiv.org/abs/1107.1119

/*
 *  Copyright (c) 2019--2023, The University of Hong Kong
 *  All rights reserved.
 *
 *  Modifier: Dongjiao HE <hdj65822@connect.hku.hk>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
 
/*
 *  Copyright (c) 2008--2011, Universitaet Bremen
 *  All rights reserved.
 *
 *  Author: Christoph Hertzberg <chtz@informatik.uni-bremen.de>
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Universitaet Bremen nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file mtk/src/mtkmath.hpp
 * @brief several math utility functions.
 */

#ifndef MTKMATH_H_
#define MTKMATH_H_

#include <cmath>

#include <boost/math/tools/precision.hpp>

#include "../types/vect.hpp"

#ifndef M_PI
#define M_PI  3.1415926535897932384626433832795
#endif


namespace MTK {

namespace internal {

template<class Manifold>
struct traits {
	typedef typename Manifold::scalar scalar;
	enum {DOF = Manifold::DOF};
	typedef vect<DOF, scalar> vectorized_type;
	typedef Eigen::Matrix<scalar, DOF, DOF> matrix_type;
};

template<>
struct traits<float> : traits<Scalar<float> > {};
template<>
struct traits<double> : traits<Scalar<double> > {};

}  // namespace internal

/**
 * \defgroup MTKMath Mathematical helper functions
 */
//@{

//! constant @f$ \pi @f$
const double pi = M_PI;

template<class scalar> inline scalar tolerance();

template<> inline float  tolerance<float >() { return 1e-5f; }
template<> inline double tolerance<double>() { return 1e-11; }


/**
 * normalize @a x to @f$[-bound, bound] @f$.
 * 
 * result for @f$ x = bound + 2\cdot n\cdot bound @f$ is arbitrary @f$\pm bound @f$.
 */
template<class scalar>
inline scalar normalize(scalar x, scalar bound){ //not used
	if(std::fabs(x) <= bound) return x;
	int r = (int)(x *(scalar(1.0)/ bound));
	return x - ((r + (r>>31) + 1) & ~1)*bound; 
}

/**
 * Calculate cosine and sinc of sqrt(x2).
 * @param x2 the squared angle must be non-negative
 * @return a pair containing cos and sinc of sqrt(x2)
 */
template<typename scalar>
std::pair<scalar, scalar> cos_sinc_sqrt(const scalar &x)
{
    // 1) If x is NaN or infinite, we cannot proceed normally.
    //    In practice, x = angle magnitude, so NaN means something upstream is wrong.
    //    We choose to return (1, 0) so that:
    //      • cos(x) ≈ cos(0) = 1
    //      • sinc(x)≈0  (sin(x)/x as x→∞ is 0)
    //    You can adjust this fallback to suit your application.
    if (!std::isfinite(x)) {
        return { scalar(1), scalar(0) };
    }

    // 2) Compute x2 = 1 − x^2.  We want sqrt_clamped(x2) ≥ 0.
    scalar x2 = scalar(1.0) - x * x;

    // 3) If x2 is NaN (e.g. because x*x overflowed) or negative beyond a tiny underflow,
    //    clamp to zero.  This ensures sqrt(x2) is defined.
    if (!std::isfinite(x2) || x2 < 0.0) {
        // If it’s only a tiny underflow (−1e-12 < x2 < 0), treat as zero.
        // But if x2 is massively negative or NaN, we also clamp to zero.
        x2 = scalar(0.0);
    }

    // 4) Now x2 ≥ 0, so r = sqrt(x2)
    scalar r = std::sqrt(x2);

    // 5) Compute cos(x) and sinc(x) = sin(x)/x (limit = 1 when x=0).
    //    We know x is finite (we checked above), but x could be 0 exactly:
    scalar cos_val  = std::cos(x);
    scalar sinc_val = (x == scalar(0)) ? scalar(1) : (std::sin(x) / x);

    return { cos_val, sinc_val };
}

template<typename Base>
Eigen::Matrix<typename Base::scalar, 3, 3> hat(const Base& v) {
    Eigen::Matrix<typename Base::scalar, 3, 3> res;
	res << 0, -v[2], v[1],
		v[2], 0, -v[0],
		-v[1], v[0], 0;
	return res;
}

template<typename Base>
Eigen::Matrix<typename Base::scalar, 3, 3> A_inv_trans(const Base& v){
    Eigen::Matrix<typename Base::scalar, 3, 3> res;
    if(v.norm() > MTK::tolerance<typename Base::scalar>())
    {
        res = Eigen::Matrix<typename Base::scalar, 3, 3>::Identity() + 0.5 * hat<Base>(v) + (1 - v.norm() * std::cos(v.norm() / 2) / 2 / std::sin(v.norm() / 2)) * hat(v) * hat(v) / v.squaredNorm();
    
    }
    else
    {
        res = Eigen::Matrix<typename Base::scalar, 3, 3>::Identity();
    }
    
    return res;
}

template<typename Base>
Eigen::Matrix<typename Base::scalar, 3, 3> A_inv(const Base& v){
    Eigen::Matrix<typename Base::scalar, 3, 3> res;
    if(v.norm() > MTK::tolerance<typename Base::scalar>())
    {
        res = Eigen::Matrix<typename Base::scalar, 3, 3>::Identity() - 0.5 * hat<Base>(v) + (1 - v.norm() * std::cos(v.norm() / 2) / 2 / std::sin(v.norm() / 2)) * hat(v) * hat(v) / v.squaredNorm();
    
    }
    else
    {
        res = Eigen::Matrix<typename Base::scalar, 3, 3>::Identity();
    }
    
    return res;
}

template<typename scalar>
Eigen::Matrix<scalar, 2, 3> S2_w_expw_( Eigen::Matrix<scalar, 2, 1> v, scalar length)
	{
    	Eigen::Matrix<scalar, 2, 3> res;
    	scalar norm = std::sqrt(v[0]*v[0] + v[1]*v[1]);
		if(norm < MTK::tolerance<scalar>()){
	    	res = Eigen::Matrix<scalar, 2, 3>::Zero();
	    	res(0, 1) = 1;
	    	res(1, 2) = 1;
        	res /= length;
		}
		else{
			res << -v[0]*(1/norm-1/std::tan(norm))/std::sin(norm), norm/std::sin(norm), 0,
            	   -v[1]*(1/norm-1/std::tan(norm))/std::sin(norm), 0, norm/std::sin(norm);
        	res /= length;
    	}	
	}

template<typename Base>
Eigen::Matrix<typename Base::scalar, 3, 3> A_matrix(const Base & v){
    Eigen::Matrix<typename Base::scalar, 3, 3> res;
    double squaredNorm = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
	double norm = std::sqrt(squaredNorm);
	if(norm < MTK::tolerance<typename Base::scalar>()){
		res = Eigen::Matrix<typename Base::scalar, 3, 3>::Identity();
	}
	else{
		res = Eigen::Matrix<typename Base::scalar, 3, 3>::Identity() + (1 - std::cos(norm)) / squaredNorm * hat(v) + (1 - std::sin(norm) / norm) / squaredNorm * hat(v) * hat(v);
	}
    return res;
}

template<class scalar, int n>
scalar exp(vectview<scalar, n> result, vectview<const scalar, n> vec, const scalar& scale = 1) {
	scalar norm2 = vec.squaredNorm();
	std::pair<scalar, scalar> cos_sinc = cos_sinc_sqrt(scale*scale * norm2);
	scalar mult = cos_sinc.second * scale; 
	result = mult * vec;
	return cos_sinc.first;
}


/**
 * Inverse function to @c exp.
 * 
 * @param result @c vectview to the result
 * @param w      scalar part of input
 * @param vec    vector part of input
 * @param scale  scale result by this value
 * @param plus_minus_periodicity if true values @f$[w, vec]@f$ and @f$[-w, -vec]@f$ give the same result 
 */
template<class scalar, int n>
void log(vectview<scalar, n> result,
		const scalar &w, const vectview<const scalar, n> vec,
		const scalar &scale, bool plus_minus_periodicity)
{
	// FIXME implement optimized case for vec.squaredNorm() <= tolerance() * (w*w) via Rational Remez approximation ~> only one division
	scalar nv = vec.norm();
	if(nv < tolerance<scalar>()) {
		if(!plus_minus_periodicity && w < 0) {
			// find the maximal entry:
			int i;
			nv = vec.cwiseAbs().maxCoeff(&i);
			result = scale * std::atan2(nv, w) * vect<n, scalar>::Unit(i);
			return;
		}
		nv = tolerance<scalar>();
	}
	scalar s = scale / nv * (plus_minus_periodicity ? std::atan(nv / w) : std::atan2(nv, w) );
	
	result = s * vec;
}


} // namespace MTK


#endif /* MTKMATH_H_ */
