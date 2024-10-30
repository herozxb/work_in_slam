///
/// @author Mohanad Youssef
/// @file KalmanFilter.h
///

#ifndef __KALMAN_FILTER_LIB_H__
#define __KALMAN_FILTER_LIB_H__

#include "types.h"
#include <iostream>
using namespace std;

namespace kf
{
    template<size_t DIM_X, size_t DIM_Z>
    class KalmanFilter
    {
    public:

        KalmanFilter()
        {

        }

        ~KalmanFilter()
        {

        }

        Vector<DIM_X> & vector_x() { return m_vector_x; }
        const Vector<DIM_X> & vecctor_x() const { return m_vector_x; }

        Matrix<DIM_X, DIM_X> & matrix_P() { return m_matrix_P; }
        const Matrix<DIM_X, DIM_X> & matrix_P() const { return m_matrix_P; }

        ///
        /// @brief predict state with a linear process model.
        /// @param matrix_F state transition matrix
        /// @param matrix_Q process noise covariance matrix
        ///
        void predict(const Matrix<DIM_X, DIM_X> & matrix_F, const Matrix<DIM_X, DIM_X> & matrix_Q )
        {
            m_vector_x = matrix_F * m_vector_x;
            m_matrix_P = matrix_F * m_matrix_P * matrix_F.transpose() + matrix_Q;
            
            //cout<<"vector x="<<m_vector_x<<endl;
            //cout<<"matrix P="<<m_matrix_P<<endl;
        }

        ///
        /// @brief correct state of with a linear measurement model.
        /// @param vector_z measurement vector
        /// @param matrix_R measurement noise covariance matrix
        /// @param matrix_H measurement transition matrix (measurement model)
        ///
        double correct(const Vector<DIM_Z> & vector_z, const Matrix<DIM_Z, DIM_Z> & matrix_R, const Matrix<DIM_Z, DIM_X> & matrix_H)
        {
            const Matrix<DIM_X, DIM_X> matrix_I{ Matrix<DIM_X, DIM_X>::Identity() }; // Identity matrix
            const Matrix<DIM_Z, DIM_Z> matrix_S_k{ matrix_H * m_matrix_P * matrix_H.transpose() + matrix_R }; // Innovation covariance
            const Matrix<DIM_X, DIM_Z> matrix_K_k{ m_matrix_P * matrix_H.transpose() * matrix_S_k.inverse() }; // Kalman Gain

            m_vector_x = m_vector_x + matrix_K_k * (vector_z - (matrix_H * m_vector_x));
            m_matrix_P = (matrix_I - matrix_K_k * matrix_H) * m_matrix_P;
            
            //cout<<"vector x correct ="<<m_vector_x<<endl;
            //cout<<"vector P correct ="<<m_matrix_P<<endl;
            return m_vector_x[0];
        }

        ///
        /// @brief predict state with a linear process model.
        /// @param prediction_model prediction model function callback
        /// @param matrix_jacob_F state jacobian matrix
        /// @param matrix_Q process noise covariance matrix
        ///
        template<typename prediction_model_callback>
        void predictEkf(prediction_model_callback prediction_model, const Matrix<DIM_X, DIM_X> & matrix_jacob_F, const Matrix<DIM_X, DIM_X> & matrix_Q)
        {
            m_vector_x = prediction_model(m_vector_x);
            m_matrix_P = matrix_jacob_F * m_matrix_P * matrix_jacob_F.transpose() + matrix_Q;
        }

        ///
        /// @brief correct state of with a linear measurement model.
        /// @param measurement_model measurement model function callback
        /// @param vector_Z measurement vector
        /// @param matrix_R measurement noise covariance matrix
        /// @param matrix_jacob_H measurement jacobian matrix
        ///
        template<typename measurement_model_callback>
        void correctEkf(measurement_model_callback measurement_model,const Vector<DIM_Z> & vector_Z, const Matrix<DIM_Z, DIM_Z> & matrix_R, const Matrix<DIM_Z, DIM_X> & matrix_jacob_H)
        {
            const Matrix<DIM_X, DIM_X> matrix_I{ Matrix<DIM_X, DIM_X>::Identity() }; // Identity matrix
            const Matrix<DIM_Z, DIM_Z> matrix_S_k{ matrix_jacob_H * m_matrix_P * matrix_jacob_H.transpose() + matrix_R }; // Innovation covariance
            const Matrix<DIM_X, DIM_Z> matrix_K_k{ m_matrix_P * matrix_jacob_H.transpose() * matrix_S_k.inverse() }; // Kalman Gain

            m_vector_x = m_vector_x + matrix_K_k * (vector_Z - measurement_model(m_vector_x));
            m_matrix_P = (matrix_I - matrix_K_k * matrix_jacob_H) * m_matrix_P;
        }

    private:
        Vector<DIM_X> m_vector_x{ Vector<DIM_X>::Zero() }; /// @brief estimated state vector
        Matrix<DIM_X, DIM_X> m_matrix_P{ Matrix<DIM_X, DIM_X>::Zero() }; /// @brief state covariance matrix
    };
}

#endif // __KALMAN_FILTER_LIB_H__
