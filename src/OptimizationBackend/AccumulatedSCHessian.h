/**
* This file is part of DSO.
* 
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"
#include "util/IndexThreadReduce.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "vector"
#include <math.h>

namespace dso
{

class EFPoint;
class EnergyFunctional;

class AccumulatedSCHessianSSE {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	inline AccumulatedSCHessianSSE()
	{
		for(int i=0;i<NUM_THREADS;i++) {
			accE[i]=0;
			accEB[i]=0;
			accD[i]=0;
			nframes[i]=0;
		}
	};
	inline ~AccumulatedSCHessianSSE()
    {
		for(int i=0;i<NUM_THREADS;i++) {
			if(accE[i] != 0) delete[] accE[i];
			if(accEB[i] != 0) delete[] accEB[i];
			if(accD[i] != 0) delete[] accD[i];
		}
	};

	inline void setZero(int n, int min=0, int max=1, Vec10* stats=0, int tid=0) {
		if(n != nframes[tid]) {
            std::cout << "n: " << n << " nframes: " << nframes[tid] << std::endl;
			if(accE[tid] != 0) delete[] accE[tid];
			if(accEB[tid] != 0) delete[] accEB[tid];
			if(accD[tid] != 0) delete[] accD[tid];
			accE[tid] = new AccumulatorXX<8,CPARS>[n*n];
			accEB[tid] = new AccumulatorX<8>[n*n];
            //! 这里为什么是n cube
			accD[tid] = new AccumulatorXX<8,8>[n*n*n];
//			accD[tid] = new AccumulatorXX<8,8>[n*n];
		}
		accbc[tid].initialize();
		accHcc[tid].initialize();

		for(int i=0;i<n*n;i++) {
			accE[tid][i].initialize();
			accEB[tid][i].initialize();

            //! 一个accE对应了n个accD
			for(int j=0;j<n;j++)
                //! accD[n*n][n]
				accD[tid][i*n+j].initialize();
		}
		nframes[tid]=n;
	}
	void stitchDouble(MatXX &H_sc, VecX &b_sc, EnergyFunctional const * const EF, int tid=0);
	void addPoint(EFPoint* p, int tid=0);

	AccumulatorXX<8,CPARS>* accE[NUM_THREADS];
	AccumulatorX<8>* accEB[NUM_THREADS];
	AccumulatorXX<8,8>* accD[NUM_THREADS];
	AccumulatorXX<CPARS,CPARS> accHcc[NUM_THREADS];
	AccumulatorX<CPARS> accbc[NUM_THREADS];
	int nframes[NUM_THREADS];

private:

};

}

