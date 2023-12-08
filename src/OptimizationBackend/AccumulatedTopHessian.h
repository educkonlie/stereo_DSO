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
#include "OptimizationBackend/MatrixAccumulators.h"
#include "vector"
#include <math.h>
#include "util/IndexThreadReduce.h"


namespace dso
{

class EFPoint;
class EnergyFunctional;

class AccumulatedTopHessianSSE {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	inline AccumulatedTopHessianSSE()
	{
		for(int tid=0;tid < NUM_THREADS; tid++) {
			nres[tid]=0;
			acc[tid]=0;
			nframes[tid]=0;
		}
        myH = NULL;
        myH_rootba = NULL;
	};
	inline ~AccumulatedTopHessianSSE()
	{
		for(int tid=0;tid < NUM_THREADS; tid++) {
			if(acc[tid] != 0) delete[] acc[tid];
		}
        if (myH != 0) delete[] myH;
        if (myH_rootba != 0) delete[] myH_rootba;
	};

	inline void setZero(int nFrames, int min=0, int max=1, Vec10* stats=0, int tid=0)
	{
//        my_big_H.setZero();
//        my_big_b.setZero();
//        my_big_H = new MatXX::Zero(nframes[tid]*8+CPARS, nframes[tid]*8+CPARS);
//        my_big_b = new VecX::Zero(nframes[tid]*8+CPARS);
		if(nFrames != nframes[tid]) {
			if(acc[tid] != 0) delete[] acc[tid];
            acc[tid] = new AccumulatorApprox[nFrames*nFrames];
            if(myH != 0) delete[] myH;
            myH = new MatPCPCf[nFrames * nFrames];
            if(myH_rootba != 0) delete[] myH_rootba;
            myH_rootba = new MatPCPCf[nFrames * nFrames];
		}

		for(int i=0;i<nFrames*nFrames;i++) {
            acc[tid][i].initialize();
            myH[i].setZero();
            myH_rootba[i].setZero();
        }

		nframes[tid]=nFrames;
		nres[tid]=0;

	}
	void stitchDouble(MatXX &H, VecX &b, EnergyFunctional const * const EF,
                      bool usePrior, bool useDelta, int tid=0);


#if 1
	template<int mode> void addPoint(EFPoint* p, EnergyFunctional const * const ef, int tid=0);
#endif

	int nframes[NUM_THREADS];
	EIGEN_ALIGN16 AccumulatorApprox* acc[NUM_THREADS];
	int nres[NUM_THREADS];

    MatPCPCf *myH;
    MatPCPCf *myH_rootba;

private:

};
}

