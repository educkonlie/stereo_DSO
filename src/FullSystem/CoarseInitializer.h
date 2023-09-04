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
#include "IOWrapper/Output3DWrapper.h"
#include "util/settings.h"
#include "vector"
#include <math.h>




namespace dso
{
struct CalibHessian;
struct FrameHessian;


struct Pnt
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// index in jacobian. never changes (actually, there is no reason why).
	float u,v;
	float idepth;
	Vec2f energy;		// (UenergyPhotometric, energyRegularizer)
	Vec2f energy_new;
	float iR;
	float my_type;
};

class CoarseInitializer {
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	CoarseInitializer(int w, int h);
	~CoarseInitializer();

	void setFirstStereo(CalibHessian* HCalib, FrameHessian* newFrameHessian);

	int frameID;
//	bool fixAffine;
	bool printDebug;

	Pnt* points[PYR_LEVELS];
	int numPoints[PYR_LEVELS];
	AffLight thisToNext_aff;
	SE3 thisToNext;

	FrameHessian* firstFrame;
private:
	Mat33 K[PYR_LEVELS];
	Mat33 Ki[PYR_LEVELS];
	double fx[PYR_LEVELS];
	double fy[PYR_LEVELS];
	double cx[PYR_LEVELS];
	double cy[PYR_LEVELS];
	int w[PYR_LEVELS];
	int h[PYR_LEVELS];
	void makeK(CalibHessian* HCalib);
	float* idepth[PYR_LEVELS];

	Eigen::DiagonalMatrix<float, 8> wM;

	// temporary buffers for H and b.
	Vec10f* JbBuffer;			// 0-7: sum(dd * dp). 8: sum(res*dd). 9: 1/(1+sum(dd*dd))=inverse hessian entry.
	Vec10f* JbBuffer_new;

//	Accumulator9 acc9;
//	Accumulator9 acc9SC;
};

}


