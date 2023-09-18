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


#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{

    //! 将p化为Hsc
void AccumulatedSCHessianSSE::addPoint(EFPoint* p, int tid)
{
	int ngoodres = 0;
    // 统计该点所有的活跃残差
	for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
    // 如果没有活跃残差
	if(ngoodres==0) {
		p->HdiF=0;
		p->bdSumF=0;
		p->data->idepth_hessian=0;
		p->data->maxRelBaseline=0;
		return;
	}

    // 逆深度的:  H = A + prior
    //! 仅在initializeFromInitializer的时候才会有priorF，大约是2500或9000，所以初始逆深度的方差不会大于
    //! 这个数的倒数。后面的priorF都是0了。
//    if (p->priorF > 0.000001)
//        printf("p->priorF: [%f], p->Hdd_accAF: [%f]\n", p->priorF, p->Hdd_accAF);
    float H = p->Hdd_accAF + p->priorF;
	if(H < 1e-10) H = 1e-10;

    // 逆深度的信息矩阵，因为逆深度是一维，所以是一个float，逆深度的协方差即1.0 / H
	p->data->idepth_hessian=H;

    // 原来HdiF即是协方差
	p->HdiF = 1.0 / H;
	p->bdSumF = p->bd_accAF;

	VecCf Hcd = p->Hcd_accAF;
    //! L * w * R.t() => i X j => A
    //! acc<i><j> A += L * w * R.t()
    //! w = H_inv
	accHcc[tid].update(Hcd,Hcd,p->HdiF);
	accbc[tid].update(Hcd, p->bdSumF * p->HdiF);

	assert(std::isfinite((float)(p->HdiF)));

	int nFrames2 = nframes[tid]*nframes[tid];
	for(EFResidual* r1 : p->residualsAll) {
		if(!r1->isActive()) continue;
		int r1ht = r1->hostIDX + r1->targetIDX*nframes[tid];

		for(EFResidual* r2 : p->residualsAll) {
			if(!r2->isActive()) continue;

            //! accD[target2][target1][host]
			accD[tid][r1ht+r2->targetIDX*nFrames2].update(r1->JpJdF, r2->JpJdF, p->HdiF);
		}

        //! accE[t][h], accEB[t][h]
		accE[tid][r1ht].update(r1->JpJdF, Hcd, p->HdiF);
		accEB[tid][r1ht].update(r1->JpJdF,p->HdiF*p->bdSumF);
	}
}


void AccumulatedSCHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const * const EF, int tid)
{
	int nf = nframes[0];
	int nframes2 = nf*nf;

	H = MatXX::Zero(nf*8+CPARS, nf*8+CPARS);
	b = VecX::Zero(nf*8+CPARS);

	for(int i=0;i<nf;i++)
		for(int j=0;j<nf;j++) {
			int iIdx = CPARS+i*8;
			int jIdx = CPARS+j*8;
			int ijIdx = i+nf*j;

			accE[tid][ijIdx].finish();
			accEB[tid][ijIdx].finish();

			Mat8C accEM = accE[tid][ijIdx].A1m.cast<double>();
			Vec8 accEBV = accEB[tid][ijIdx].A1m.cast<double>();

			H.block<8,CPARS>(iIdx,0) += EF->adHost[ijIdx] * accEM;
			H.block<8,CPARS>(jIdx,0) += EF->adTarget[ijIdx] * accEM;

			b.segment<8>(iIdx) += EF->adHost[ijIdx] * accEBV;
			b.segment<8>(jIdx) += EF->adTarget[ijIdx] * accEBV;

			for(int k=0;k<nf;k++) {
				int kIdx = CPARS+k*8;
				int ijkIdx = ijIdx + k*nframes2;
				int ikIdx = i+nf*k;

				accD[tid][ijkIdx].finish();
				if(accD[tid][ijkIdx].num == 0) continue;
				Mat88 accDM = accD[tid][ijkIdx].A1m.cast<double>();

                //! 对于相对位姿(t, h)的偏导转为对于绝对位姿t, h的偏导，并且因为是在制作H，会有四块
				H.block<8,8>(iIdx, iIdx) +=
                        EF->adHost[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
				H.block<8,8>(jIdx, kIdx) +=
                        EF->adTarget[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
				H.block<8,8>(jIdx, iIdx) +=
                        EF->adTarget[ijIdx] * accDM * EF->adHost[ikIdx].transpose();
				H.block<8,8>(iIdx, kIdx) +=
                        EF->adHost[ijIdx] * accDM * EF->adTarget[ikIdx].transpose();
			}
		}

	accHcc[tid].finish();
	accbc[tid].finish();
    //! H左上角<CPARS, CPARS>块和对应的b最上面<CPARS>行
	H.topLeftCorner<CPARS,CPARS>() = accHcc[tid].A1m.cast<double>();
	b.head<CPARS>() = accbc[tid].A1m.cast<double>();

	// ----- new: copy transposed parts for calibration only.
	for(int h=0;h<nf;h++) {
		int hIdx = CPARS+h*8;
        //! 右上角的横长条 = 左下角的竖长条的转置
        //! 逐格填充，每格<CPARS, 8>或<8, CPARS>
		H.block<CPARS,8>(0,hIdx).noalias() =
                H.block<8,CPARS>(hIdx,0).transpose();
	}
}

}
