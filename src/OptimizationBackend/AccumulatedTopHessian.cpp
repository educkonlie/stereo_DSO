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


#include "OptimizationBackend/AccumulatedTopHessian.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include <iostream>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{
    //   Top的addPoint和Bot的addPoint

#if 1
#ifndef FRAMES
#define FRAMES (nframes[0])
//#define FRAMES (8)
#endif
template<int mode>
void AccumulatedTopHessianSSE::addPoint(MatXX &H, VecX &b,
                                        MatXX &Hsc_rootba, VecX &bsc_rootba,
                                        EFPoint* p, EnergyFunctional const * const ef, int tid)	// 0 = active, 1 = linearized, 2=marginalize
{
    MatXXf Jr1;
    Jr1 = MatXXf::Zero(8 * FRAMES, CPARS + FRAMES * 8);
    VecXf Jr2;
    Jr2 = VecXf::Zero(8 * FRAMES);
    VecXf Jl;
    Jl  = VecXf::Zero(8 * FRAMES);
//    = resApprox;
	assert(mode==0 || mode==2);

	VecCf dc = ef->cDeltaF;

	float bd_acc=0;
	float Hdd_acc=0;
	VecCf  Hcd_acc = VecCf::Zero();

    //! 对该点所有的残差计算相应的矩阵块。Top里的是该残差对应的C, xi部分的偏导，Sch里的是该残差对应的舒尔补
    int old_hostIdx = -1;
    int k = 0;
	for(EFResidual* r : p->residualsAll) {
		if(mode==0) {
            assert(!r->isLinearized);
			if(r->isLinearized || !r->isActive()) continue;
		}
		if(mode==2) {
            // 这里还是有已经固定线性化的残差的
//            assert(!r->isLinearized);
			if(!r->isActive()) continue;
			assert(r->isLinearized);
		}

        //! for test purpose. 同一个point下的residual都是同一个host。
        if (old_hostIdx != -1) {
            assert(r->hostIDX == old_hostIdx);
        } else {
            old_hostIdx = r->hostIDX;
        }

		RawResidualJacobian* rJ = r->J;
		int htIDX = r->hostIDX + r->targetIDX*nframes[tid];
		Mat18f dp = ef->adHTdeltaF[htIDX];

		VecNRf resApprox;
		if(mode==0)
			resApprox = rJ->resF;
		if(mode==2)
			resApprox = r->res_toZeroF;

		// need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
		Vec2f JI_r(0,0);
		Vec2f Jab_r(0,0);
		float rr=0;
		for(int i=0;i<patternNum;i++) {
			JI_r[0] += resApprox[i] *rJ->JIdx[0][i];
			JI_r[1] += resApprox[i] *rJ->JIdx[1][i];
			Jab_r[0] += resApprox[i] *rJ->JabF[0][i];
			Jab_r[1] += resApprox[i] *rJ->JabF[1][i];
			rr += resApprox[i]*resApprox[i];
		}
        //! 打印host, target, C, xi, ab
//        std::cout << "C:\n" << rJ->JIdx[0] * rJ->Jpdc[0].transpose() +
//                rJ->JIdx[1] * rJ->Jpdc[1].transpose() << std::endl;
//        std::cout << "xi: \n" << rJ->JIdx[0] * rJ->Jpdxi[0].transpose() +
//                rJ->JIdx[1] * rJ->Jpdxi[1].transpose() << std::endl;
//        std::cout << "a: \n" << rJ->JabF[0].transpose() << std::endl;
//        std::cout << "b: \n" << rJ->JabF[1].transpose() << std::endl;
//        std::cout << "res: \n" << resApprox.transpose() << std::endl;
//        std::cout << std::endl;
        //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导


#if 1
#if 0
//        std::cout << "point has " << p->residualsAll.size() << " residuals" << std::endl;
        Eigen::Matrix<float, 8, 13> Jr = Eigen::Matrix<float, 8, 13>::Zero();
        Jr.block<8, 4>(0, 0) = rJ->JIdx[0] * rJ->Jpdc[0].transpose()
                               + rJ->JIdx[1] * rJ->Jpdc[1].transpose();
        Jr.block<8, 6>(0, 4) = rJ->JIdx[0] * rJ->Jpdxi[0].transpose()
                               + rJ->JIdx[1] * rJ->Jpdxi[1].transpose();
        Jr.block<8, 1>(0, 10) = rJ->JabF[0];
        Jr.block<8, 1>(0, 11) = rJ->JabF[1];
        Jr.block<8, 1>(0, 12) = resApprox;
        myH[htIDX] += Jr.transpose() * Jr;
#endif

#if 1
        Eigen::Matrix<float, 8, 8> J_th = Eigen::Matrix<float, 8, 8>::Zero();
        J_th.block<8, 6>(0, 0) = rJ->JIdx[0] * rJ->Jpdxi[0].transpose()
                                 + rJ->JIdx[1] * rJ->Jpdxi[1].transpose();
        J_th.block<8, 1>(0, 6) = rJ->JabF[0];
        J_th.block<8, 1>(0, 7) = rJ->JabF[1];

//        Jr1 = MatXXf::Zero(8, CPARS + nframes[0] * 8);
        Jr1.block<8, 4>(k * 8, 0)
                = rJ->JIdx[0] * rJ->Jpdc[0].transpose() + rJ->JIdx[1] * rJ->Jpdc[1].transpose();

        Jr1.block<8, 8>(k * 8, r->hostIDX * 8 + 4)
                = J_th * ef->adHostF[htIDX].transpose();

        Jr1.block<8, 8>(k * 8, r->targetIDX * 8 + 4)
                = J_th * ef->adTargetF[htIDX].transpose();


//        Eigen::Matrix<float, 8, 1> Jr2 = resApprox;
        Jr2.block<8, 1>(k * 8, 0) = resApprox;

//        Eigen::Matrix<float, 8 * , 1> Jl = Eigen::Matrix<float, 8, 1>::Zero();
        Jl.block<8, 1>(k * 8, 0) = rJ->JIdx[0] * rJ->Jpdd(0, 0)
                               + rJ->JIdx[1] * rJ->Jpdd(1, 0);
        //! 可以把窗口固定为8, 然后做成固定大小的矩阵，如果帧数小于8， 就用0填充

//        H += (Jr1.transpose() * Jr1).cast<double>();
//        b += (Jr1.transpose() * Jr2).cast<double>();
#endif

#endif


#if 0
		acc[tid][htIDX].update(
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				rJ->JIdx2(0,0),rJ->JIdx2(0,1),rJ->JIdx2(1,1));


		acc[tid][htIDX].updateBotRight(
				rJ->Jab2(0,0), rJ->Jab2(0,1), Jab_r[0],
				rJ->Jab2(1,1), Jab_r[1],rr);

		acc[tid][htIDX].updateTopRight(
				rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
				rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
				rJ->JabJIdx(0,0), rJ->JabJIdx(0,1),
				rJ->JabJIdx(1,0), rJ->JabJIdx(1,1),
				JI_r[0], JI_r[1]);
#endif

//        acc[tid][htIDX].finish();
//        std::cout << "accH:\n" << acc[tid][htIDX].H << std::endl;

		Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
		bd_acc +=  JI_r[0]*rJ->Jpdd[0] + JI_r[1]*rJ->Jpdd[1];
		Hdd_acc += Ji2_Jpdd.dot(rJ->Jpdd);
		Hcd_acc += rJ->Jpdc[0]*Ji2_Jpdd[0] + rJ->Jpdc[1]*Ji2_Jpdd[1];

		nres[tid]++;
        k++;
        assert(k <= FRAMES);
	}
//    std::cout << "Jr1:\n" << Jr1 << std::endl;
//    std::cout << "Jr2:\n" << Jr2.transpose() << std::endl;
#ifdef ROOTBA
    MatXXf Q;
    Q = MatXXf::Zero(8 * FRAMES,  8 * FRAMES);

    VecXf Jl1 = VecXf::Zero(8 * FRAMES);
    //! QR_decomp还只支持8行1列，要修改
//    this->QR_decomp(Jl, Q, Jl1);
//    VecXf R;
//    R = VecXf::Zero(8 * FRAMES);
    Eigen::HouseholderQR<VecXf > qr;
    qr.compute(Jl);
    Q = qr.householderQ();
    Jl1 = qr.matrixQR().triangularView<Eigen::Upper>();

//    std::cout << "Jl " << Jl.transpose() << std::endl;
//    std::cout << "Jl1 " << Jl1.transpose() << std::endl;
//    std::cout << std::endl;

    MatXXf Q1;
    Q1 = Q.col(0);
//    std::cout << "Q1:\n" << Q1.transpose() << std::endl;

    H += (Jr1.transpose() * Jr1).cast<double>();
    b += (Jr1.transpose() * Jr2).cast<double>();

    Hsc_rootba += (Jr1.transpose() * Q1 * Q1.transpose() * Jr1).cast<double>();
    bsc_rootba += (Jr1.transpose() * Q1 * Q1.transpose() * Jr2).cast<double>();

//    std::cout << "b:\n" << b.transpose() << std::endl;
//    Eigen::Matrix<float, 8, 1> Q1 = Q.block<8, 1>(0, 0);
/*
    if (Jr.block<8, 12>(0, 0) == Eigen::Matrix<float, 8, 12>::Zero()) {
        std::cout << "Q * Jl1:" << (Q * Jl1).transpose() << std::endl;
        std::cout << "Jl:       " << Jl.transpose() << std::endl;
        std::cout << std::endl;
        std::cout << "H_rootba:\n" << Jr.transpose() * Q2 * Q2.transpose() * Jr << std::endl;
        std::cout << "H:\n" << Jr.transpose() * Jr << std::endl;
    }*/
//                    std::cout << "I:\n" << Q1 * Q1.transpose() + Q2 * Q2.transpose() << std::endl;
//        myH_rootba[htIDX] += Jr.transpose() * Q2 * Q2.transpose() * Jr;
    //! 好像Jr.t * Q1 * Q1.t * Jr就是舒尔补部分
    //! 我知道问题在哪里了，每个点的所有残差是互相关联的，或者说是一个整体。
    //! 舒尔补的时候，会每个点的所有残差互乘，QR分解的时候，也是同一个点（同一列Jl）的所有givens rotation的
    //! Q1矩阵连乘
    //! 乘法即是关联的，加法才是可以并行化的
    //! 所以还是要改回遍历point的模式，并且要直接对绝对位姿下进行求解
#endif
    p->Hdd_accAF = Hdd_acc;
    p->bd_accAF = bd_acc;
    p->Hcd_accAF = Hcd_acc;
}
#endif
#ifdef ROOTBA
//    void AccumulatedTopHessianSSE::QR_decomp(Vec8f A, Mat88f &Q, Vec8f &R)
    void AccumulatedTopHessianSSE::QR_decomp(VecXf A, MatXXf &Q, VecXf &R)
    {
        Q.setIdentity();
        R = A;
        MatXXf Q1;
        for (int i = A.size(); i >= 1; i--) {
            float b = R(i);
            if (std::abs(b) < 0.00001)
                continue;
            Q1.setIdentity();

            float a = R(i - 1);
            float r = std::sqrt(a * a + b * b);
            float c = a / r;
            float s = -b / r;
            Q1(i - 1, i - 1) = c;
            Q1(i - 1, i) = s;
            Q1(i, i - 1) = -s;
            Q1(i, i) = c;

            Q = Q * Q1;

            R(i - 1) = r;
            R(i) = 0.0;
        }
    }
#endif
#if 0
    template<int mode>
    void AccumulatedTopHessianSSE::my_addPoint(EnergyFunctional const * const ef,
                                               int tid)	// 0 = active, 1 = linearized, 2=marginalize
    {
        assert(mode==0 || mode==2);

//        return;

        VecCf dc = ef->cDeltaF;

        //! 对该点所有的残差计算相应的矩阵块。Top里的是该残差对应的C, xi部分的偏导，Sch里的是该残差对应的舒尔补
        for (int host = 0; host < nframes[0]; host++)
            for (int target = 0; target < nframes[0]; target++) {
                std::vector<EFResidual *> rv = ef->my_stack[host + target * nframes[0]];
                if (rv.empty())
                    continue;
                for (EFResidual *r: rv) {
//        for(EFResidual* r : p->residualsAll) {
                    if (mode == 0) {
                        assert(!r->isLinearized);
                        if (r->isLinearized || !r->isActive()) continue;
                    }
                    if (mode == 2) {
                        if (!r->isActive()) continue;
                        assert(r->isLinearized);
                    }

                    RawResidualJacobian *rJ = r->J;
                    int htIDX = r->hostIDX + r->targetIDX * nframes[tid];
                    assert(htIDX == host + target * nframes[tid]);

                    Mat18f dp = ef->adHTdeltaF[htIDX];

                    VecNRf resApprox;
                    if (mode == 0)
                        resApprox = rJ->resF;
                    if (mode == 2)
                        resApprox = r->res_toZeroF;

                    // need to compute JI^T * r, and Jab^T * r. (both are 2-vectors).
                    Vec2f JI_r(0, 0);
                    Vec2f Jab_r(0, 0);
                    float rr = 0;
                    for (int i = 0; i < patternNum; i++) {
                        JI_r[0] += resApprox[i] * rJ->JIdx[0][i];
                        JI_r[1] += resApprox[i] * rJ->JIdx[1][i];
                        Jab_r[0] += resApprox[i] * rJ->JabF[0][i];
                        Jab_r[1] += resApprox[i] * rJ->JabF[1][i];
                        rr += resApprox[i] * resApprox[i];
                    }
                    //! 打印host, target, C, xi, ab
//        std::cout << "C:\n" << rJ->JIdx[0] * rJ->Jpdc[0].transpose() +
//                rJ->JIdx[1] * rJ->Jpdc[1].transpose() << std::endl;
//        std::cout << "xi: \n" << rJ->JIdx[0] * rJ->Jpdxi[0].transpose() +
//                rJ->JIdx[1] * rJ->Jpdxi[1].transpose() << std::endl;
//        std::cout << "a: \n" << rJ->JabF[0].transpose() << std::endl;
//        std::cout << "b: \n" << rJ->JabF[1].transpose() << std::endl;
//        std::cout << "res: \n" << resApprox.transpose() << std::endl;
//        std::cout << std::endl;
                    //! 上面的是重投影误差的偏导，另外还要有8×2的矩阵JIdx，即8维的residual和x, y的偏导


//        std::cout << "point has " << p->residualsAll.size() << " residuals" << std::endl;
#ifndef USE_ACC_INSTEAD_OF_myH
                    Eigen::Matrix<float, 8, 13> Jr = Eigen::Matrix<float, 8, 13>::Zero();
                    Jr.block<8, 4>(0, 0) = rJ->JIdx[0] * rJ->Jpdc[0].transpose()
                                           + rJ->JIdx[1] * rJ->Jpdc[1].transpose();
                    Jr.block<8, 6>(0, 4) = rJ->JIdx[0] * rJ->Jpdxi[0].transpose()
                                           + rJ->JIdx[1] * rJ->Jpdxi[1].transpose();
                    Jr.block<8, 1>(0, 10) = rJ->JabF[0];
                    Jr.block<8, 1>(0, 11) = rJ->JabF[1];
                    Jr.block<8, 1>(0, 12) = resApprox;

                    Eigen::Matrix<float, 8, 1> Jl = Eigen::Matrix<float, 8, 1>::Zero();
                    Jl.block<8, 1>(0, 0) = rJ->JIdx[0] * rJ->Jpdd(0, 0)
                                           + rJ->JIdx[1] * rJ->Jpdd(1, 0);
#ifdef ROOTBA
                    Mat88f Q;
                    Eigen::Matrix<float, 8, 1> Jl1 = Eigen::Matrix<float, 8, 1>::Zero();
                    this->QR_decomp(Jl, Q, Jl1);

                    Eigen::Matrix<float, 8, 7> Q2 = Q.block<8, 7>(0, 1);
                    Eigen::Matrix<float, 8, 1> Q1 = Q.block<8, 1>(0, 0);

                    if (Jr.block<8, 12>(0, 0) == Eigen::Matrix<float, 8, 12>::Zero()) {
                        std::cout << "Q * Jl1:" << (Q * Jl1).transpose() << std::endl;
                        std::cout << "Jl:       " << Jl.transpose() << std::endl;
                        std::cout << std::endl;
                        std::cout << "H_rootba:\n" << Jr.transpose() * Q2 * Q2.transpose() * Jr << std::endl;
                        std::cout << "H:\n" << Jr.transpose() * Jr << std::endl;
                    }
//                    std::cout << "I:\n" << Q1 * Q1.transpose() + Q2 * Q2.transpose() << std::endl;
                    myH_rootba[htIDX] += Jr.transpose() * Q2 * Q2.transpose() * Jr;
                    //! 好像Jr.t * Q1 * Q1.t * Jr就是舒尔补部分
                    //! 我知道问题在哪里了，每个点的所有残差是互相关联的，或者说是一个整体。
                    //! 舒尔补的时候，会每个点的所有残差互乘，QR分解的时候，也是同一个点（同一列Jl）的所有givens rotation的
                    //! Q1矩阵连乘
                    //! 乘法即是关联的，加法才是可以并行化的
                    //! 所以还是要改回遍历point的模式，并且要直接对绝对位姿下进行求解
#endif
                    myH[htIDX] += Jr.transpose() * Jr;
#else
                    acc[tid][htIDX].update(
                            rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
                            rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                            rJ->JIdx2(0,0),rJ->JIdx2(0,1),rJ->JIdx2(1,1));


                    acc[tid][htIDX].updateBotRight(
                            rJ->Jab2(0,0), rJ->Jab2(0,1), Jab_r[0],
                            rJ->Jab2(1,1), Jab_r[1],rr);

                    acc[tid][htIDX].updateTopRight(
                            rJ->Jpdc[0].data(), rJ->Jpdxi[0].data(),
                            rJ->Jpdc[1].data(), rJ->Jpdxi[1].data(),
                            rJ->JabJIdx(0,0), rJ->JabJIdx(0,1),
                            rJ->JabJIdx(1,0), rJ->JabJIdx(1,1),
                            JI_r[0], JI_r[1]);
#endif
#if 1
                    nres[tid]++;

                    Vec2f Ji2_Jpdd = rJ->JIdx2 * rJ->Jpdd;
                    r->point->Hdd_accAF += Ji2_Jpdd.dot(rJ->Jpdd);
                    r->point->bd_accAF += JI_r[0] * rJ->Jpdd[0] + JI_r[1] * rJ->Jpdd[1];
                    r->point->Hcd_accAF += rJ->Jpdc[0] * Ji2_Jpdd[0] + rJ->Jpdc[1] * Ji2_Jpdd[1];
#endif
                }
            }
    }
#endif
#if 1
    template void AccumulatedTopHessianSSE::addPoint<0>
            (MatXX &H, VecX &b,
             MatXX &Hsc_rootba, VecX &bsc_rootba,
             EFPoint* p, EnergyFunctional const * const ef, int tid);
//template void AccumulatedTopHessianSSE::addPoint<1>(EFPoint* p, EnergyFunctional const * const ef, int tid);
    template void AccumulatedTopHessianSSE::addPoint<2>
            (MatXX &H, VecX &b,
             MatXX &Hsc_rootba, VecX &bsc_rootba,
             EFPoint* p, EnergyFunctional const * const ef, int tid);
#endif

//template void AccumulatedTopHessianSSE::my_addPoint<0>(EnergyFunctional const * const ef, int tid);
//template void AccumulatedTopHessianSSE::my_addPoint<2>(EnergyFunctional const * const ef, int tid);

void AccumulatedTopHessianSSE::stitchDouble(MatXX &H, VecX &b, EnergyFunctional const * const EF, bool usePrior, bool useDelta, int tid)
{
//    H = MatXX::Zero(nframes[tid]*8+CPARS, nframes[tid]*8+CPARS);
//	b = VecX::Zero(nframes[tid]*8+CPARS);

#if 0
	for(int h=0;h<nframes[tid];h++)
		for(int t=0;t<nframes[tid];t++) {
            //! h:[0, nframes - 1], t:[0, nframes - 1]
			int hIdx = CPARS+h*8;
			int tIdx = CPARS+t*8;
			int aidx = h+nframes[tid]*t;

#ifdef USE_ACC_INSTEAD_OF_myH
			acc[tid][aidx].finish();
			if(acc[tid][aidx].num==0) continue;
			MatPCPC accH = acc[tid][aidx].H.cast<double>();
#else
            MatPCPC accH = myH[aidx].cast<double>();
#endif

			H.block<8,8>(hIdx, hIdx).noalias() +=
                    EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adHost[aidx].transpose();

			H.block<8,8>(tIdx, tIdx).noalias() +=
                    EF->adTarget[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

			H.block<8,8>(hIdx, tIdx).noalias() +=
                    EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

			H.block<8,CPARS>(hIdx,0).noalias() +=
                    EF->adHost[aidx] * accH.block<8,CPARS>(CPARS,0);

			H.block<8,CPARS>(tIdx,0).noalias() +=
                    EF->adTarget[aidx] * accH.block<8,CPARS>(CPARS,0);

            //! <C, C>是没有伴随的
			H.topLeftCorner<CPARS,CPARS>().noalias() += accH.block<CPARS,CPARS>(0,0);

			b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8,1>(CPARS,8+CPARS);

			b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8,1>(CPARS,8+CPARS);

			b.head<CPARS>().noalias() += accH.block<CPARS,1>(0,8+CPARS);
		}

	// ----- new: copy transposed parts.
	for(int h=0;h<nframes[tid];h++) {
		int hIdx = CPARS+h*8;
		H.block<CPARS,8>(0,hIdx).noalias() = H.block<8,CPARS>(hIdx,0).transpose();

		for(int t=h+1;t<nframes[tid];t++) {
			int tIdx = CPARS+t*8;
			H.block<8,8>(hIdx, tIdx).noalias() += H.block<8,8>(tIdx, hIdx).transpose();
			H.block<8,8>(tIdx, hIdx).noalias() = H.block<8,8>(hIdx, tIdx).transpose();
		}
	}
#endif

	if(usePrior) {
		assert(useDelta);
		H.diagonal().head<CPARS>() += EF->cPrior;
		b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
		for (int h=0;h<nframes[tid];h++) {
            H.diagonal().segment<8>(CPARS+h*8) += EF->frames[h]->prior;
            b.segment<8>(CPARS+h*8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
		}
	}

}

#if 0
    void AccumulatedTopHessianSSE::stitchDouble_rootba(MatXX &H, VecX &b, EnergyFunctional const * const EF, bool usePrior, bool useDelta, int tid)
    {
        H = MatXX::Zero(nframes[tid]*8+CPARS, nframes[tid]*8+CPARS);
        b = VecX::Zero(nframes[tid]*8+CPARS);

        for(int h=0;h<nframes[tid];h++)
            for(int t=0;t<nframes[tid];t++) {
                //! h:[0, nframes - 1], t:[0, nframes - 1]
                int hIdx = CPARS+h*8;
                int tIdx = CPARS+t*8;
                int aidx = h+nframes[tid]*t;

#ifdef USE_ACC_INSTEAD_OF_myH
                acc[tid][aidx].finish();
			if(acc[tid][aidx].num==0) continue;
			MatPCPC accH = acc[tid][aidx].H.cast<double>();
#else
                MatPCPC accH = myH_rootba[aidx].cast<double>();
#endif

                H.block<8,8>(hIdx, hIdx).noalias() +=
                        EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adHost[aidx].transpose();

                H.block<8,8>(tIdx, tIdx).noalias() +=
                        EF->adTarget[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

                H.block<8,8>(hIdx, tIdx).noalias() +=
                        EF->adHost[aidx] * accH.block<8,8>(CPARS,CPARS) * EF->adTarget[aidx].transpose();

                H.block<8,CPARS>(hIdx,0).noalias() +=
                        EF->adHost[aidx] * accH.block<8,CPARS>(CPARS,0);

                H.block<8,CPARS>(tIdx,0).noalias() +=
                        EF->adTarget[aidx] * accH.block<8,CPARS>(CPARS,0);

                //! <C, C>是没有伴随的
                H.topLeftCorner<CPARS,CPARS>().noalias() += accH.block<CPARS,CPARS>(0,0);

                b.segment<8>(hIdx).noalias() += EF->adHost[aidx] * accH.block<8,1>(CPARS,8+CPARS);

                b.segment<8>(tIdx).noalias() += EF->adTarget[aidx] * accH.block<8,1>(CPARS,8+CPARS);

                b.head<CPARS>().noalias() += accH.block<CPARS,1>(0,8+CPARS);
            }

        // ----- new: copy transposed parts.
        for(int h=0;h<nframes[tid];h++) {
            int hIdx = CPARS+h*8;
            H.block<CPARS,8>(0,hIdx).noalias() = H.block<8,CPARS>(hIdx,0).transpose();

            for(int t=h+1;t<nframes[tid];t++) {
                int tIdx = CPARS+t*8;
                H.block<8,8>(hIdx, tIdx).noalias() += H.block<8,8>(tIdx, hIdx).transpose();
                H.block<8,8>(tIdx, hIdx).noalias() = H.block<8,8>(hIdx, tIdx).transpose();
            }
        }

        if(usePrior) {
            assert(useDelta);
            H.diagonal().head<CPARS>() += EF->cPrior;
            b.head<CPARS>() += EF->cPrior.cwiseProduct(EF->cDeltaF.cast<double>());
            for (int h=0;h<nframes[tid];h++) {
                H.diagonal().segment<8>(CPARS+h*8) += EF->frames[h]->prior;
                b.segment<8>(CPARS+h*8) += EF->frames[h]->prior.cwiseProduct(EF->frames[h]->delta_prior);
            }
        }

    }
#endif


}


