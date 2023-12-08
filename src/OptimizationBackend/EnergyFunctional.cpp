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


#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "OptimizationBackend/AccumulatedSCHessian.h"
#include "OptimizationBackend/AccumulatedTopHessian.h"

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


bool EFAdjointsValid = false;
bool EFIndicesValid = false;
bool EFDeltaValid = false;


void EnergyFunctional::setAdjointsF(CalibHessian* Hcalib)
{

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;
	adHost = new Mat88[nFrames*nFrames];
	adTarget = new Mat88[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++) {
			FrameHessian* host = frames[h]->data;
			FrameHessian* target = frames[t]->data;

			SE3 hostToTarget = target->get_worldToCam_evalPT() * host->get_worldToCam_evalPT().inverse();

			Mat88 AH = Mat88::Identity();
			Mat88 AT = Mat88::Identity();

			AH.topLeftCorner<6,6>() = -hostToTarget.Adj().transpose();
			AT.topLeftCorner<6,6>() = Mat66::Identity();


			Vec2f affLL = AffLight::fromToVecExposure(host->ab_exposure, target->ab_exposure, host->aff_g2l_0(), target->aff_g2l_0()).cast<float>();
			AT(6,6) = -affLL[0];
			AH(6,6) = affLL[0];
			AT(7,7) = -1;
			AH(7,7) = affLL[0];

			AH.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AH.block<3,8>(3,0) *= SCALE_XI_ROT;
			AH.block<1,8>(6,0) *= SCALE_A;
			AH.block<1,8>(7,0) *= SCALE_B;
			AT.block<3,8>(0,0) *= SCALE_XI_TRANS;
			AT.block<3,8>(3,0) *= SCALE_XI_ROT;
			AT.block<1,8>(6,0) *= SCALE_A;
			AT.block<1,8>(7,0) *= SCALE_B;

			adHost[h+t*nFrames] = AH;
			adTarget[h+t*nFrames] = AT;
		}
	cPrior = VecC::Constant(setting_initialCalibHessian);


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	adHostF = new Mat88f[nFrames*nFrames];
	adTargetF = new Mat88f[nFrames*nFrames];

	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++) {
			adHostF[h+t*nFrames] = adHost[h+t*nFrames].cast<float>();
			adTargetF[h+t*nFrames] = adTarget[h+t*nFrames].cast<float>();
		}

//	cPriorF = cPrior.cast<float>();

	EFAdjointsValid = true;
}



EnergyFunctional::EnergyFunctional()
{
	adHost=0;
	adTarget=0;


//	red=0;

	adHostF=0;
	adTargetF=0;
	adHTdeltaF=0;

	nFrames = nResiduals = nPoints = 0;

	HM = MatXX::Zero(CPARS,CPARS);
	bM = VecX::Zero(CPARS);

//	accSSE_top_L = new AccumulatedTopHessianSSE();
	accSSE_top_A = new AccumulatedTopHessianSSE();
	accSSE_bot = new AccumulatedSCHessianSSE();

	resInA = resInL = resInM = 0;
//	currentLambda=0;
}
EnergyFunctional::~EnergyFunctional()
{
	for(EFFrame* f : frames) {
		for(EFPoint* p : f->points) {
			for(EFResidual* r : p->residualsAll) {
				r->data->efResidual=0;
				delete r;
			}
			p->data->efPoint=0;
			delete p;
		}
        // f -> data -> efFrame即f自己
		f->data->efFrame=0;
		delete f;
	}

	if(adHost != 0) delete[] adHost;
	if(adTarget != 0) delete[] adTarget;


	if(adHostF != 0) delete[] adHostF;
	if(adTargetF != 0) delete[] adTargetF;
	if(adHTdeltaF != 0) delete[] adHTdeltaF;


	delete accSSE_top_A;
	delete accSSE_bot;
}

// 从前端读取数据，设置f -> delta, f -> delta_prior, p -> deltaF
void EnergyFunctional::setDeltaF(CalibHessian* HCalib)
{
	if(adHTdeltaF != 0) delete[] adHTdeltaF;
	adHTdeltaF = new Mat18f[nFrames*nFrames];
	for(int h=0;h<nFrames;h++)
		for(int t=0;t<nFrames;t++) {
			int idx = h+t*nFrames;
			adHTdeltaF[idx] =
                    frames[h]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() *
                            adHostF[idx] +
                            frames[t]->data->get_state_minus_stateZero().head<8>().cast<float>().transpose() *
                                    adTargetF[idx];
		}

	cDeltaF = HCalib->value_minus_value_zero.cast<float>();
	for(EFFrame* f : frames) {
		f->delta = f->data->get_state_minus_stateZero().head<8>();
		f->delta_prior = f->data->get_state().head<8>();

		/*for(EFPoint* p : f->points) {
            p->deltaF = p->data->idepth - p->data->idepth_zero;
            // 逆深度其实并没有用FEJ，所以delta === 0.0
            assert(p -> deltaF < 0.00001);
        }*/
	}

	EFDeltaValid = true;
}

// accumulates & shifts L.
void EnergyFunctional::accumulateAF_MT(MatXX &H, VecX &b, bool MT)
{
    accSSE_top_A->setZero(nFrames);


//        TicToc timer_addPoint;
        for (EFFrame *f : frames)
            for (EFPoint *p : f->points) {
                accSSE_top_A->addPoint<0>(p, this, 0);
            }
//        auto times_addPoint = timer_addPoint.toc();
//        std::cout << "addPoint cost time " << times_addPoint << std::endl;

//        TicToc timer_stitchDouble;
        accSSE_top_A->stitchDouble(H,b,this,true, true);
//        auto times_stitchDouble = timer_stitchDouble.toc();
//        std::cout << "stitchDouble cost time " << times_stitchDouble << std::endl;

    resInA = accSSE_top_A->nres[0];
}
#if 1
void EnergyFunctional::accumulateSCF_MT(MatXX &H, VecX &b, bool MT)
{
//    TicToc timer_SCF;
    accSSE_bot->setZero(nFrames);
    for(EFFrame* f : frames)
        for(EFPoint* p : f->points)
            accSSE_bot->addPoint(p);
    accSSE_bot->stitchDouble(H, b,this);
//    auto times_SCF = timer_SCF.toc();
//    std::cout << "SCF cost time " << times_SCF << std::endl;
}
#endif
//! resubstitute是输出，把后端的解输出到前端
void EnergyFunctional::resubstituteF_MT(VecX x, CalibHessian* HCalib, bool MT)
{
	assert(x.size() == CPARS+nFrames*8);

	VecXf xF = x.cast<float>();
	HCalib->step = - x.head<CPARS>();

	Mat18f* xAd = new Mat18f[nFrames*nFrames];
	VecCf cstep = xF.head<CPARS>();

//    std::cout << "....cstep...." << cstep.matrix() << std::endl;

	for(EFFrame* h : frames) {
        // 后端数据生成，推送到前端( ->data )
		h->data->step.head<8>() = - x.segment<8>(CPARS+8*h->idx);
		h->data->step.tail<2>().setZero();

		for(EFFrame* t : frames)
            // xAd[h][t] = x[h][0].t() * adHostF[t][h] + x[t][0].t() * adTargetF[t][h]
			xAd[nFrames*h->idx + t->idx] = xF.segment<8>(CPARS+8*h->idx).transpose()
                                           * adHostF[h->idx+nFrames*t->idx]
                                           + xF.segment<8>(CPARS+8*t->idx).transpose()
                                             * adTargetF[h->idx+nFrames*t->idx];
	}
    //! 通过逆深度对于光度内参的偏导，乘以光度参数的delta，可以得到优化更新的逆深度
    resubstituteFPt(cstep, xAd, 0, allPoints.size(), 0,0);

	delete[] xAd;
}

void EnergyFunctional::resubstituteFPt(
        const VecCf &xc, Mat18f* xAd, int min, int max, Vec10* stats, int tid)
{
	for(int k=min;k<max;k++) {
        // 后端的points
		EFPoint* p = allPoints[k];

		int ngoodres = 0;
		for(EFResidual* r : p->residualsAll) if(r->isActive()) ngoodres++;
		if(ngoodres==0) {
			p->data->step = 0;
			continue;
		}
		float b = p->bdSumF;
//        std::cout << " 1 b: " << b << std::endl;
//        std::cout << " xc: " << xc.matrix() << std::endl;
		b -= xc.dot(p->Hcd_accAF /*+ p->Hcd_accLF*/);
//        std::cout << " 2 b: " << b << std::endl;

		for(EFResidual* r : p->residualsAll) {
			if(!r->isActive()) continue;
			b -= xAd[r->hostIDX*nFrames + r->targetIDX] * r->JpJdF;

//            std::cout << " 3 b: " << b << std::endl;
		}

//        assert(std::isfinite(b));
//        assert(std::isfinite(p->HdiF));
		p->data->step = - b*p->HdiF;
		assert(std::isfinite(p->data->step));
	}
}

double EnergyFunctional::calcMEnergyF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	VecX delta = getStitchedDeltaF();
	return delta.dot(2*bM + HM*delta);
}

EFResidual* EnergyFunctional::insertResidual(PointFrameResidual* r)
{
//  存雅克比
	EFResidual* efr = new EFResidual(r, r->point->efPoint, r->host->efFrame, r->target->efFrame);
	efr->idxInAll = r->point->efPoint->residualsAll.size();
	r->point->efPoint->residualsAll.push_back(efr);
	if(efr->data->stereoResidualFlag == false)
	    connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;
//     connectivityMap[(((uint64_t)efr->host->frameID) << 32) + ((uint64_t)efr->target->frameID)][0]++;
	nResiduals++;
	r->efResidual = efr;
	
	return efr;
}
EFFrame* EnergyFunctional::insertFrame(FrameHessian* fh, CalibHessian* Hcalib)
{
	EFFrame* eff = new EFFrame(fh);
	eff->idx = frames.size();
	frames.push_back(eff);

	nFrames++;
	fh->efFrame = eff;
	//stereo
    // 这里增加右图
	EFFrame* eff_right = new EFFrame(fh->frame_right);
	eff_right->idx = frames.size()+1000000;
	fh->frame_right->efFrame = eff_right;

	assert(HM.cols() == 8*nFrames+CPARS-8);
	bM.conservativeResize(8*nFrames+CPARS);
	HM.conservativeResize(8*nFrames+CPARS,8*nFrames+CPARS);
	bM.tail<8>().setZero();
	HM.rightCols<8>().setZero();
	HM.bottomRows<8>().setZero();

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	setAdjointsF(Hcalib);
	makeIDX();

	for(EFFrame* fh2 : frames) {
        connectivityMap[(((uint64_t)eff->frameID) << 32) + ((uint64_t)fh2->frameID)] = Eigen::Vector2i(0,0);
		if(fh2 != eff)
            connectivityMap[(((uint64_t)fh2->frameID) << 32) + ((uint64_t)eff->frameID)] = Eigen::Vector2i(0,0);
	}

	return eff;
}
EFPoint* EnergyFunctional::insertPoint(PointHessian* ph)
{
	EFPoint* efp = new EFPoint(ph, ph->host->efFrame);
	efp->idxInPoints = ph->host->efFrame->points.size();
	ph->host->efFrame->points.push_back(efp);

	nPoints++;
	ph->efPoint = efp;

	EFIndicesValid = false;

	return efp;
}

void EnergyFunctional::dropResidual(EFResidual* r)
{
	EFPoint* p = r->point;
	assert(r == p->residualsAll[r->idxInAll]);

	p->residualsAll[r->idxInAll] = p->residualsAll.back();
	p->residualsAll[r->idxInAll]->idxInAll = r->idxInAll;
	p->residualsAll.pop_back();

	if(r->isActive())
		r->host->data->shell->statistics_goodResOnThis++;
	else
		r->host->data->shell->statistics_outlierResOnThis++;

	if(r->data->stereoResidualFlag == false)
		connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
//     connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][0]--;
	nResiduals--;
	r->data->efResidual=0;
	delete r;
}
void EnergyFunctional::marginalizeFrame(EFFrame* fh)
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	assert((int)fh->points.size()==0);

    std::cout << "marg frame: " << fh->frameID << std::endl;
	int ndim = nFrames*8+CPARS-8;// new dimension
	int odim = nFrames*8+CPARS;// old dimension

//	VecX eigenvaluesPre = HM.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//
	if((int)fh->idx != (int)frames.size()-1) {
		int io = fh->idx*8+CPARS;	// index of frame to move to end
		int ntail = 8*(nFrames-fh->idx-1);
		assert((io+8+ntail) == nFrames*8+CPARS);

		Vec8 bTmp = bM.segment<8>(io);
		VecX tailTMP = bM.tail(ntail);
		bM.segment(io,ntail) = tailTMP;
		bM.tail<8>() = bTmp;

		MatXX HtmpCol = HM.block(0,io,odim,8);
		MatXX rightColsTmp = HM.rightCols(ntail);
		HM.block(0,io,odim,ntail) = rightColsTmp;
		HM.rightCols(8) = HtmpCol;

		MatXX HtmpRow = HM.block(io,0,8,odim);
		MatXX botRowsTmp = HM.bottomRows(ntail);
		HM.block(io,0,ntail,odim) = botRowsTmp;
		HM.bottomRows(8) = HtmpRow;
	}


//	// marginalize. First add prior here, instead of to active.
    HM.bottomRightCorner<8,8>().diagonal() += fh->prior;
    //! cwiseProduct 点乘
    //! 逐点相乘，如果是两个向量，结果也为一个等维度向量
    //! 因为这个fh->prior是在对角线上，所以会有这么一个操作
    bM.tail<8>() += fh->prior.cwiseProduct(fh->delta_prior);

//	std::cout << std::setprecision(16) << "HMPre:\n" << HM << "\n\n";

	VecX SVec = (HM.diagonal().cwiseAbs()+VecX::Constant(HM.cols(), 10)).cwiseSqrt();
	VecX SVecI = SVec.cwiseInverse();

//	std::cout << std::setprecision(16) << "SVec: " << SVec.transpose() << "\n\n";
//	std::cout << std::setprecision(16) << "SVecI: " << SVecI.transpose() << "\n\n";

	// scale!
	MatXX HMScaled = SVecI.asDiagonal() * HM * SVecI.asDiagonal();
	VecX bMScaled =  SVecI.asDiagonal() * bM;

	// invert bottom part!
	Mat88 hpi = HMScaled.bottomRightCorner<8,8>();
	hpi = 0.5f*(hpi+hpi);
	hpi = hpi.inverse();
	hpi = 0.5f*(hpi+hpi);

	// schur-complement!
    //! 把右下角舒尔补掉了
	MatXX bli = HMScaled.bottomLeftCorner(8,ndim).transpose() * hpi;
	HMScaled.topLeftCorner(ndim,ndim).noalias() -=
            bli * HMScaled.bottomLeftCorner(8,ndim);
	bMScaled.head(ndim).noalias() -= bli*bMScaled.tail<8>();

	//unscale!
	HMScaled = SVec.asDiagonal() * HMScaled * SVec.asDiagonal();
	bMScaled = SVec.asDiagonal() * bMScaled;

	// set.
    //! 把右下角丢弃
	HM = 0.5*(HMScaled.topLeftCorner(ndim,ndim) +
            HMScaled.topLeftCorner(ndim,ndim).transpose());
	bM = bMScaled.head(ndim);

//    std::cout << "bM: " << bM.size() << std::endl;
//    std::cout << "bMScaled: " << bMScaled.size() << std::endl;

	// remove from vector, without changing the order!
	for(unsigned int i=fh->idx; i+1<frames.size();i++) {
		frames[i] = frames[i+1];
		frames[i]->idx = i;
	}
	frames.pop_back();
	nFrames--;
	fh->data->efFrame=0;

	assert((int)frames.size()*8+CPARS == (int)HM.rows());
	assert((int)frames.size()*8+CPARS == (int)HM.cols());
	assert((int)frames.size()*8+CPARS == (int)bM.size());
	assert((int)frames.size() == (int)nFrames);

//	VecX eigenvaluesPost = HM.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());

//	std::cout << std::setprecision(16) << "HMPost:\n" << HM << "\n\n";

//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";
//	std::cout << "EigPost: " << eigenvaluesPost.transpose() << "\n";

	EFIndicesValid = false;
	EFAdjointsValid=false;
	EFDeltaValid=false;

	makeIDX();
	delete fh;
}


void EnergyFunctional::marginalizePointsF()
{
	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	allPointsToMarg.clear();
	for(EFFrame* f : frames) {
		for(int i=0;i<(int)f->points.size();i++) {
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_MARGINALIZE) {
				p->priorF *= setting_idepthFixPriorMargFac;
				for(EFResidual* r : p->residualsAll)
					if(r->isActive())
						if(r->data->stereoResidualFlag == false)
							 connectivityMap[(((uint64_t)r->host->frameID) << 32) + ((uint64_t)r->target->frameID)][1]++;
				allPointsToMarg.push_back(p);
			}
		}
	}
    std::cout << "marg points: " << allPointsToMarg.size() <<  std::endl;
// 	LOG(INFO)<<"allPointsToMarg.size(): "<<allPointsToMarg.size();

    MatXX M, Msc;
    VecX Mb, Mbsc;

#ifdef ROOTBA
    MatXX M1;
    VecX Mb1;
#endif
    //! HM, bM是祖传的。这里M - Msc, Mb - Mbsc是本次新的增量，再加到祖传的HM，bM上。

	accSSE_bot->setZero(nFrames);
	accSSE_top_A->setZero(nFrames);

	for(EFPoint* p : allPointsToMarg) {
        accSSE_top_A->addPoint<2>(p,this);
		accSSE_bot->addPoint(p,false);
		removePoint(p);
	}

	accSSE_top_A->stitchDouble(M,Mb,this,false,false);

    // bot指的是從bot生成（舒爾補）對應的top
	accSSE_bot->stitchDouble(Msc,Mbsc,this);

	resInM+= accSSE_top_A->nres[0];

	MatXX H =  M-Msc;
    VecX b =  Mb-Mbsc;

    // 每一个点都对应一个完整的H, b。或者说，Marg: point -> (H_nxn, b_n)
	HM += setting_margWeightFac*H;
	bM += setting_margWeightFac*b;


//    std::cout << "bM size in marg points: " << bM.size() << std::endl;

	EFIndicesValid = false;
	makeIDX();
}

void EnergyFunctional::dropPointsF()
{
    int count = 0;
	for(EFFrame* f : frames) {
		for(int i=0;i<(int)f->points.size();i++) {
			EFPoint* p = f->points[i];
			if(p->stateFlag == EFPointStatus::PS_DROP) {
				removePoint(p);
				i--;
                count++;
			}
		}
	}
    std::cout << "drop points: " << count << std::endl;

	EFIndicesValid = false;
	makeIDX();
}


void EnergyFunctional::removePoint(EFPoint* p)
{
	for(EFResidual* r : p->residualsAll)
		dropResidual(r);

	EFFrame* h = p->host;
	h->points[p->idxInPoints] = h->points.back();
	h->points[p->idxInPoints]->idxInPoints = p->idxInPoints;
	h->points.pop_back();

	nPoints--;
	p->data->efPoint = 0;

	EFIndicesValid = false;

	delete p;
}

void EnergyFunctional::orthogonalize(VecX* b, MatXX* H)
{
//	VecX eigenvaluesPre = H.eigenvalues().real();
//	std::sort(eigenvaluesPre.data(), eigenvaluesPre.data()+eigenvaluesPre.size());
//	std::cout << "EigPre:: " << eigenvaluesPre.transpose() << "\n";

	// decide to which nullspaces to orthogonalize.
	std::vector<VecX> ns;
	ns.insert(ns.end(), lastNullspaces_pose.begin(), lastNullspaces_pose.end());
	ns.insert(ns.end(), lastNullspaces_scale.begin(), lastNullspaces_scale.end());
//	if(setting_affineOptModeA <= 0)
//		ns.insert(ns.end(), lastNullspaces_affA.begin(), lastNullspaces_affA.end());
//	if(setting_affineOptModeB <= 0)
//		ns.insert(ns.end(), lastNullspaces_affB.begin(), lastNullspaces_affB.end());

	// make Nullspaces matrix
	MatXX N(ns[0].rows(), ns.size());
	for(unsigned int i=0;i<ns.size();i++)
		N.col(i) = ns[i].normalized();

	// compute Npi := N * (N' * N)^-1 = pseudo inverse of N.
	Eigen::JacobiSVD<MatXX> svdNN(N, Eigen::ComputeThinU | Eigen::ComputeThinV);

	VecX SNN = svdNN.singularValues();
	double minSv = 1e10, maxSv = 0;
	for(int i=0;i<SNN.size();i++) {
		if(SNN[i] < minSv) minSv = SNN[i];
		if(SNN[i] > maxSv) maxSv = SNN[i];
	}
	for(int i=0;i<SNN.size();i++)
		{ if(SNN[i] > setting_solverModeDelta*maxSv) SNN[i] = 1.0 / SNN[i]; else SNN[i] = 0; }

	MatXX Npi = svdNN.matrixU() * SNN.asDiagonal() * svdNN.matrixV().transpose(); 	// [dim] x 9.
	MatXX NNpiT = N*Npi.transpose(); 	// [dim] x [dim].
	MatXX NNpiTS = 0.5*(NNpiT + NNpiT.transpose());	// = N * (N' * N)^-1 * N'.

	if(b!=0) *b -= NNpiTS * *b;
	if(H!=0) *H -= NNpiTS * *H * NNpiTS;

//	std::cout << std::setprecision(16) << "Orth SV: " << SNN.reverse().transpose() << "\n";

//	VecX eigenvaluesPost = H.eigenvalues().real();
//	std::sort(eigenvaluesPost.data(), eigenvaluesPost.data()+eigenvaluesPost.size());
//	std::cout << "EigPost:: " << eigenvaluesPost.transpose() << "\n";
}

void EnergyFunctional::solveSystemF(int iteration, double lambda, CalibHessian* HCalib)
{

    TicToc timer_solveSystemF;

	if(setting_solverMode & SOLVER_USE_GN) lambda=0;
	if(setting_solverMode & SOLVER_FIX_LAMBDA) lambda = 1e-5;

	assert(EFDeltaValid);
	assert(EFAdjointsValid);
	assert(EFIndicesValid);

	MatXX  HA_top, H_sc, H1;
	VecX   bA_top, bM_top, b_sc, b1;

#ifdef ROOTBA
//    test_QR_decomp();
//    exit(1);
#endif

    // 边缘化后的求解，三个元素。详见涂金戈。
    // A是不受边缘化影响的残差和雅克比， L是线性化的残差和雅克比，SCF是计算舒尔补
    // 不过其实一个已经线性化的残差都没有

//    TicToc timer_AF;
    accumulateAF_MT(HA_top, bA_top, multiThreading);
//    auto times_AF = timer_AF.toc();
//    std::cout << "AF cost time " << times_AF << std::endl;

//  跟上次邊緣化的幀無關的points們，加上新補增的points們，計算最新的殘差，SC掉points，生成最新的H, b系統
//    TicToc timer_SCF;
	accumulateSCF_MT(H_sc, b_sc,multiThreading);
//    auto times_SCF = timer_SCF.toc();
//    std::cout << "SCF cost time " << times_SCF << std::endl;

//    std::cout << "b_sc:\n" << b_sc.transpose() << std::endl;
//    std::cout << std::endl;

    //! 这里的关键是zero点，也就是固定线性化点，之前制作HM, bM的时候，在固定线性化点做了一次优化，然后再回退，
    //! 即回退到了zero点，制作了HM，bM;
    //! 之后应该是在前端一直保存了zero点，所以每次要使用这个祖传HM, bM的时候，都需要现求一次state - state_zero
    //! 然后将这个bM调整如下
	bM_top = (bM+ HM * getStitchedDeltaF());
//    std::cout << "getStitchedDeltaF(): " << getStitchedDeltaF() << std::endl;

	MatXX HFinal_top;
	VecX bFinal_top;

    {  // 就是 M + A + SC，只是因爲lambda稍有改動
		HFinal_top =  HM + HA_top;
		bFinal_top =  bM_top + bA_top - b_sc;

//        std::cout << "bM size: " << bM.size() << std::endl;
//        std::cout << "bA_top size: " << bA_top.size() << std::endl;
//        std::cout << "bM_top size: " << bM_top.size() << std::endl;

        // lastHS和lastbS只有打印時會調用，無實際作用
		lastHS = HFinal_top - H_sc;
		lastbS = bFinal_top;

		for(int i=0;i<8*nFrames+CPARS;i++) HFinal_top(i,i) *= (1+lambda);

        HFinal_top -= H_sc * (1.0f/(1+lambda));

        // 上面的把HM, HA_top略爲放大，把H_sc略爲縮小，整體的HFinal_top略爲放大
//        std::cout << "H:\n" << HFinal_top << std::endl;
//        std::cout << "b:\n" << bFinal_top << std::endl;
	}
//    exit(1);
    auto times_solveSystemF = timer_solveSystemF.toc();
//    std::cout << "solveSystemF cost time " << times_solveSystemF << std::endl;

	VecX x;

//    printf("setting solverMode: %x\n", setting_solverMode);
	if(setting_solverMode & SOLVER_SVD) {
		VecX SVecI = HFinal_top.diagonal().cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		VecX bFinalScaled  = SVecI.asDiagonal() * bFinal_top;
		Eigen::JacobiSVD<MatXX> svd(HFinalScaled, Eigen::ComputeThinU | Eigen::ComputeThinV);

		VecX S = svd.singularValues();
		double minSv = 1e10, maxSv = 0;
		for(int i=0;i<S.size();i++) {
			if(S[i] < minSv) minSv = S[i];
			if(S[i] > maxSv) maxSv = S[i];
		}

		VecX Ub = svd.matrixU().transpose()*bFinalScaled;
		int setZero=0;
		for(int i=0;i<Ub.size();i++) {
			if(S[i] < setting_solverModeDelta*maxSv)
			{ Ub[i] = 0; setZero++; }

			if((setting_solverMode & SOLVER_SVD_CUT7) && (i >= Ub.size()-7))
			{ Ub[i] = 0; setZero++; }
			else Ub[i] /= S[i];
		}
		x = SVecI.asDiagonal() * svd.matrixV() * Ub;
	} else {
//        printf("........hello\n");
		VecX SVecI = (HFinal_top.diagonal()+VecX::Constant(HFinal_top.cols(), 10)).cwiseSqrt().cwiseInverse();
		MatXX HFinalScaled = SVecI.asDiagonal() * HFinal_top * SVecI.asDiagonal();
		x = SVecI.asDiagonal() * HFinalScaled.llt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
//        x = SVecI.asDiagonal() * HFinalScaled.ldlt().solve(SVecI.asDiagonal() * bFinal_top);//  SVec.asDiagonal() * svd.matrixV() * Ub;
	}

	if((setting_solverMode & SOLVER_ORTHOGONALIZE_X) ||
            (iteration >= 2 &&
                    (setting_solverMode & SOLVER_ORTHOGONALIZE_X_LATER))) {
		VecX xOld = x;
		orthogonalize(&x, 0);
	}

	lastX = x;

	//resubstituteF(x, HCalib);
	resubstituteF_MT(x, HCalib,multiThreading);


}
void EnergyFunctional::makeIDX()
{
	for(unsigned int idx=0;idx<frames.size();idx++)
		frames[idx]->idx = idx;

	allPoints.clear();

	for(EFFrame* f : frames)
		for(EFPoint* p : f->points) {
			allPoints.push_back(p);
			for(EFResidual* r : p->residualsAll) {
				r->hostIDX = r->host->idx;
				r->targetIDX = r->target->idx;
				if(r->data->stereoResidualFlag == true) {
				  r->targetIDX = frames[frames.size()-1]->idx;
				}
			}
		}

	EFIndicesValid=true;
}

VecX EnergyFunctional::getStitchedDeltaF() const
{
	VecX d = VecX(CPARS+nFrames*8).setZero();
    d.head<CPARS>() = cDeltaF.cast<double>();
    // rkf
	for(int h=0;h<nFrames;h++)
        d.segment<8>(CPARS+8*h) = frames[h]->delta;
	return d;
}

#ifdef ROOTBA
#include <Eigen/SparseCore>
#include <Eigen/IterativeLinearSolvers>
    void EnergyFunctional::test_QR_decomp()
    {
        int m=1000000, n = 100;
        VecX x(n), b(m);
//    Eigen::SparseMatrix<double> A(m, n);
        Eigen::SparseMatrix<double, 1000000, 100> A = Eigen::SparseMatrix<double, 1000000, 100>::Random();
// fill A and b
//    Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double> > lscg;
        Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double, 1000000, 100> > lscg;
        lscg.compute(A);
        x = lscg.solve(b);
        std::cout << "#iterations:     " << lscg.iterations() << std::endl;
        std::cout << "estimated error: " << lscg.error()      << std::endl;
        std::cout << "x: " << x.transpose() << std::endl;
// update b, and solve again
        x = lscg.solve(b);
//    VecXf A;
//    A.setZero(8, 1);
//
//    A << 3.5, 2.1, 7.7, 0.7, 9.6, 10.1, 2.3, 5.5;
//
//    MatXXf Q2;
//
//    accSSE_top_A->my_generate_Q2(Q2, A);
//
//    std::cout << "Q2:\n" << Q2 << std::endl;
//    std::cout << "Q2.transpose() * A:\n" << Q2.transpose() * A << std::endl;

//    MatXXf Q;
//    VecXf R;

// Q2: 第0列到第2列，总共是0-3总共4列。
//    int n = 5;
//    for (int i = 3; i >= 0; i--)
//        accSSE_top_A->my_print_Q2(A, i, n);
    }
#endif


}
