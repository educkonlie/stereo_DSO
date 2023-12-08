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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"
 
#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ResidualProjections.h"
#include "FullSystem/ImmaturePoint.h"

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/CoarseInitializer.h"

#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "IOWrapper/Output3DWrapper.h"

#include "util/ImageAndExposure.h"

#include <cmath>

namespace dso
{
int FrameHessian::instanceCounter=0;
int PointHessian::instanceCounter=0;
int CalibHessian::instanceCounter=0;

FullSystem::FullSystem()
{
//    Vec3 stereo_warp_t = Vec3(-baseline, 0, 0);
    Vec3 stereo_warp_t = Vec3(-baseline, 0, 0);
    SO3 stereo_warp_R = SO3(Mat33::Identity());
    stereo_warp_Rt = SE3(stereo_warp_R, stereo_warp_t);

	int retstat =0;
	if(setting_logStuff) {
		retstat += system("rm -rf logs");
		retstat += system("mkdir logs");

		retstat += system("rm -rf mats");
		retstat += system("mkdir mats");

		calibLog = new std::ofstream();
		calibLog->open("logs/calibLog.txt", std::ios::trunc | std::ios::out);
		calibLog->precision(12);

		numsLog = new std::ofstream();
		numsLog->open("logs/numsLog.txt", std::ios::trunc | std::ios::out);
		numsLog->precision(10);

		coarseTrackingLog = new std::ofstream();
		coarseTrackingLog->open("logs/coarseTrackingLog.txt", std::ios::trunc | std::ios::out);
		coarseTrackingLog->precision(10);

		eigenAllLog = new std::ofstream();
		eigenAllLog->open("logs/eigenAllLog.txt", std::ios::trunc | std::ios::out);
		eigenAllLog->precision(10);

		eigenPLog = new std::ofstream();
		eigenPLog->open("logs/eigenPLog.txt", std::ios::trunc | std::ios::out);
		eigenPLog->precision(10);

		eigenALog = new std::ofstream();
		eigenALog->open("logs/eigenALog.txt", std::ios::trunc | std::ios::out);
		eigenALog->precision(10);

		DiagonalLog = new std::ofstream();
		DiagonalLog->open("logs/diagonal.txt", std::ios::trunc | std::ios::out);
		DiagonalLog->precision(10);

		variancesLog = new std::ofstream();
		variancesLog->open("logs/variancesLog.txt", std::ios::trunc | std::ios::out);
		variancesLog->precision(10);

		nullspacesLog = new std::ofstream();
		nullspacesLog->open("logs/nullspacesLog.txt", std::ios::trunc | std::ios::out);
		nullspacesLog->precision(10);
	} else {
		nullspacesLog=0;
		variancesLog=0;
		DiagonalLog=0;
		eigenALog=0;
		eigenPLog=0;
		eigenAllLog=0;
		numsLog=0;
		calibLog=0;
	}

	assert(retstat!=293847);

	selectionMap = new float[wG[0]*hG[0]];

	coarseDistanceMap = new CoarseDistanceMap(wG[0], hG[0]);
	coarseTracker = new CoarseTracker(wG[0], hG[0]);
	coarseTracker_forNewKF = new CoarseTracker(wG[0], hG[0]);
	coarseInitializer = new CoarseInitializer(wG[0], hG[0]);
	pixelSelector = new PixelSelector(wG[0], hG[0]);

    // 制作好运动先验估计的扰动模版
    make_motion_prediction_based_on_assumption();

	statistics_lastNumOptIts=0;
	statistics_numDroppedPoints=0;
	statistics_numActivatedPoints=0;
	statistics_numCreatedPoints=0;
	statistics_numForceDroppedResBwd = 0;
	statistics_numForceDroppedResFwd = 0;
	statistics_numMargResFwd = 0;
	statistics_numMargResBwd = 0;

	lastCoarseRMSE.setConstant(100);

	currentMinActDist=2;
	initialized=false;

	ef = new EnergyFunctional();
//	ef->red = &this->treadReduce;

	isLost=false;
	initFailed=false;

	needNewKFAfter = -1;

	linearizeOperation=true;
	runMapping=true;
	mappingThread = boost::thread(&FullSystem::mappingLoop, this);
	lastRefStopID=0;

	minIdJetVisDebug = -1;
	maxIdJetVisDebug = -1;
	minIdJetVisTracker = -1;
	maxIdJetVisTracker = -1;
}

FullSystem::~FullSystem()
{
	blockUntilMappingIsFinished();

	if(setting_logStuff)
	{
		calibLog->close(); delete calibLog;
		numsLog->close(); delete numsLog;
		coarseTrackingLog->close(); delete coarseTrackingLog;
		//errorsLog->close(); delete errorsLog;
		eigenAllLog->close(); delete eigenAllLog;
		eigenPLog->close(); delete eigenPLog;
		eigenALog->close(); delete eigenALog;
		DiagonalLog->close(); delete DiagonalLog;
		variancesLog->close(); delete variancesLog;
		nullspacesLog->close(); delete nullspacesLog;
	}

	delete[] selectionMap;

	for(FrameShell* s : allFrameHistory)
		delete s;
	for(FrameHessian* fh : unmappedTrackedFrames)
		delete fh;

	delete coarseDistanceMap;
	delete coarseTracker;
	delete coarseTracker_forNewKF;
	delete coarseInitializer;
	delete pixelSelector;
	delete ef;
}


void FullSystem::setGammaFunction(float* BInv)
{
	if(BInv==0) return;

	// copy BInv.
	memcpy(Hcalib.Binv, BInv, sizeof(float)*256);

	// invert.
	for(int i=1;i<255;i++) {
		// find val, such that Binv[val] = i.
		// I dont care about speed for this, so do it the stupid way.
		for(int s=1;s<255;s++) {
			if(BInv[s] <= i && BInv[s+1] >= i) {
				Hcalib.B[i] = s+(i - BInv[s]) / (BInv[s+1]-BInv[s]);
				break;
			}
		}
	}
	Hcalib.B[0] = 0;
	Hcalib.B[255] = 255;
}

//void FullSystem::printResult(std::string file)
//{
//	boost::unique_lock<boost::mutex> lock(trackMutex);
//	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
//
//	std::ofstream myfile;
//	myfile.open (file.c_str());
//	myfile << std::setprecision(15);
//
//	for(FrameShell* s : allFrameHistory)
//	{
//		if(!s->poseValid) continue;
//
//		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) continue;
//
//		myfile << s->timestamp <<
//			" " << s->camToWorld.translation().transpose()<<
//			" " << s->camToWorld.so3().unit_quaternion().x()<<
//			" " << s->camToWorld.so3().unit_quaternion().y()<<
//			" " << s->camToWorld.so3().unit_quaternion().z()<<
//			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
//	}
//	myfile.close();
//}
void FullSystem::printResult(std::string file)
{
	boost::unique_lock<boost::mutex> lock(trackMutex);
	boost::unique_lock<boost::mutex> crlock(shellPoseMutex);

	std::ofstream myfile;
	myfile.open (file.c_str());
	myfile << std::setprecision(15);
	int i = 0;

//	Eigen::Matrix<double,3,3> last_R = (*(allFrameHistory.begin()))->camToWorld.so3().matrix();
//	Eigen::Matrix<double,3,1> last_T = (*(allFrameHistory.begin()))->camToWorld.translation().transpose();

	for(FrameShell* s : allFrameHistory) {

        if(!s->poseValid) {
            printf("............not poseValid.........\n");
            continue;
        }
		if(setting_onlyLogKFPoses && s->marginalizedAt == s->id) {
            printf("...........s->marginalizedAt s->id....\n");
            continue;
        }
		const Eigen::Matrix<double,3,3> R = s->camToWorld.so3().matrix();
		const Eigen::Matrix<double,3,1> T = s->camToWorld.translation().transpose();

//		last_R = R;
//		last_T = T;

		myfile<< R(0,0) <<" "<<R(0,1)<<" "<<R(0,2)<<" "<<T(0,0)<<" "<<
			  R(1,0) <<" "<<R(1,1)<<" "<<R(1,2)<<" "<<T(1,0)<<" "<<
			  R(2,0) <<" "<<R(2,1)<<" "<<R(2,2)<<" "<<T(2,0)<<"\n";

//		myfile << s->timestamp <<
//			" " << s->camToWorld.translation().transpose()<<
//			" " << s->camToWorld.so3().unit_quaternion().x()<<
//			" " << s->camToWorld.so3().unit_quaternion().y()<<
//			" " << s->camToWorld.so3().unit_quaternion().z()<<
//			" " << s->camToWorld.so3().unit_quaternion().w() << "\n";
		i++;
	}
	myfile.close();
}

// 因为BA优化的是关键帧，所以在coarse tracking里都是计算ref -> fh
Vec4 FullSystem::track(FrameHessian* fh)
{
	assert(allFrameHistory.size() > 0);
	// set pose initialization.

    for(IOWrap::Output3DWrapper* ow : outputWrapper)
        ow->pushLiveFrame(fh);

    // 跟踪的是lastRef -> fh
    FrameHessian* lastF = coarseTracker->lastRef;
    AffLight aff_last_2_l = AffLight(0,0);
    std::vector<SE3,Eigen::aligned_allocator<SE3>> lastF_2_fh_tries_1;

    SE3 slast_2_sprelast;
    SE3 lastF_2_slast;
    SE3 fh_2_slast;
    SE3 motion_pre;
    if(allFrameHistory.size() == 2) {
        // 只使用金字塔0层
        // fh只是初始化了一下，没有跟第1帧进行关联匹配
        // initializer->first_frame 送入 fullsystem.frameHessians
        printf(".....initializer:  allFrameHistory id[%d] id[%d]........\n",
               allFrameHistory[allFrameHistory.size() - 1]->id,
               allFrameHistory[allFrameHistory.size() - 2]->id);
        initializeFromInitializer(fh);

        printf("...........allFrameHistory.size() == 2...............\n");

        lastF_2_fh_tries_1.push_back(SE3());
        motion_pre = SE3();

        coarseTracker->makeK(&Hcalib);
        // 这里开始初始化逆深度，set coarseTracker reference for 1st Frame
        // 将FullSystem里的frameHessians扔给跟踪器
        // 这里做好了lastF，为下文的tracking做预备
        coarseTracker->setCoarseTrackingRef(this->frameHessians, this->Hcalib, true);
        lastF = coarseTracker->lastRef;
        printf("...........lastF = ..................\n");
    } else {
// last two frames
        // slast:  shell of last frame
        // sprelast: shell of pre-last frame
        //  ........[sprelast]-[slast]-[lastF]
        FrameShell *slast = allFrameHistory[allFrameHistory.size() - 2];
        FrameShell *sprelast = allFrameHistory[allFrameHistory.size() - 3];

        {    // lock on global pose consistency! 防止其他线程对全局位姿的修改
            boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
            slast_2_sprelast = sprelast->camToWorld.inverse() * slast->camToWorld;
            lastF_2_slast = slast->camToWorld.inverse() * lastF->shell->camToWorld;
            aff_last_2_l = slast->aff_g2l;
        }

        // 这个就是匀速运动假设，已知的是last到pre-last的位姿变化，取fh到last的位姿变化跟它相同
        // 这里没有牵涉到lastF，因为lastF是参考帧，而我们最终要给出的就是参考帧到当前帧的运动先验估计
        SE3 fh_2_slast = slast_2_sprelast;// assumed to be the same as fh_2_slast.
        //于是我们预估出来了上一帧到当前帧的运动先验，将之与参考帧到上一帧（这个是已知的，因为参考帧和上一帧的世界位姿
        // 都是已知的）的运动变化组合，就得到当前帧到参考帧的运动先验估计，扔给下一步处理。
        motion_pre = fh_2_slast.inverse() * lastF_2_slast;
        //  ........[sprelast]-[slast]-[lastF]
        // get last delta-movement.
        // 对从lastF到fh的SE3，做一个先验，首选匀速运动假设
        lastF_2_fh_tries_1.push_back(fh_2_slast.inverse() * lastF_2_slast);    // assume constant motion.
        lastF_2_fh_tries_1.push_back(
                fh_2_slast.inverse() * fh_2_slast.inverse()
                * lastF_2_slast);    // assume double motion (frame skipped)
        lastF_2_fh_tries_1.push_back(SE3::exp(fh_2_slast.log() * 0.5).inverse()
                                     * lastF_2_slast); // assume half motion.
        lastF_2_fh_tries_1.push_back(lastF_2_slast); // assume zero motion.
        lastF_2_fh_tries_1.push_back(SE3()); // assume zero motion FROM KF.
        if (!slast->poseValid || !sprelast->poseValid || !lastF->shell->poseValid) {
            lastF_2_fh_tries_1.clear();
            lastF_2_fh_tries_1.push_back(SE3());
        }
    }

    Vec3 flowVecs = Vec3(100,100,100);
    SE3 lastF_2_fh = SE3();
    AffLight aff_g2l = AffLight(0,0);

    // as long as maxResForImmediateAccept is not reached, I'll continue through the options.
    // I'll keep track of the so-far best achieved residual for each level in achievedRes.
    // If on a coarse level, tracking is WORSE than achievedRes, we will not continue to save time.

    Vec5 achievedRes = Vec5::Constant(NAN);
    bool haveOneGood = false;
    int tryIterations=0;

    for (unsigned int i=0;i< lastF_2_fh_tries_1.size() + this -> lastF_2_fh_tries.size();i++) {
        AffLight aff_g2l_this = aff_last_2_l;
        SE3 lastF_2_fh_this;
        if (i < lastF_2_fh_tries_1.size())
            lastF_2_fh_this = lastF_2_fh_tries_1[i];
        else
            lastF_2_fh_this = motion_pre * lastF_2_fh_tries[i - lastF_2_fh_tries_1.size()];
// 		LOG(INFO)<<"lastF_2_fh_this: "<<lastF_2_fh_this.translation().transpose();
// 		LOG(INFO)<<"aff_g2l_this: "<<aff_g2l_this.vec().transpose();
// 这里achievedRes不是更新量，只是判断迭代是否有效的条件，要到下面才会判断是否更新此结果
// 这里的结果是存于coarseTracker->lastResiduals和coarseTracker->lastFlowIndicators
        bool trackingIsGood = coarseTracker->track(
                fh, lastF_2_fh_this, aff_g2l_this,
                pyrLevelsUsed - 1,
                achievedRes);    // in each level has to be at least as good as the last try.
        tryIterations++;
        if (i >= 1)
            printf("..............constant motion failed......\n");
        // do we have a new winner?
        // 这里是得到了更小的残差，在下面haveOneGood为
        if(trackingIsGood && std::isfinite((float)coarseTracker->lastResiduals[0]) &&
                !(coarseTracker->lastResiduals[0] >= achievedRes[0])) {
            // 只有上一次的achievedRes[0] >= lastCoarseRMSE[0]*setting_reTrackThreshold，才有可能
            // 用新的运动先验再次计算得到achievedRes
            if (std::isfinite((float)achievedRes[0]))
                printf("take over. minRes %f -> %f!\n", achievedRes[0], coarseTracker->lastResiduals[0]);
            flowVecs = coarseTracker->lastFlowIndicators;
            aff_g2l = aff_g2l_this;
            lastF_2_fh = lastF_2_fh_this;
            haveOneGood = true;
        }
        // take over achieved res (always).
        if(haveOneGood) {
            for(int i=0;i<5;i++) {
                // take over if achievedRes is either bigger or NAN.
                // 如果得到了更小的残差，就更新achievedRes
                if(!std::isfinite((float)achievedRes[i]))
                    achievedRes[i] = coarseTracker->lastResiduals[i];
                else if (achievedRes[i] > coarseTracker->lastResiduals[i]) {
                    // 一般情况下不会进入这里，即lastResiduals只会设置一轮
                    // 只有上一次的achievedRes[0] >= lastCoarseRMSE[0]*setting_reTrackThreshold，才有可能
                    // 用新的运动先验再次计算得到achievedRes
                    printf(" better res [%lf] -> [%lf], lvl [%d]\n",
                           achievedRes[i], coarseTracker->lastResiduals[i], i);
//                    assert(false);
                    achievedRes[i] = coarseTracker->lastResiduals[i];
                }
            }
        }
        if(haveOneGood &&  achievedRes[0] < lastCoarseRMSE[0]*setting_reTrackThreshold)
            break;
    }

    if(!haveOneGood) {
        printf("BIG ERROR! tracking failed entirely. ");
        printf("Take predictred pose and hope we may somehow recover.\n");
        flowVecs = Vec3(0,0,0);
        aff_g2l = aff_last_2_l;
        lastF_2_fh = lastF_2_fh_tries[0];
    }

    lastCoarseRMSE = achievedRes;

    // no lock required, as fh is not used anywhere yet.
    fh->shell->camToTrackingRef = lastF_2_fh.inverse();  // 这里将求解出的位姿赋予camToTrackingRef
    fh->shell->trackingRef = lastF->shell;
    fh->shell->aff_g2l = aff_g2l;
    fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

//    std::cout << "coarseTracker with ref frame: " << coarseTracker->refFrameID << std::endl;
    if(coarseTracker->firstCoarseRMSE < 0) {
//        std::cout << "firstCoarseRMSE is initialized as " << achievedRes[0] << std::endl;
        coarseTracker->firstCoarseRMSE = achievedRes[0];
    } else {
//        std::cout << "...firstCoarseRMSE: " << coarseTracker->firstCoarseRMSE
//        << "...current RMSE[0]: " << achievedRes[0] << std::endl;
    }

    if(!setting_debugout_runquiet)
        printf("Coarse Tracker tracked ab = %f %f (exp %f). Res %f!\n", aff_g2l.a, aff_g2l.b, fh->ab_exposure, achievedRes[0]);

    if(setting_logStuff) {
        (*coarseTrackingLog) << std::setprecision(16)
                             << fh->shell->id << " "
                             << fh->shell->timestamp << " "
                             << fh->ab_exposure << " "
                             << fh->shell->camToWorld.log().transpose() << " "
                             << aff_g2l.a << " "
                             << aff_g2l.b << " "
                             << achievedRes[0] << " "
                             << tryIterations << "\n";
    }

    return Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
}

// process nonkey frame to refine key frame idepth
// 检查未成熟点，增加成熟点
// 它主要是基于跟踪的结果来收敛深度不确定的点
// 面向的是深度不确定的点，因此不会破坏优化（PBA），因为优化是针对窗口内的已经确定的位姿和三维点进行微调
// DSO用的是位姿圖優化，在優化過程中，三維點只是作爲約束
// maturelize_newframe
void FullSystem::maturelize_window_based_on_newframe(FrameHessian *fh) {
	boost::unique_lock<boost::mutex> lock(mapMutex);
	// new idepth after refinement
	float idepth_min_update = 0;
	float idepth_max_update = 0;

	Mat33f K = Mat33f::Identity();
	K(0, 0) = Hcalib.fxl();
	K(1, 1) = Hcalib.fyl();
	K(0, 2) = Hcalib.cxl();
	K(1, 2) = Hcalib.cyl();

	Mat33f Ki = K.inverse();

    //遍历 窗口， 遍历窗口内每一帧的每一个未成熟点
    // 这时候新帧还没在窗口里
	for (FrameHessian *host : frameHessians) {        // go through all active frames
//		number++;
		int trace_total = 0, trace_good = 0, trace_oob = 0, trace_out = 0, trace_skip = 0, trace_badcondition = 0, trace_uninitialized = 0;

		// trans from reference keyframe to newest frame
		SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
		// KRK-1
		Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
		// KRi
		Mat33f KRi = K * hostToNew.rotationMatrix().inverse().cast<float>();
		// Kt
		Vec3f Kt = K * hostToNew.translation().cast<float>();
		// t
		Vec3f t = hostToNew.translation().cast<float>();

		//aff
		Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure,
                                                host->aff_g2l(), fh->aff_g2l()).cast<float>();

		for (ImmaturePoint *ph : host->immaturePoints) {
			// do temperol stereo match
            // 收割成熟点
            // temporal match. trace 3d immature point on fh
			ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt,
                                                            aff, &Hcalib, false);
			if (phTrackStatus == ImmaturePointStatus::IPS_GOOD) {
				ImmaturePoint *phNonKey = new ImmaturePoint(ph->lastTraceUV(0), ph->lastTraceUV(1), fh, &Hcalib);

				// project onto newest frame
				Vec3f ptpMin = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_min) + Kt;
				float idepth_min_project = 1.0f / ptpMin[2];
				Vec3f ptpMax = KRKi * (Vec3f(ph->u, ph->v, 1) / ph->idepth_max) + Kt;
				float idepth_max_project = 1.0f / ptpMax[2];

				phNonKey->idepth_min = idepth_min_project;
				phNonKey->idepth_max = idepth_max_project;
				phNonKey->u_stereo = phNonKey->u;
				phNonKey->v_stereo = phNonKey->v;
				phNonKey->idepth_min_stereo = phNonKey->idepth_min;
				phNonKey->idepth_max_stereo = phNonKey->idepth_max;

				// do static stereo match from left image to right
				ImmaturePointStatus phNonKeyStereoStatus = phNonKey->traceStereo(fh->frame_right,
                                                                                 &Hcalib, 1);
				if(phNonKeyStereoStatus == ImmaturePointStatus::IPS_GOOD) {
				    ImmaturePoint* phNonKeyRight = new ImmaturePoint(phNonKey->lastTraceUV(0),
                                                                     phNonKey->lastTraceUV(1),
                                                                     fh->frame_right, &Hcalib );
				    phNonKeyRight->u_stereo = phNonKeyRight->u;
				    phNonKeyRight->v_stereo = phNonKeyRight->v;
				    phNonKeyRight->idepth_min_stereo = phNonKey->idepth_min;
				    phNonKeyRight->idepth_max_stereo = phNonKey->idepth_max;

				    // do static stereo match from right image to left
				    ImmaturePointStatus  phNonKeyRightStereoStatus = phNonKeyRight->traceStereo(fh, &Hcalib, 0);

				    // change of u after two different stereo match
				    float u_stereo_delta = abs(phNonKey->u_stereo - phNonKeyRight->lastTraceUV(0));
				    float disparity = phNonKey->u_stereo - phNonKey->lastTraceUV[0];

				    // free to debug the threshold
				    if(u_stereo_delta > 1 && disparity < 10) {
                        ph->lastTraceStatus = ImmaturePointStatus :: IPS_OUTLIER;
                        continue;
				    } else {
                        // project back
                        Vec3f pinverse_min = KRi *
                                             (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) /
                                              phNonKey->idepth_min_stereo - t);
                        idepth_min_update = 1.0f / pinverse_min(2);

                        Vec3f pinverse_max = KRi *
                                             (Ki * Vec3f(phNonKey->u_stereo, phNonKey->v_stereo, 1) /
                                              phNonKey->idepth_max_stereo - t);
                        idepth_max_update = 1.0f / pinverse_max(2);

                        ph->idepth_min = idepth_min_update;
                        ph->idepth_max = idepth_max_update;

                        delete phNonKey;
                        delete phNonKeyRight;
				    }
				} else {
				    delete phNonKey;
				    continue;
				}
			}
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_GOOD) trace_good++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OOB) trace_oob++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_OUTLIER) trace_out++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
			if (ph->lastTraceStatus == ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
			trace_total++;
		}
	}
}


//process keyframe
#if 0
void FullSystem::traceNewCoarseKey(FrameHessian* fh)
{
		boost::unique_lock<boost::mutex> lock(mapMutex);

		int trace_total=0, trace_good=0, trace_oob=0, trace_out=0, trace_skip=0, trace_badcondition=0, trace_uninitialized=0;

		Mat33f K = Mat33f::Identity();
		K(0,0) = Hcalib.fxl();
		K(1,1) = Hcalib.fyl();
		K(0,2) = Hcalib.cxl();
		K(1,2) = Hcalib.cyl();

		for(FrameHessian* host : frameHessians) {		// go through all active frames
			// trans from reference key frame to the newest one
			SE3 hostToNew = fh->PRE_worldToCam * host->PRE_camToWorld;
			//KRK-1
			Mat33f KRKi = K * hostToNew.rotationMatrix().cast<float>() * K.inverse();
			//Kt
			Vec3f Kt = K * hostToNew.translation().cast<float>();

			Vec2f aff = AffLight::fromToVecExposure(host->ab_exposure, fh->ab_exposure, host->aff_g2l(), fh->aff_g2l()).cast<float>();

			for(ImmaturePoint* ph : host->immaturePoints) {
				ImmaturePointStatus phTrackStatus = ph->traceOn(fh, KRKi, Kt, aff, &Hcalib, false );
// 				if(ph->u==145&&ph->v==21){
// 				    LOG(INFO)<<"trace depthmax: "<<ph->idepth_max<<" min: "<<ph->idepth_min;
// 				}
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_GOOD) trace_good++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_BADCONDITION) trace_badcondition++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OOB) trace_oob++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_OUTLIER) trace_out++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_SKIPPED) trace_skip++;
				if(ph->lastTraceStatus==ImmaturePointStatus::IPS_UNINITIALIZED) trace_uninitialized++;
				trace_total++;
			}
		}

}
#endif


void FullSystem::activatePointsMT_Reductor(
		std::vector<PointHessian*>* optimized,
		std::vector<ImmaturePoint*>* toOptimize,
        int min, int max,
		Vec10* stats, int tid)
{
	ImmaturePointTemporaryResidual* tr = new ImmaturePointTemporaryResidual[frameHessians.size()];
    // min === 0, max === toOptimize.size()
	for(int k = min;k < max;k++) {
        // 將點從未成熟變成活躍
		(*optimized)[k] = optimizeImmaturePoint((*toOptimize)[k],1,tr);
	}
	delete[] tr;
}

void FullSystem::activatePointsMT()
{
	if(ef->nPoints < setting_desiredPointDensity*0.66)
		currentMinActDist -= 0.8;
	if(ef->nPoints < setting_desiredPointDensity*0.8)
		currentMinActDist -= 0.5;
	else if(ef->nPoints < setting_desiredPointDensity*0.9)
		currentMinActDist -= 0.2;
	else if(ef->nPoints < setting_desiredPointDensity)
		currentMinActDist -= 0.1;

	if(ef->nPoints > setting_desiredPointDensity*1.5)
		currentMinActDist += 0.8;
	if(ef->nPoints > setting_desiredPointDensity*1.3)
		currentMinActDist += 0.5;
	if(ef->nPoints > setting_desiredPointDensity*1.15)
		currentMinActDist += 0.2;
	if(ef->nPoints > setting_desiredPointDensity)
		currentMinActDist += 0.1;

	if(currentMinActDist < 0) currentMinActDist = 0;
	if(currentMinActDist > 4) currentMinActDist = 4;

    if(!setting_debugout_runquiet)
        printf("SPARSITY:  MinActDist %f (need %d points, have %d points)!\n",
                currentMinActDist, (int)(setting_desiredPointDensity), ef->nPoints);

	FrameHessian* newestHs = frameHessians.back();

	// make dist map.
	coarseDistanceMap->makeK(&Hcalib);
	coarseDistanceMap->makeDistanceMap(frameHessians, newestHs);

	//coarseTracker->debugPlotDistMap("distMap");

	std::vector<ImmaturePoint*> toOptimize; toOptimize.reserve(20000);

    //遍历窗口内所有未成熟点
	for(FrameHessian* host : frameHessians) {		// go through all active frames
		if(host == newestHs) continue;

		SE3 fhToNew = newestHs->PRE_worldToCam * host->PRE_camToWorld;
		Mat33f KRKi = (coarseDistanceMap->K[1] * fhToNew.rotationMatrix().cast<float>() * coarseDistanceMap->Ki[0]);
		Vec3f Kt = (coarseDistanceMap->K[1] * fhToNew.translation().cast<float>());

		for(unsigned int i=0;i<host->immaturePoints.size();i+=1) {
			ImmaturePoint* ph = host->immaturePoints[i];
			ph->idxInImmaturePoints = i;

			// delete points that have never been traced successfully, or that are outlier on the last trace.
            // 清理掉idepth_max爲無限大的點，還有被判定爲外點的點
			if(!std::isfinite(ph->idepth_max) || ph->lastTraceStatus == IPS_OUTLIER) {
//				immature_invalid_deleted++;
				// remove point.
				delete ph;
				host->immaturePoints[i]=0;
				continue;
			}

            // 遍历这些未成熟点, 後面會用idepth的均值作爲逆深度的初值
			// can activate only if this is true.
			bool canActivate = (ph->lastTraceStatus == IPS_GOOD
					|| ph->lastTraceStatus == IPS_SKIPPED
					|| ph->lastTraceStatus == IPS_BADCONDITION
					|| ph->lastTraceStatus == IPS_OOB )
							&& ph->lastTracePixelInterval < 8
							&& ph->quality > setting_minTraceQuality
							&& (ph->idepth_max+ph->idepth_min) > 0;

			// if I cannot activate the point, skip it. Maybe also delete it.
            // 已经完全没救了，看看宿主(的位姿）是否已经停止优化更新，如果是的，就删除该点
            // 已经被边缘化的帧，应该要退出窗口了
			if(!canActivate) {
				// if point will be out afterwards, delete it instead.
                // 如果它的宿主帧已经是边缘化的，那么直接删除
				if(ph->host->flaggedForMarginalization || ph->lastTraceStatus == IPS_OOB) {
//					immature_notReady_deleted++;
					delete ph;
					host->immaturePoints[i]=0;
				}
//				immature_notReady_skipped++;
				continue;
			}

			// see if we need to activate point due to distance map.
            // 用的是逆深度的均值（在高斯分布下，即MAP值）
            // (0.5f*(ph->idepth_max+ph->idepth_min)
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1)
                        + Kt*(0.5f*(ph->idepth_max+ph->idepth_min));
            //上面这句warp到newestHs，这个是窗口内的最新帧（应该也就是跟踪器里的参考帧）
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;

			if((u > 0 && v > 0 && u < wG[1] && v < hG[1])) {
				float dist = coarseDistanceMap->fwdWarpedIDDistFinal[u+wG[1]*v] + (ptp[0]-floorf((float)(ptp[0])));
				if(dist>=currentMinActDist* ph->my_type) {
					coarseDistanceMap->addIntoDistFinal(u,v);
					toOptimize.push_back(ph);
				}
			} else {
				delete ph;
				host->immaturePoints[i]=0;
			}
		}
	}
// 	LOG(INFO)<<"toOptimize.size(): "<<toOptimize.size();
//	printf("ACTIVATE: %d. (del %d, notReady %d, marg %d, good %d, marg-skip %d)\n",
//			(int)toOptimize.size(), immature_deleted, immature_notReady, immature_needMarg, immature_want, immature_margskip);

	std::vector<PointHessian*> optimized; optimized.resize(toOptimize.size());

    // 將所有的未成熟點變成活躍點
	if(multiThreading)
		treadReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this,
                                       &optimized, &toOptimize, _1, _2, _3, _4),
                           0, toOptimize.size(), 50);
	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);

    int insert_count = 0;
// 	LOG(INFO)<<"toOptimize.size(): "<<toOptimize.size();
	for(unsigned k = 0;k < toOptimize.size();k++) {
		PointHessian* newpoint = optimized[k];
		ImmaturePoint* ph = toOptimize[k];

		if(newpoint != 0 && newpoint != (PointHessian*)((long)(-1))) {
			newpoint->host->immaturePoints[ph->idxInImmaturePoints]=0;
			newpoint->host->pointHessians.push_back(newpoint);
			ef->insertPoint(newpoint);
            insert_count++;
			for(PointFrameResidual* r : newpoint->residuals)
				ef->insertResidual(r);
			assert(newpoint->efPoint != 0);
			delete ph;
		} else if(newpoint == (PointHessian*)((long)(-1)) || ph->lastTraceStatus==IPS_OOB) {
			delete ph;
			ph->host->immaturePoints[ph->idxInImmaturePoints]=0;
		} else {
			assert(newpoint == 0 || newpoint == (PointHessian*)((long)(-1)));
		}
	}

    std::cout << "insert points: " << insert_count << std::endl;

    // 整理fh->immaturePoints隊列
	for(FrameHessian* host : frameHessians) {
		for(int i=0;i<(int)host->immaturePoints.size();i++) {
			if(host->immaturePoints[i]==0) {
				host->immaturePoints[i] = host->immaturePoints.back();
				host->immaturePoints.pop_back();
				i--;
			}
		}
	}
}

void FullSystem::activatePointsOldFirst()
{
	assert(false);
}

void FullSystem::flagPointsForRemoval()
{
	assert(EFIndicesValid);

	std::vector<FrameHessian*> fhsToKeepPoints;
	std::vector<FrameHessian*> fhsToMargPoints;

//	if(setting_margPointVisWindow>0)
    {
		for(int i=((int)frameHessians.size())-1;i>=0 && i >= ((int)frameHessians.size());i--)
			if(!frameHessians[i]->flaggedForMarginalization) fhsToKeepPoints.push_back(frameHessians[i]);

		for(int i=0; i< (int)frameHessians.size();i++)
			if(frameHessians[i]->flaggedForMarginalization) fhsToMargPoints.push_back(frameHessians[i]);
	}

	//ef->setAdjointsF();
	//ef->setDeltaF(&Hcalib);
	int flag_oob=0, flag_in=0, flag_inin=0, flag_nores=0;

	for(FrameHessian* host : frameHessians) {		// go through all active frames
		for(unsigned int i=0;i<host->pointHessians.size();i++) {
			PointHessian* ph = host->pointHessians[i];
			if(ph==0) continue;

            // 如果逆深度小于0，或者已经没有观测（残差为0）, 则需要丢弃该点
			if(ph->idepth_scaled < 0 || ph->residuals.size()==0) {
				host->pointHessiansOut.push_back(ph);
				ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
				host->pointHessians[i]=0;
				flag_nores++;
               //
			} else if(ph->isOOB(fhsToKeepPoints, fhsToMargPoints)
                    || host->flaggedForMarginalization) {
				flag_oob++;
				if(ph->isInlierNew()) {
					flag_in++;
					int ngoodRes=0;
					for(PointFrameResidual* r : ph->residuals) {
// 						if(r->efResidual->idxInAll==0)continue;
						r->resetOOB();
						if(r->stereoResidualFlag == true)
							r->linearizeStereo(&Hcalib);
						else
							r->linearize(&Hcalib);
// 						r->linearize(&Hcalib);
						r->efResidual->isLinearized = false;
                        // 计算残差值
						r->applyRes();
						if(r->efResidual->isActive()) {
                            //! 固定线性化
							r->efResidual->fixLinearizationF(ef);
							ngoodRes++;
						}
					}
					if(ph->idepth_hessian > setting_minIdepthH_marg) {
						flag_inin++;
						ph->efPoint->stateFlag = EFPointStatus::PS_MARGINALIZE;
						host->pointHessiansMarginalized.push_back(ph);
					} else {
						ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
						host->pointHessiansOut.push_back(ph);
					}
				} else {
					host->pointHessiansOut.push_back(ph);
					ph->efPoint->stateFlag = EFPointStatus::PS_DROP;
					//printf("drop point in frame %d (%d goodRes, %d activeRes)\n", ph->host->idx, ph->numGoodResiduals, (int)ph->residuals.size());
				}
				host->pointHessians[i]=0;
			}
		}
		for(int i=0;i<(int)host->pointHessians.size();i++) {
			if(host->pointHessians[i]==0) {
				host->pointHessians[i] = host->pointHessians.back();
				host->pointHessians.pop_back();
				i--;
			}
		}
	}
}

void FullSystem::addActiveFrame( ImageAndExposure* image, ImageAndExposure* image_right, int id ) {
//	LOG(INFO)<<"id: "<<id;
    if (isLost) return;
    boost::unique_lock<boost::mutex> lock(trackMutex);

    // =========================== add into allFrameHistory =========================
    FrameShell *shell = new FrameShell();
    shell->camToWorld = SE3();        // no lock required, as fh is not used anywhere yet.
    shell->aff_g2l = AffLight(0, 0);
    shell->marginalizedAt = shell->id = allFrameHistory.size();
    shell->timestamp = image->timestamp;
    shell->incoming_id = id;
    allFrameHistory.push_back(shell);

    FrameHessian *fh = new FrameHessian();
    fh->frame_right = new FrameHessian();
    fh->shell = shell;
    fh->frame_right->shell = shell;

    // =========================== make Images / derivatives etc. =========================
    fh->ab_exposure = image->exposure_time;
    fh->makeImages(image->image, &Hcalib);
    fh->frame_right->ab_exposure = image_right->exposure_time;
    fh->frame_right->makeImages(image_right->image, &Hcalib);

    if (!initialized) {
        // 这时候只有第0帧的shell在allFramehistory里
        printf("...............init stereo..........\n");
        // use initializer!
        if (coarseInitializer->frameID < 0) {    // first frame set. fh is kept by coarseInitializer.
            coarseInitializer->setFirstStereo(&Hcalib, fh);
            initialized = true;
        }
        return;
    } else {    // do front-end operation.
        // =========================== SWAP tracking reference?. =========================
        if (coarseTracker_forNewKF->refFrameID > coarseTracker->refFrameID) {
//            printf(".............SWAP tracking reference............\n");
//            printf("forNewKF:[%d]   coarseTracker:[%d]\n", coarseTracker_forNewKF->refFrameID,
//                   coarseTracker->refFrameID);
            boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
            CoarseTracker *tmp = coarseTracker;
            coarseTracker = coarseTracker_forNewKF;
            coarseTracker_forNewKF = tmp;
        }
//        Vec4(achievedRes[0], flowVecs[0], flowVecs[1], flowVecs[2]);
// 根据成熟点制作金字塔，计算残差，计算位姿
        auto t1 = std::chrono::steady_clock::now();
        Vec4 tres = this -> track(fh);
        auto t2 = std::chrono::steady_clock::now();
        trackNewCoarse_total_time +=
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() *
                1000;
// 		LOG(INFO)<<"track done";

        if (!std::isfinite((double) tres[0]) || !std::isfinite((double) tres[1])
            || !std::isfinite((double) tres[2])
            || !std::isfinite((double) tres[3])) {
            printf("...........................Initial Tracking failed: LOST!\n");
            isLost = true;
            return;
        }
        bool needToMakeKF = false;
        if (setting_keyframesPerSecond > 0) {   // 固定每秒N帧关键帧
            needToMakeKF = allFrameHistory.size() == 1 ||
                           (fh->shell->timestamp - allKeyFramesHistory.back()->timestamp) >
                           0.95f / setting_keyframesPerSecond;
        } else {
            Vec2 refToFh = AffLight::fromToVecExposure(coarseTracker->lastRef->ab_exposure, fh->ab_exposure,
                                                       coarseTracker->lastRef_aff_g2l, fh->shell->aff_g2l);

            // BRIGHTNESS CHECK
            double delta = setting_kfGlobalWeight * setting_maxShiftWeightT
                           * sqrtf((double) tres[1]) / (wG[0] + hG[0]) +
                           setting_kfGlobalWeight * setting_maxShiftWeightR
                           * sqrtf((double) tres[2]) / (wG[0] + hG[0]) +
                           setting_kfGlobalWeight * setting_maxShiftWeightRT
                           * sqrtf((double) tres[3]) / (wG[0] + hG[0]) +
                           setting_kfGlobalWeight * setting_maxAffineWeight
                           * fabs(logf((float) refToFh[0]));
//            std::cout << "coarseTraker->firstCoarseRMSE " << coarseTracker->firstCoarseRMSE << std::endl;
//            std::cout << "tres[0] " << tres[0] << std::endl;
// firstCoarseRMSE是跟踪器得到新的参考帧时，第一次计算出的RMSE，如果使用该参考帧得到的后续帧的RMSE突然变大，
// 即大于2倍的第一次的值，我们就认为这个参考帧已经不太好用了，要触发关键帧的管理
            needToMakeKF =
                    allFrameHistory.size() == 1 ||
                    delta > 1.0 ||
                    2 * coarseTracker->firstCoarseRMSE < tres[0];

//            std::cout << "delta: " << delta << std::endl;
// 			LOG(INFO)<<"delta: "<<delta;
// 			LOG(INFO)<<"tres: "<<tres.transpose()<<" refToFh[0]: "<<refToFh[0];
// 			std::cout <<"tres: "<<tres.transpose()<<" refToFh[0]: "<<refToFh[0] << std::endl;
            if (needToMakeKF) {
                num_kf++;
//                std::cout << "need to make new key frame" << std::endl;
            }
        }

// 		LOG(INFO)<<"needToMakeKF: "<<(int)needToMakeKF;
// 		LOG(INFO)<<"coarseTracker->firstCoarseRMSE: "<<coarseTracker->firstCoarseRMSE<<" tres[0]:"<<tres[0];
// 		LOG(INFO)<<"allFrameHistory.size(): "<<allFrameHistory.size();

        for (IOWrap::Output3DWrapper *ow: outputWrapper)
            ow->publishCamPose(fh->shell, &Hcalib);

        lock.unlock();
        // 粗的跟踪结束，释放跟踪锁。接下来要处理跟踪以外的事了。
        // 成熟未成熟点，增加未成熟点（根据selection_map）
        auto t3 = std::chrono::steady_clock::now();
        deliverTrackedFrame(fh, needToMakeKF);
        auto t4 = std::chrono::steady_clock::now();
        deliverTrackedFrame_total_time +=
                std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count() *
                1000;
// 		LOG(INFO)<<"fh->worldToCam_evalPT: "<<allFrameHistory[allFrameHistory.size()-1]->camToWorld.translation().transpose();
// 		LOG(INFO)<<"fh->shell->aff_g2l: "<<allFrameHistory[allFrameHistory.size()-1]->aff_g2l.vec().transpose();
// 		LOG(INFO)<<"fh->shell->camToTrackingRef: "<<fh->shell->camToTrackingRef.translation().transpose();
// 		LOG(INFO)<<"fh->shell->trackingRef->camToWorld : "<<fh->shell->trackingRef->camToWorld.translation().transpose();
// 		exit(1);
        return;

    }
}
void FullSystem::deliverTrackedFrame(FrameHessian* fh, bool needKF)
{
//    goStepByStep = true;
    // linearizeOperation: playbackspeed == 0, i.e, no time stamp
    // 即没有时间戳。我们现在就用这个模式。
	if(linearizeOperation) {
        if (goStepByStep)
            if(lastRefStopID != coarseTracker->refFrameID) {
                MinimalImageF3 img(wG[0], hG[0], fh->dI);
                IOWrap::displayImage("frameToTrack", &img);

                MinimalImageF3 img_ref(wG[0], hG[0], coarseTracker->lastRef->dI);
                IOWrap::displayImage("frameForTrack", &img_ref);
                while(true) {
                    char k=IOWrap::waitKey(0);
                    if(k==' ') break;
                    handleKey( k );
                }
                lastRefStopID = coarseTracker->refFrameID;
            } else {
                MinimalImageF3 img(wG[0], hG[0], fh->dI);
                IOWrap::displayImage("frameToTrack", &img);

                while(true) {
                    char k=IOWrap::waitKey(0);
                    if(k==' ') break;
                    handleKey( k );
                }
//            handleKey(IOWrap::waitKey(1));
            }
        // end goStepByStep

		if(needKF) {
            auto t1 = std::chrono::steady_clock::now();
            makeKeyFrame(fh);
            auto t2 = std::chrono::steady_clock::now();
            makeKeyFrame_total_time +=
                    std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() *
                            1000;
        } else {
            auto t3 = std::chrono::steady_clock::now();
            makeNonKeyFrame(fh);
            auto t4 = std::chrono::steady_clock::now();
            makeNonKeyFrame_total_time +=
                    std::chrono::duration_cast<std::chrono::duration<double>>(t4 - t3).count() *
                            1000;
        }
	} else {
		boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
		unmappedTrackedFrames.push_back(fh);
		if(needKF) needNewKFAfter=fh->shell->trackingRef->id;
		trackedFrameSignal.notify_all();

		while(coarseTracker_forNewKF->refFrameID == -1 && coarseTracker->refFrameID == -1 ) {
			mappedFrameSignal.wait(lock);
		}

		lock.unlock();
	}
}

void FullSystem::mappingLoop()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);

	while(runMapping) {
		while(unmappedTrackedFrames.size()==0) {
			trackedFrameSignal.wait(lock);
			if(!runMapping) return;
		}

//        std::cout << "...........mappingLoop running............" << std::endl;
//        assert(false);

		FrameHessian* fh = unmappedTrackedFrames.front();
		unmappedTrackedFrames.pop_front();
		FrameHessian* fh_right = unmappedTrackedFrames_right.front();
		unmappedTrackedFrames_right.pop_front();

		// guaranteed to make a KF for the very first two tracked frames.
		if(allKeyFramesHistory.size() <= 2) {
			lock.unlock();
			makeKeyFrame(fh);
			lock.lock();
			mappedFrameSignal.notify_all();
			continue;
		}

		if(unmappedTrackedFrames.size() > 3)
			needToKetchupMapping=true;

		if(unmappedTrackedFrames.size() > 0) { // if there are other frames to tracke, do that first.
			lock.unlock();
			makeNonKeyFrame(fh);
			lock.lock();

			if(needToKetchupMapping && unmappedTrackedFrames.size() > 0) {
				FrameHessian* fh = unmappedTrackedFrames.front();
				unmappedTrackedFrames.pop_front();
				{
					boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
					assert(fh->shell->trackingRef != 0);
					fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
					fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);

				}
				delete fh;
			}

		} else {
			if(setting_realTimeMaxKF || needNewKFAfter >= frameHessians.back()->shell->id) {
				lock.unlock();
				makeKeyFrame(fh);
				needToKetchupMapping=false;
				lock.lock();
			} else {
				lock.unlock();
				makeNonKeyFrame(fh);
				lock.lock();
			}
		}
		mappedFrameSignal.notify_all();
	}
	printf("MAPPING FINISHED!\n");
}

void FullSystem::blockUntilMappingIsFinished()
{
	boost::unique_lock<boost::mutex> lock(trackMapSyncMutex);
	runMapping = false;
	trackedFrameSignal.notify_all();
	lock.unlock();

	mappingThread.join();

}

void FullSystem::makeNonKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread. no lock required since we are in mapping thread.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
//        printf("make non key frame........fh[%d].........trackingRef[%d]........\n",
//               fh->shell->id, fh->shell->trackingRef->id);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;
		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}

    // 通过非关键帧将窗口内的未成熟点成熟
    maturelize_window_based_on_newframe(fh);

	delete fh->frame_right;
	delete fh;
}

void FullSystem::makeKeyFrame( FrameHessian* fh)
{
	// needs to be set by mapping thread
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
		assert(fh->shell->trackingRef != 0);
//        printf("make key frame........fh[%d].........trackingRef[%d]........\n",
//               fh->shell->id,fh->shell->trackingRef->id);
		fh->shell->camToWorld = fh->shell->trackingRef->camToWorld * fh->shell->camToTrackingRef;

		fh->setEvalPT_scaled(fh->shell->camToWorld.inverse(),fh->shell->aff_g2l);
	}
// 	LOG(INFO)<<"make keyframe";

//  基于新帧制作未成熟点
//    plant_on_newframe(fh, 0);

// 通过新帧收斂窗口内的未成熟点，這裏只是收斂，並沒有激活
    maturelize_window_based_on_newframe(fh);

	boost::unique_lock<boost::mutex> lock(mapMutex);

	// =========================== Flag Frames to be Marginalized. =========================
	flagFramesForMarginalization(fh);

	// =========================== add New Frame to Hessian Struct. =========================
    // 新帧加入了窗口（現在才成爲關鍵幀）
	fh->idx = frameHessians.size();
	frameHessians.push_back(fh);
	fh->frameID = allKeyFramesHistory.size();
	fh->frame_right->frameID = 1000000 + allKeyFramesHistory.size();

	allKeyFramesHistory.push_back(fh->shell);
	ef->insertFrame(fh, &Hcalib);
#ifdef DSO_LITE
//    ef->insertFrame(fh->frame_right, &Hcalib);
#endif

// 这里需要加入对右图的支持
    setPrecalcValues();

	// =========================== add new residuals for old points =========================
#ifndef DSO_LITE
	int numFwdResAdde=0;
    // 所有的老关键帧上的所有的活跃点投影到新关键帧上的残差收集起来
    // 能量的计算，永远只在关键帧之间
	for(FrameHessian* fh1 : frameHessians) {		// go through all active frames
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians) {
			PointFrameResidual* r = new PointFrameResidual(ph, fh1, fh);
			r->setState(ResState::IN);
			ph->residuals.push_back(r);
            //能量项增加r
			ef->insertResidual(r);
			ph->lastResiduals[1] = ph->lastResiduals[0];
            // ph -> r
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r, ResState::IN);
			numFwdResAdde+=1;
		}
	}
#else
    int numFwdResAdde=0;
    // 所有的老关键帧上的所有的活跃点投影到新关键帧上的残差收集起来
    // 能量的计算，永远只在关键帧之间
	for(FrameHessian* fh1 : frameHessians) {		// go through all active frames
		if(fh1 == fh) continue;
		for(PointHessian* ph : fh1->pointHessians) {
			PointFrameResidual* r1 = new PointFrameResidual(ph, fh1, fh);
            PointFrameResidual* r2 = new PointFrameResidual(ph, fh1, fh);
			r1->setState(ResState::IN);
            r2->setState(ResState::IN);
			ph->residuals.push_back(r1);
            ph->residuals.push_back(r2);

            r2->stereoResidualFlag = true;

            //能量项增加r
			ef->insertResidual(r1);
            ef->insertResidual(r2);
			ph->lastResiduals[1] = ph->lastResiduals[0];
            // ph -> r
			ph->lastResiduals[0] = std::pair<PointFrameResidual*, ResState>(r1, ResState::IN);
			numFwdResAdde+=1;
		}
    }
#endif

	// =========================== Activate Points (& flag for marginalization). =========================
    // 这里会加入stereo的优化（即stereo和temporal的）。
    // 为PBA准备活跃点
    // 会遍历窗口内所有未成熟点，去除掉需要被删除的，
    // 用逆深度的均值作为先验，计算投影到窗口内的最新帧的残差，然后也会将收敛点的能量项搜集

    // 它可能只是在制作H, b，最终在优化中是被边缘化的，并不会实际计算
	activatePointsMT();
//    std::cout << "npoints: " << ef->nPoints << std::endl;
	ef->makeIDX();

	// =========================== OPTIMIZE ALL =========================
    // 优化关键帧组

	fh->frameEnergyTH = frameHessians.back()->frameEnergyTH;
//    std::cout << "optimize start" << std::endl;
	float rmse = optimize(setting_maxOptIterations);
//    std::cout << "optimize end" << std::endl;
//    std::cout << "rmse: " << rmse << std::endl;

	// =========================== Figure Out if INITIALIZATION FAILED =========================
	if(allKeyFramesHistory.size() <= 4) {
		if(allKeyFramesHistory.size()==2 && rmse > 20*benchmark_initializerSlackFactor) {
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==3 && rmse > 13*benchmark_initializerSlackFactor) {
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
		if(allKeyFramesHistory.size()==4 && rmse > 9*benchmark_initializerSlackFactor) {
			printf("I THINK INITIALIZATINO FAILED! Resetting.\n");
			initFailed=true;
		}
	}

    if(isLost) return;

// 	LOG(INFO)<<"remove outliers.";
	// =========================== REMOVE OUTLIER =========================
	removeOutliers();
    // 有新的参考帧了，要设置，给新跟踪器关联参考帧，然后根据该帧的成熟点给该帧制作跟踪器专用的金字塔
	{
		boost::unique_lock<boost::mutex> crlock(coarseTrackerSwapMutex);
		coarseTracker_forNewKF->makeK(&Hcalib);
		coarseTracker_forNewKF->setCoarseTrackingRef(frameHessians, Hcalib, false);
        coarseTracker_forNewKF->debugPlotIDepthMap(&minIdJetVisTracker, &maxIdJetVisTracker, outputWrapper);
        coarseTracker_forNewKF->debugPlotIDepthMapFloat(outputWrapper);
	}

	debugPlot("post Optimize");

// 	LOG(INFO)<<"flagPointsForRemoval";
	// =========================== (Activate-)Marginalize Points =========================
    // 将一些残差固定线性化
	flagPointsForRemoval();
// 	LOG(INFO)<<"ef->dropPointsF()";
	ef->dropPointsF();
// 	LOG(INFO)<<"after dropPoints: "<<ef->nPoints;
// 	LOG(INFO)<<"getNullspaces";
	getNullspaces(
			ef->lastNullspaces_pose,
			ef->lastNullspaces_scale,
			ef->lastNullspaces_affA,
			ef->lastNullspaces_affB);
// 	LOG(INFO)<<"ef->marginalizePointsF();";
	ef->marginalizePointsF();

// 	LOG(INFO)<<"makeNewTraces";
	// =========================== add new Immature points & new residuals =========================
//	makeNewTraces(fh, 0);
    plant_on_newframe(fh, 0);

// 	LOG(INFO)<<"makeNewTraces end";

    for(IOWrap::Output3DWrapper* ow : outputWrapper) {
        ow->publishGraph(ef->connectivityMap);
        ow->publishKeyframes(frameHessians, false, &Hcalib);
    }

// 	LOG(INFO)<<"marginalizeFrame";
	// =========================== Marginalize Frames =========================

	for(unsigned int i=0;i<frameHessians.size();i++)
		if(frameHessians[i]->flaggedForMarginalization)
			{marginalizeFrame(frameHessians[i]); i=0;}
// 	LOG(INFO)<<"make key end";
// 	delete fh_right;

// 	printLogLine();
    //printEigenValLine();

}

// 从CoarseInitializer的firstFrame制作allKeyFramesHistory和ef里的firstFrame
void FullSystem::initializeFromInitializer(FrameHessian* newFrame)
{
	boost::unique_lock<boost::mutex> lock(mapMutex);

	// add firstframe.
	FrameHessian* firstFrame = coarseInitializer->firstFrame;
//    assert(coarseInitializer->firstFrame == newFrame);
    // idx应该为0
	firstFrame->idx = frameHessians.size();
    assert(firstFrame->idx == 0);
    // 从这里把initializer里的frame送入fullsystem->frameHessians
	frameHessians.push_back(firstFrame);
	firstFrame->frameID = allKeyFramesHistory.size();
	firstFrame->frame_right->frameID = 1000000+allKeyFramesHistory.size();
	allKeyFramesHistory.push_back(firstFrame->shell);
	ef->insertFrame(firstFrame, &Hcalib);

	setPrecalcValues();

    FrameHessian* firstFrameRight = coarseInitializer->firstFrame->frame_right;

	//int numPointsTotal = makePixelStatus(firstFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	//int numPointsTotal = pixelSelector->makeMaps(firstFrame->dIp, selectionMap,setting_desiredDensity);

    // 使用const_lvl层来初始化
        int const_lvl = 0;
    printf("rkf:   wG[0]:{%d}, hG[0]:{%d}, wG[1]:{%d}, hG[1]:{%d}\n", wG[0], hG[0], wG[1], hG[1]);
    printf("all frame history size {%ld}\n", allFrameHistory.size());
	firstFrame->pointHessians.reserve(wG[const_lvl]*hG[const_lvl]*0.2f);
	firstFrame->pointHessiansMarginalized.reserve(wG[const_lvl]*hG[const_lvl]*0.2f);
	firstFrame->pointHessiansOut.reserve(wG[const_lvl]*hG[const_lvl]*0.2f);

	float idepthStereo = 0;
	float sumID=1e-5, numID=1e-5;
	for(int i=0;i<coarseInitializer->numPoints[const_lvl];i++) {
		sumID += coarseInitializer->points[const_lvl][i].iR;
		numID++;
	}
	float rescaleFactor = 1 / (sumID / numID);

	// randomly sub-select the points I need.
	float keepPercentage = setting_desiredPointDensity / coarseInitializer->numPoints[const_lvl];

    if (true)
        printf("Initialization: keep %.1f%% (need %d, have %d)!\n", 100*keepPercentage,
               (int)(setting_desiredPointDensity), coarseInitializer->numPoints[const_lvl] );

	for(int i=0;i<coarseInitializer->numPoints[const_lvl];i++) {
		if(rand()/(float)RAND_MAX > keepPercentage) continue;
// 对于未成熟点，使用右图来推断逆深度
		Pnt* point = coarseInitializer->points[const_lvl]+i;
		ImmaturePoint* pt = new ImmaturePoint(point->u+0.5f,point->v+0.5f,firstFrame,point->my_type, &Hcalib);
		
		pt->u_stereo = pt->u;
		pt->v_stereo = pt->v;
		pt->idepth_min_stereo = 0;
		pt->idepth_max_stereo = NAN;

        // 对角化得到pt的逆深度
		pt->traceStereo(firstFrameRight, &Hcalib, 1);

		pt->idepth_min = pt->idepth_min_stereo;
		pt->idepth_max = pt->idepth_max_stereo;
		idepthStereo = pt->idepth_stereo;
		
		if(!std::isfinite(pt->energyTH) || !std::isfinite(pt->idepth_min) || !std::isfinite(pt->idepth_max)
				|| pt->idepth_min < 0 || pt->idepth_max < 0) {
		    delete pt;
		    continue;
		}
		PointHessian* ph = new PointHessian(pt, &Hcalib);
		delete pt;
		if(!std::isfinite(ph->energyTH)) {delete ph; continue;}
		
		ph->setIdepthScaled(idepthStereo);
		ph->setIdepthZero(idepthStereo);
		ph->hasDepthPrior=true;
		ph->setPointStatus(PointHessian::ACTIVE);

// 		ph->setIdepthScaled(point->iR*rescaleFactor);
// 		ph->setIdepthZero(ph->idepth);
// 		ph->hasDepthPrior=true;
// 		ph->setPointStatus(PointHessian::ACTIVE);

		firstFrame->pointHessians.push_back(ph);
        ef->insertPoint(ph);
#ifndef DSO_LITE
        for (int i = 0; i < 1/*coupling_factor*/; i++) {
            PointFrameResidual *r = new PointFrameResidual(ph,
                                                           ph->host,
                                                           ph->host->frame_right);
            r->state_NewEnergy = r->state_energy = 0;
            r->state_NewState = ResState::OUTLIER;
            r->setState(ResState::IN);
            r->stereoResidualFlag = true;
            // put r in ph
            ph->residuals.push_back(r);
            // put r in ef
            ef->insertResidual(r);
        }
#else
//        PointFrameResidual *r = new PointFrameResidual(ph,
//                                                           ph->host,
//                                                           ph->host);
//        r->state_NewEnergy = r->state_energy = 0;
//        r->state_NewState = ResState::OUTLIER;
//        r->setState(ResState::IN);
//        r->stereoResidualFlag = true;
//        ph->residuals.push_back(r);
//        ef->insertResidual(r);
#endif
	}

	SE3 firstToNew = coarseInitializer->thisToNext;
    // thisToNext为SE3()
    std::cout <<"thisToNext: " << coarseInitializer->thisToNext.matrix() << std::endl;
    std::cout <<"rescaleFactor " << rescaleFactor << std::endl;
	firstToNew.translation() /= rescaleFactor;

	// really no lock required, as we are initializing.
	{
		boost::unique_lock<boost::mutex> crlock(shellPoseMutex);
        // 第0帧的位姿为SE3()
		firstFrame->shell->camToWorld = SE3();
		firstFrame->shell->aff_g2l = AffLight(0,0);
		firstFrame->setEvalPT_scaled(firstFrame->shell->camToWorld.inverse(),
                                     firstFrame->shell->aff_g2l);

        // 关联0 -> first
		firstFrame->shell->trackingRef=0;
		firstFrame->shell->camToTrackingRef = SE3();

        // newFrame也设置成了SE3()
		newFrame->shell->camToWorld = firstToNew.inverse();
		newFrame->shell->aff_g2l = AffLight(0,0);
		newFrame->setEvalPT_scaled(newFrame->shell->camToWorld.inverse(),newFrame->shell->aff_g2l);

        //关联first -> new
		newFrame->shell->trackingRef = firstFrame->shell;
        // 已经有了第1帧的相对位姿，这里好像有点问题？没有旋转。
		newFrame->shell->camToTrackingRef = firstToNew.inverse();
	}

	initialized=true;
	printf("INITIALIZE FROM INITIALIZER (%d pts)!\n", (int)firstFrame->pointHessians.size());
}

// 根据新帧增加未成熟点
// plant_on_newframe
void FullSystem::plant_on_newframe(FrameHessian* newFrame, float* gtDepth)
{
	pixelSelector->allowFast = true;
	//int numPointsTotal = makePixelStatus(newFrame->dI, selectionMap, wG[0], hG[0], setting_desiredDensity);
	int numPointsTotal = pixelSelector->makeMaps(newFrame,
                                                 selectionMap,
                                                 setting_desiredImmatureDensity);

	newFrame->pointHessians.reserve(numPointsTotal*1.2f);
	//fh->pointHessiansInactive.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansMarginalized.reserve(numPointsTotal*1.2f);
	newFrame->pointHessiansOut.reserve(numPointsTotal*1.2f);

	for(int y=patternPadding+1;y<hG[0]-patternPadding-2;y++)
	for(int x=patternPadding+1;x<wG[0]-patternPadding-2;x++) {
		int i = x+y*wG[0];
        // 如果没有选中
		if(selectionMap[i]==0) continue;

		ImmaturePoint* impt = new ImmaturePoint(x,y,newFrame, selectionMap[i], &Hcalib);
		if(!std::isfinite(impt->energyTH)) delete impt;
		else newFrame->immaturePoints.push_back(impt);
	}
	//printf("MADE %d IMMATURE POINTS!\n", (int)newFrame->immaturePoints.size());

}
void FullSystem::setPrecalcValues()
{
	for(FrameHessian* fh : frameHessians) {
		fh->targetPrecalc.resize(frameHessians.size());

		for(unsigned int i=0;i<frameHessians.size();i++) {
            fh->targetPrecalc[i].set(fh, frameHessians[i], &Hcalib);
//            fh->targetPrecalc_right[i].set(fh, frameHessians[i]->frame_right, &Hcalib);
        }
	}
	ef->setDeltaF(&Hcalib);
}

// 制作一个运动先验估计的模版，第一个是零运动，后面的都是微小的旋转扰动
void FullSystem::make_motion_prediction_based_on_assumption()
{
        // 第1帧到第2帧，没有运动先验，则取R=I, t=0为先验
        // 两个过程是类似的，只是第1帧到第2帧完全没有运动先验
//		this -> lastF_2_fh_tries.push_back(SE3(Eigen::Matrix<double, 3, 3>::Identity(),
//                                       Eigen::Matrix<double,3,1>::Zero() ));

		for(float rotDelta=0.02; rotDelta < 0.05; rotDelta = rotDelta + 0.02) {
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,rotDelta), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,0), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,0,-rotDelta), Vec3(0,0,0)));			// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,0), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,0,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,0,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,-rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,-rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,-rotDelta), Vec3(0,0,0)));	// assume constant motion.
		    lastF_2_fh_tries.push_back(SE3(Sophus::Quaterniond(1,rotDelta,rotDelta,rotDelta), Vec3(0,0,0)));	// assume constant motion.
		}
}

}
