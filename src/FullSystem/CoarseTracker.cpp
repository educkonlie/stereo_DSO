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

#include "FullSystem/CoarseTracker.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/ImmaturePoint.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"
#include "IOWrapper/ImageRW.h"
#include <algorithm>
#include <execution>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{


template<int b, typename T>
T* allocAligned(int size, std::vector<T*> &rawPtrVec)
{
    const int padT = 1 + ((1 << b)/sizeof(T));
    T* ptr = new T[size + padT];
    rawPtrVec.push_back(ptr);
    T* alignedPtr = (T*)(( ((uintptr_t)(ptr+padT)) >> b) << b);
    return alignedPtr;
}


CoarseTracker::CoarseTracker(int ww, int hh) : lastRef_aff_g2l(0,0)
{
	// make coarse tracking templates.
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++) {
		int wl = ww>>lvl;
        int hl = hh>>lvl;

        idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        weightSums_bak[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);

        pc_u[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_v[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_idepth[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
        pc_color[lvl] = allocAligned<4,float>(wl*hl, ptrToDelete);
	}

	// warped buffers
    buf_warped_idepth = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_u = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_v = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dx = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_dy = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_residual = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_weight = allocAligned<4,float>(ww*hh, ptrToDelete);
    buf_warped_refColor = allocAligned<4,float>(ww*hh, ptrToDelete);


	newFrame = 0;
	lastRef = 0;
	debugPlot = debugPrint = true;
	w[0]=h[0]=0;
	refFrameID=-1;
}
CoarseTracker::~CoarseTracker()
{
    for(float* ptr : ptrToDelete)
        delete[] ptr;
    ptrToDelete.clear();
}

void CoarseTracker::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level) {
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level) {
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
	}
}

/*void CoarseTracker::makeCoarseDepthForFirstFrame()
{
    // make coarse tracking templates for latestRef.
    memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
    memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);

    FrameHessian *fh = this->lastRef;

    printf("..........makeCoarseDepthForFirstFrame..lastRef[%d]........\n", fh->shell->id);
    // here fh denotes for lastRef
    // 从这里开始给lastRef重建金字塔，initializer的金字塔确实是多余的
    printf(".......lastRef -> pointHessians.size(){%ld}....\n", lastRef->pointHessians.size());
    for(PointHessian* ph : fh->pointHessians) {
        int u = ph->u + 0.5f;
        int v = ph->v + 0.5f;

        float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));
        // 其实weight就是1，也就是其实就是计数
        assert(abs(weight - 1.0) < 0.01);

        // 有ph的像素的逆深度才会设置，没有的就隐式设为0
        // +=其实就是在累加和计数
        idepth[0][u+w[0]*v] += ph->idepth * weight;
        weightSums[0][u+w[0]*v] += weight;
    }
    normalize_pyramid();
}*/
// 为新的参考帧制作好跟踪底版
// 新的参考帧总是新帧，也是新的关键帧
void CoarseTracker::makeCoarseDepthL0(std::vector<FrameHessian*> frameHessians,
                                      CalibHessian Hcalib, bool first)
{
	// make coarse tracking templates for latestRef.
	memset(idepth[0], 0, sizeof(float)*w[0]*h[0]);
	memset(weightSums[0], 0, sizeof(float)*w[0]*h[0]);
    //确认一下lastRef已经被设置好了
	FrameHessian* fh_target = frameHessians.back();
    assert(fh_target == lastRef);

    int k_gt_50 = 0;
    int k_total = 0;
    int k_err   = 0;

    if (first) {
    printf("..........makeCoarseDepthForFirstFrame..lastRef[%d]........\n", fh_target->shell->id);
        // here fh denotes for lastRef
        // 从这里开始给lastRef重建金字塔，initializer的金字塔确实是多余的
    printf(".......lastRef -> pointHessians.size(){%ld}....\n", lastRef->pointHessians.size());
        for (PointHessian *ph: lastRef->pointHessians) {
            int u = ph->u + 0.5f;
            int v = ph->v + 0.5f;

            float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));
            // 其实weight就是1，也就是其实就是计数
            assert(abs(weight - 1.0) < 0.01);

            // 有ph的像素的逆深度才会设置，没有的就隐式设为0
            // +=其实就是在累加和计数
            // 已经有了一堆host为lastRef的点，只要用它们做一个金字塔就可以了
            idepth[0][u + w[0] * v] += ph->idepth * weight;
            weightSums[0][u + w[0] * v] += weight;
        }
    } else {
        // 窗口内的所有活跃点刚有投影残差在lastRef上，所以可以直接求得(u, v, idepth)
        // 但这里又将左图得到的idepth变为(idepth_min, idepth_max)，分别为idepth * 0.1, idepth * 1.9
        // 然后根据右图确定深度值

        // TRACE_STEREO_AGAIN這裏的收斂過程，在trace(maturelize)的時候對idepth_min idepth_max已經做過了
        // 是因爲optimize後，幀間的位姿改變了，所以才會有重新stereo的需求（优化的时候也会根据新的内参优化逆深度的）

        // 根本原因是優化的時候，參考幀的右目還沒有加入，參考幀右目無法加入的原因是host爲參考幀的未成熟點尚不存在
        // 所以才會有dso-lite對於這方面的改造

#ifdef TRACE_STEREO_AGAIN
        for (FrameHessian *fh: frameHessians) {
            for (PointHessian *ph: fh->pointHessians) {
                if (ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) {
                    //contains information about residuals to the last two (!) frames.
                    // ([0] = latest, [1] = the one before).
                    PointFrameResidual *r = ph->lastResiduals[0].first;
                    assert(r->efResidual->isActive() && r->target == lastRef);
                    int u = r->centerProjectedTo[0] + 0.5f;
                    int v = r->centerProjectedTo[1] + 0.5f;

                    float new_idepth = 0;
                    k_total++;
//                    if (1.0 / (r -> centerProjectedTo[2]) > 50.0) {
                    if (r -> centerProjectedTo[2] < 0.02) {
                        k_gt_50++;
                        // 這裏的深度值來源於激活未成熟點的時候，直接取了中值 0.5 * (idepth_min + idepth_max)
                        new_idepth = r->centerProjectedTo[2];
                        float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));
                        idepth[0][u + w[0] * v] += new_idepth * weight;
                        weightSums[0][u + w[0] * v] += weight;
                        continue;
                    }

                    ImmaturePoint *pt_track = new ImmaturePoint((float) u, (float) v, lastRef, &Hcalib);

                    pt_track->u_stereo = pt_track->u;
                    pt_track->v_stereo = pt_track->v;

                    // free to debug
                    // 將中值再次拆分
                    pt_track->idepth_min_stereo = r->centerProjectedTo[2] * 0.3f;
                    pt_track->idepth_max_stereo = r->centerProjectedTo[2] * 1.7f;
                    ImmaturePointStatus pt_track_right = pt_track->traceStereo(lastRef->frame_right, &Hcalib, 1);

                    if (pt_track_right == ImmaturePointStatus::IPS_GOOD) {
                        ImmaturePoint *pt_track_back = new ImmaturePoint(pt_track->lastTraceUV(0),
                                                                         pt_track->lastTraceUV(1),
                                                                         lastRef->frame_right, &Hcalib);
                        pt_track_back->u_stereo = pt_track_back->u;
                        pt_track_back->v_stereo = pt_track_back->v;
                        // 0.1 1.9是比較狠的放縮
                        pt_track_back->idepth_min_stereo = r->centerProjectedTo[2] * 0.3f;
                        pt_track_back->idepth_max_stereo = r->centerProjectedTo[2] * 1.7f;

                        ImmaturePointStatus pt_track_left = pt_track_back->traceStereo(lastRef, &Hcalib, 0);

                        float depth = 1.0f / pt_track->idepth_stereo;
                        float u_delta = abs(pt_track->u - pt_track_back->lastTraceUV(0));
                        if (u_delta < 1 && depth > 0 && depth < 50) {
                            if (std::abs(depth - 1.0 / r->centerProjectedTo[2]) > 5.0) {
                                k_err++;
                            }
                            new_idepth = pt_track->idepth_stereo;
                            delete pt_track;
                            delete pt_track_back;
                        } else {
                            new_idepth = r->centerProjectedTo[2];
                            delete pt_track;
                            delete pt_track_back;
                        }
                    } else {
                        new_idepth = r->centerProjectedTo[2];
                        delete pt_track;
                    }
                    float weight = sqrtf(1e-3 / (ph->efPoint->HdiF + 1e-12));
                    idepth[0][u + w[0] * v] += new_idepth * weight;
                    weightSums[0][u + w[0] * v] += weight;
                }
            }
        }
    // 下面这个没有引入右图
#else
        for(FrameHessian* fh : frameHessians) {
            for(PointHessian* ph : fh->pointHessians) {
                if(ph->lastResiduals[0].first != 0 && ph->lastResiduals[0].second == ResState::IN) {
                    PointFrameResidual* r = ph->lastResiduals[0].first;
                    // 笃定这些点刚刚都在lastRef上参与过优化.
                    // 即r为ph投影到target==lastRef上的残差
                    assert(r->efResidual->isActive() && r->target == lastRef);

                    // 还不太明白这三个的计算过程
                    //是在linearize和linearizeStereo里： centerProjectedTo = Vec3f(Ku, Kv, new_idepth);
                    // 所以就是warp到新的帧上
                    int u = r->centerProjectedTo[0] + 0.5f;
                    int v = r->centerProjectedTo[1] + 0.5f;
                    float new_idepth = r->centerProjectedTo[2];

                    float weight = sqrtf(1e-3 / (ph->efPoint->HdiF+1e-12));

                    idepth[0][u+w[0]*v] += new_idepth *weight;
                    weightSums[0][u+w[0]*v] += weight;
                }
            }
        }
#endif
    }
//    std::cout << "k_total " << k_total << " k_gt_50 " << k_gt_50 << " k_err " << k_err << std::endl;
    normalize_pyramid();
}

void CoarseTracker::calcGSSSE(int lvl, Mat88 &H_out, Vec8 &b_out, const SE3 &refToNew, AffLight aff_g2l)
{
	acc.initialize();

	__m128 fxl = _mm_set1_ps(fx[lvl]);
	__m128 fyl = _mm_set1_ps(fy[lvl]);
	__m128 b0 = _mm_set1_ps(lastRef_aff_g2l.b);
	__m128 a = _mm_set1_ps((float)(AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l)[0]));

	__m128 one = _mm_set1_ps(1);
	__m128 minusOne = _mm_set1_ps(-1);
	__m128 zero = _mm_set1_ps(0);
	
	int n = buf_warped_n;
	assert(n%4==0);
	for(int i=0;i<n;i+=4) {
		__m128 dx = _mm_mul_ps(_mm_load_ps(buf_warped_dx+i), fxl);
		__m128 dy = _mm_mul_ps(_mm_load_ps(buf_warped_dy+i), fyl);
		__m128 u = _mm_load_ps(buf_warped_u+i);
		__m128 v = _mm_load_ps(buf_warped_v+i);
		__m128 id = _mm_load_ps(buf_warped_idepth+i);

        /*
         *                id * dx
         *                id * dy
         *                -id * (u * dx + v * dy)
         *
         */
		acc.updateSSE_eighted(
				_mm_mul_ps(id,dx),
				_mm_mul_ps(id,dy),
				_mm_sub_ps(zero, _mm_mul_ps(id,_mm_add_ps(_mm_mul_ps(u,dx), _mm_mul_ps(v,dy)))),
				_mm_sub_ps(zero, _mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dx),
						_mm_mul_ps(dy,_mm_add_ps(one, _mm_mul_ps(v,v))))),
				_mm_add_ps(
						_mm_mul_ps(_mm_mul_ps(u,v),dy),
						_mm_mul_ps(dx,_mm_add_ps(one, _mm_mul_ps(u,u)))),
				_mm_sub_ps(_mm_mul_ps(u,dy), _mm_mul_ps(v,dx)),
				_mm_mul_ps(a,_mm_sub_ps(b0, _mm_load_ps(buf_warped_refColor+i))),
				minusOne,
				_mm_load_ps(buf_warped_residual+i),
				_mm_load_ps(buf_warped_weight+i));
	}

	acc.finish();
    //! 看来8X8的H和8X1的b合成了acc.H
	H_out = acc.H.topLeftCorner<8,8>().cast<double>() * (1.0f/n);
	b_out = acc.H.topRightCorner<8,1>().cast<double>() * (1.0f/n);

	H_out.block<8,3>(0,0) *= SCALE_XI_ROT;
	H_out.block<8,3>(0,3) *= SCALE_XI_TRANS;
	H_out.block<8,1>(0,6) *= SCALE_A;
	H_out.block<8,1>(0,7) *= SCALE_B;
	H_out.block<3,8>(0,0) *= SCALE_XI_ROT;
	H_out.block<3,8>(3,0) *= SCALE_XI_TRANS;
	H_out.block<1,8>(6,0) *= SCALE_A;
	H_out.block<1,8>(7,0) *= SCALE_B;
	b_out.segment<3>(0) *= SCALE_XI_ROT;
	b_out.segment<3>(3) *= SCALE_XI_TRANS;
	b_out.segment<1>(6) *= SCALE_A;
	b_out.segment<1>(7) *= SCALE_B;
}

// 计算残差
/*Vec6 CoarseTracker::calcRes(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH)
{
	float E = 0;
	int numTermsInE = 0;
	int numTermsInWarped = 0;
	int numSaturated=0;

	int wl = w[lvl];
	int hl = h[lvl];
	Eigen::Vector3f* dINewl = newFrame -> dIp[lvl];
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f t = (refToNew.translation()).cast<float>();
	Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure,
                                              newFrame->ab_exposure,
                                              lastRef_aff_g2l, aff_g2l).cast<float>();

	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.

    MinimalImageB3* resImage = 0;
	if(debugPlot) {
		resImage = new MinimalImageB3(wl,hl);
		resImage->setConst(Vec3b(0,128,0));
	}

    // 跟踪器里的金字塔数据
	int nl = pc_n[lvl];
	float* lpc_u = pc_u[lvl];
	float* lpc_v = pc_v[lvl];
	float* lpc_idepth = pc_idepth[lvl];
	float* lpc_color = pc_color[lvl];

    // 这里可以并行化
	for(int i=0;i<nl;i++) {
		float id = lpc_idepth[i];
		float x = lpc_u[i];
		float y = lpc_v[i];

        // 这里就是warping: (x, y, id) -> (Ku, Kv, new_id)
		Vec3f pt = RKi * Vec3f(x, y, 1) + t*id;
		float u = pt[0] / pt[2];
		float v = pt[1] / pt[2];
		float Ku = fxl * u + cxl;
		float Kv = fyl * v + cyl;
		float new_idepth = id/pt[2];

        // 第0层，每32个点
		if(lvl==0 && i%32==0) {
			// translation only (positive)
			Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t*id;
			float uT = ptT[0] / ptT[2];
			float vT = ptT[1] / ptT[2];
			float KuT = fxl * uT + cxl;
			float KvT = fyl * vT + cyl;

			// translation only (negative)
			Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t*id;
			float uT2 = ptT2[0] / ptT2[2];
			float vT2 = ptT2[1] / ptT2[2];
			float KuT2 = fxl * uT2 + cxl;
			float KvT2 = fyl * vT2 + cyl;

			//translation and rotation (negative)
			Vec3f pt3 = RKi * Vec3f(x, y, 1) - t*id;
			float u3 = pt3[0] / pt3[2];
			float v3 = pt3[1] / pt3[2];
			float Ku3 = fxl * u3 + cxl;
			float Kv3 = fyl * v3 + cyl;

			//translation and rotation (positive)
			//already have it.

			sumSquaredShiftT += (KuT-x)*(KuT-x) + (KvT-y)*(KvT-y);
			sumSquaredShiftT += (KuT2-x)*(KuT2-x) + (KvT2-y)*(KvT2-y);
			sumSquaredShiftRT += (Ku-x)*(Ku-x) + (Kv-y)*(Kv-y);
			sumSquaredShiftRT += (Ku3-x)*(Ku3-x) + (Kv3-y)*(Kv3-y);
			sumSquaredShiftNum+=2;
		}

        // Ku, Kv在图像边沿，或逆深度<=0，则抛弃该点
		if(!(Ku > 2 && Kv > 2 && Ku < wl-3 && Kv < hl-3 && new_idepth > 0)) continue;
        //于是是有效点，对残差项进行计数
        numTermsInE++;

		float refColor = lpc_color[i];
        Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
        if(!std::isfinite((float)hitColor[0])) continue;
        float residual = hitColor[0] - (float)(affLL[0] * refColor + affLL[1]);
        // hw是 max{阈值/残差, 1}
        // huber weight
        float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

        // 如果outlier，即残差太大了，则用maxEnergy表示
        if(fabs(residual) > cutoffTH) {
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));
			E += maxEnergy;
			numSaturated++;
        } else {
			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i],
                                              Vec3b(residual+128,residual+128,residual+128));
            // 其实就是残差的平方，但是做了一些处理
            // hw: huber weight
            // hw*residual和(2-hw)*residual爲兩個相同的residual值拆成，當hw=1.0時，面積最大
			E += hw *residual*residual*(2-hw);

			buf_warped_idepth[numTermsInWarped] = new_idepth;
			buf_warped_u[numTermsInWarped] = u;
			buf_warped_v[numTermsInWarped] = v;
			buf_warped_dx[numTermsInWarped] = hitColor[1];
			buf_warped_dy[numTermsInWarped] = hitColor[2];
			buf_warped_residual[numTermsInWarped] = residual;
			buf_warped_weight[numTermsInWarped] = hw;
			buf_warped_refColor[numTermsInWarped] = lpc_color[i];
			numTermsInWarped++;
		}
	}

    // 填充0, 对齐
	while(numTermsInWarped%4!=0) {
		buf_warped_idepth[numTermsInWarped] = 0;
		buf_warped_u[numTermsInWarped] = 0;
		buf_warped_v[numTermsInWarped] = 0;
		buf_warped_dx[numTermsInWarped] = 0;
		buf_warped_dy[numTermsInWarped] = 0;
		buf_warped_residual[numTermsInWarped] = 0;
		buf_warped_weight[numTermsInWarped] = 0;
		buf_warped_refColor[numTermsInWarped] = 0;
		numTermsInWarped++;
	}
	buf_warped_n = numTermsInWarped;

	if(debugPlot) {
		IOWrap::displayImage("RES", resImage, false);
		IOWrap::waitKey(0); // ruan kefeng
		delete resImage;
	}

    // 关心的一般是平均残差rs[0]/rs[1]。 rs[5]用来表示outlier占比，如果过高则放大阈值重新计算
    // rs[2] rs[3] rs[4]是三个flowIndicator，用来判断是否要增加关键帧
	Vec6 rs;
	rs[0] = E;
	rs[1] = numTermsInE;
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
	rs[3] = 0;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
	rs[5] = numSaturated / (float)numTermsInE;

	return rs;
}*/

// 这里会设置新的参考帧，时机是创建新关键帧的时候，会制作for_newkey跟踪器
// 当来新帧的时候，比较for_newkey跟踪器和现有跟踪器，如果for_newkey跟踪器的帧更新，就使用for_newkey跟踪器
void CoarseTracker::setCoarseTrackingRef(
		std::vector<FrameHessian*> frameHessians, CalibHessian Hcalib, bool first)
{
	assert(frameHessians.size()>0);
	this -> lastRef = frameHessians.back();

//    printf("......setCoarseTrackingRef...size[%ld]...new lastRef set[%d]....\n",
//           frameHessians.size(), lastRef->shell->id);

        // 给lastRef重建金字塔
//    pc_n[0] is 24025
//    pc_n[1] is 17885
//    pc_n[2] is 8441
//    pc_n[3] is 3421
    makeCoarseDepthL0(frameHessians, Hcalib, first);

	refFrameID = lastRef->shell->id;
	lastRef_aff_g2l = lastRef->aff_g2l();

	firstCoarseRMSE=-1;
}

// 这个才是金字塔跟踪
// 最终结果是lastToNew_out aff_g2l_out和lastResiduals
bool CoarseTracker::track(
		FrameHessian* newFrameHessian,
		SE3 &lastToNew_out, AffLight &aff_g2l_out,
		int coarsestLvl,
		Vec5 minResForAbort,
		IOWrap::Output3DWrapper* wrap)
{
	debugPlot = setting_render_displayCoarseTrackingFull;
//    debugPlot = true;
	debugPrint = false;

	assert(coarsestLvl < 5 && coarsestLvl < pyrLevelsUsed);

	lastResiduals.setConstant(NAN);
	lastFlowIndicators.setConstant(1000);

    // 这里把新帧送给跟踪器
	this->newFrame = newFrameHessian;

	int maxIterations[] = {10,20,50,50,50};
	float lambdaExtrapolationLimit = 0.001;

	SE3 refToNew_current = lastToNew_out;
	AffLight aff_g2l_current = aff_g2l_out;

	bool haveRepeated = false;

    // 按金字塔从粗到精来计算。每层的残差分别存放，但因为求解的位姿是持续迭代的，
    // 所以最终取residual[0]为残差结果，而位姿也是最终结果
	for(int lvl=coarsestLvl; lvl>=0; lvl--) {
//        lvl = 0;
		Mat88 H; Vec8 b;
		float levelCutoffRepeat=1;
/*	if(multiThreading)
 * //因爲treadReduce也可能被map_running線程調用，所以最好換一個新的線程池
		treadPoolReduce.reduce(boost::bind(&FullSystem::activatePointsMT_Reductor, this,
                                       &optimized, &toOptimize, _1, _2, _3, _4),
                           0, toOptimize.size(), 50);
	else
		activatePointsMT_Reductor(&optimized, &toOptimize, 0, toOptimize.size(), 0, 0);*/

		Vec6 resOld = calcRes_MT(lvl, refToNew_current,
                              aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);
        while(resOld[5] > 0.6 && levelCutoffRepeat < 50) { // outlier的比例过高, 把阈值加倍
			levelCutoffRepeat*=2;
            //            Vec6 rs;
//            rs[0] = E;
//            rs[1] = numTermsInE;
//            rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
//            rs[3] = 0;
//            rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
//            rs[5] = numSaturated / (float)numTermsInE;
			resOld = calcRes_MT(lvl, refToNew_current,
                             aff_g2l_current, setting_coarseCutoffTH*levelCutoffRepeat);

            printf("INCREASING cutoff to %f (ratio is %f)!\n",
                   setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
//            if(!setting_debugout_runquiet)
//                printf("INCREASING cutoff to %f (ratio is %f)!\n",
//                       setting_coarseCutoffTH*levelCutoffRepeat, resOld[5]);
		}
        // 设置H， b
        // 取的数据是warp后的，所以已经隐含了refToNew_current
		calcGSSSE(lvl, H, b, refToNew_current, aff_g2l_current);

		float lambda = 0.01;
		if(debugPrint) {
			Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure,
                                                       newFrame->ab_exposure,
                                                       lastRef_aff_g2l, aff_g2l_current).cast<float>();
			printf("lvl%d, iteration %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
					lvl, -1, lambda, 1.0f,
					"INITIA",
					0.0f,
					resOld[0] / resOld[1],
					 0,(int)resOld[1],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << aff_g2l_current.vec().transpose() <<" (rel " << relAff.transpose() << ")\n";
		}

        // 层内的迭代
		for(int iteration=0; iteration < maxIterations[lvl]; iteration++) {
			Mat88 Hl = H;
            // lambda的正确用法应当如下MY_LAMBDA。原代码有误，所以去掉了lambda的使用。
            // 原代码的做法的LM算法似乎也是有的，虽然无法让半正定变正定，但是可以增加一个damp，
            // 这样的好处是会减弱每次的step，让系统更鲁棒。
            // 贺博说VIO中就会这样做。
#ifdef MY_LAMBDA
            for(int i=0;i<8;i++) Hl(i,i) += lambda /** 0.001*/; // 鲁棒
#else
            for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda); // 鲁棒
#endif

			Vec8 inc = Hl.ldlt().solve(-b);

            // inc已经解出来了，因为不同的设置，可能要抛弃掉a, b中的一两项，同时也要重新计算inc
			if(setting_affineOptModeA < 0 && setting_affineOptModeB < 0) {	// fix a, b
				inc.head<6>() = Hl.topLeftCorner<6,6>().ldlt().solve(-b.head<6>());
			 	inc.tail<2>().setZero();
			}
			if(!(setting_affineOptModeA < 0) && setting_affineOptModeB < 0)	{// fix b
				inc.head<7>() = Hl.topLeftCorner<7,7>().ldlt().solve(-b.head<7>());
			 	inc.tail<1>().setZero();
			}
            // 这里的计算是没有涉及边缘化的，直接就是砍掉了不需要的行
			if(setting_affineOptModeA < 0 && !(setting_affineOptModeB < 0)) {	// fix a
				Mat88 HlStitch = Hl;
				Vec8 bStitch = b;
				HlStitch.col(6) = HlStitch.col(7);
				HlStitch.row(6) = HlStitch.row(7);
				bStitch[6] = bStitch[7];
				Vec7 incStitch = HlStitch.topLeftCorner<7,7>().ldlt().solve(-bStitch.head<7>());
				inc.setZero();
				inc.head<6>() = incStitch.head<6>();
				inc[6] = 0;
				inc[7] = incStitch[6];
			}

			float extrapFac = 1;
			if(lambda < lambdaExtrapolationLimit) extrapFac = sqrt(sqrt(lambdaExtrapolationLimit / lambda));
			inc *= extrapFac;

			Vec8 incScaled = inc;
			incScaled.segment<3>(0) *= SCALE_XI_ROT;
			incScaled.segment<3>(3) *= SCALE_XI_TRANS;
			incScaled.segment<1>(6) *= SCALE_A;
			incScaled.segment<1>(7) *= SCALE_B;

            if(!std::isfinite(incScaled.sum())) incScaled.setZero();

            // 解Hx = -b，得到x_inc，然后将位姿更新量incScaled.head<6>()加性操作到先验运动估计值上去
			SE3 refToNew_new = SE3::exp((Vec6)(incScaled.head<6>())) * refToNew_current;
			AffLight aff_g2l_new = aff_g2l_current;
			aff_g2l_new.a += incScaled[6];
			aff_g2l_new.b += incScaled[7];

            // 计算一下更新后的位姿估计值下的残差
			Vec6 resNew = calcRes_MT(lvl, refToNew_new, aff_g2l_new, setting_coarseCutoffTH*levelCutoffRepeat);

            // resNew[0] == E;   resNew[1] == numTermsInE;
                // 6 degree residuals
//            Vec6 rs;
//            rs[0] = E;
//            rs[1] = numTermsInE;
//            rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
//            rs[3] = 0;
//            rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
//            rs[5] = numSaturated / (float)numTermsInE;

//           新的平均绝对残差比老的小, 更新有效
			bool accept = (resNew[0] / resNew[1]) < (resOld[0] / resOld[1]);

			if(debugPrint) {
                if (!accept) {
                    Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure,
                                                               lastRef_aff_g2l, aff_g2l_new).cast<float>();
                    printf("lvl %d, it %d (l=%f / %f) %s: %.3f->%.3f (%d -> %d) (|inc| = %f)! \t",
                           lvl, iteration, lambda,
                           extrapFac,
                           (accept ? "ACCEPT" : "REJECT"),
                           resOld[0] / resOld[1],
                           resNew[0] / resNew[1],
                           (int) resOld[1], (int) resNew[1],
                           inc.norm());
                    std::cout << refToNew_new.log().transpose() << " AFF " << aff_g2l_new.vec().transpose() << " (rel "
                              << relAff.transpose() << ")\n";
                }
			}
			if(accept) {
				calcGSSSE(lvl, H, b, refToNew_new, aff_g2l_new);
                // 这里完成old到new的更新
				resOld = resNew;
				aff_g2l_current = aff_g2l_new;
				refToNew_current = refToNew_new;
				lambda *= 0.5;
			} else {
				lambda *= 4;
				if(lambda < lambdaExtrapolationLimit) lambda = lambdaExtrapolationLimit;
			}

            // 在金字塔该层精度下的inc已经很小了，要退出迭代，进入金字塔下一层的优化了
			if(!(inc.norm() > 1e-3)) {
				if(debugPrint)
					printf("lvl[%d] inc too small, break!\n", lvl);
				break;
			}
		}
        // 一般来说，都是从上面的break退出迭代，于是最终的结果是在resOld里

		// set last residual for that level, as well as flow indicators.
		this -> lastResiduals[lvl] = sqrtf((float)(resOld[0] / resOld[1]));
		this -> lastFlowIndicators = resOld.segment<3>(2);
        // 判断条件minResForAbort只在这里有效
		if(lastResiduals[lvl] > 1.5*minResForAbort[lvl]) return false;

		if(levelCutoffRepeat > 1 && !haveRepeated) {
			lvl++;
			haveRepeated=true;
			printf("REPEAT LEVEL! threshold has been enlarged(to handle too much outliers)..\n");
		}
	}

	// set!
	lastToNew_out = refToNew_current;
	aff_g2l_out = aff_g2l_current;

	if((setting_affineOptModeA != 0 && (fabsf(aff_g2l_out.a) > 1.2))
	|| (setting_affineOptModeB != 0 && (fabsf(aff_g2l_out.b) > 200)))
		return false;

	Vec2f relAff = AffLight::fromToVecExposure(lastRef->ab_exposure, newFrame->ab_exposure, lastRef_aff_g2l, aff_g2l_out).cast<float>();

	if((setting_affineOptModeA == 0 && (fabsf(logf((float)relAff[0])) > 1.5))
	|| (setting_affineOptModeB == 0 && (fabsf((float)relAff[1]) > 200)))
		return false;

	if(setting_affineOptModeA < 0) aff_g2l_out.a=0;
	if(setting_affineOptModeB < 0) aff_g2l_out.b=0;

	return true;
}

void CoarseTracker::debugPlotIDepthMap(float* minID_pt, float* maxID_pt, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;

	int lvl = 0;

	{
		std::vector<float> allID;
		for(int i=0;i<h[lvl]*w[lvl];i++) {
			if(idepth[lvl][i] > 0)
				allID.push_back(idepth[lvl][i]);
		}
		std::sort(allID.begin(), allID.end());
		int n = allID.size()-1;

		float minID_new = allID[(int)(n*0.05)];
		float maxID_new = allID[(int)(n*0.95)];

		float minID, maxID;
		minID = minID_new;
		maxID = maxID_new;
		if(minID_pt!=0 && maxID_pt!=0) {
			if(*minID_pt < 0 || *maxID_pt < 0) {
				*maxID_pt = maxID;
				*minID_pt = minID;
			} else {
				// slowly adapt: change by maximum 10% of old span.
				float maxChange = 0.3*(*maxID_pt - *minID_pt);

				if(minID < *minID_pt - maxChange)
					minID = *minID_pt - maxChange;
				if(minID > *minID_pt + maxChange)
					minID = *minID_pt + maxChange;

				if(maxID < *maxID_pt - maxChange)
					maxID = *maxID_pt - maxChange;
				if(maxID > *maxID_pt + maxChange)
					maxID = *maxID_pt + maxChange;

				*maxID_pt = maxID;
				*minID_pt = minID;
			}
		}

		MinimalImageB3 mf(w[lvl], h[lvl]);
		mf.setBlack();
		for(int i=0;i<h[lvl]*w[lvl];i++)
		{
			int c = lastRef->dIp[lvl][i][0]*0.9f;
			if(c>255) c=255;
			mf.at(i) = Vec3b(c,c,c);
		}
		int wl = w[lvl];
		for(int y=3;y<h[lvl]-3;y++)
			for(int x=3;x<wl-3;x++) {
				int idx=x+y*wl;
				float sid=0, nid=0;
				float* bp = idepth[lvl]+idx;

				if(bp[0] > 0) {sid+=bp[0]; nid++;}
				if(bp[1] > 0) {sid+=bp[1]; nid++;}
				if(bp[-1] > 0) {sid+=bp[-1]; nid++;}
				if(bp[wl] > 0) {sid+=bp[wl]; nid++;}
				if(bp[-wl] > 0) {sid+=bp[-wl]; nid++;}

				if(bp[0] > 0 || nid >= 3) {
					float id = ((sid / nid)-minID) / ((maxID-minID));
					mf.setPixelCirc(x,y,makeJet3B(id));
					//mf.at(idx) = makeJet3B(id);
				}
			}
        //IOWrap::displayImage("coarseDepth LVL0", &mf, false);

        for(IOWrap::Output3DWrapper* ow : wraps)
            ow->pushDepthImage(&mf);

		if(debugSaveImages) {
			char buf[1000];
			snprintf(buf, 1000, "images_out/predicted_%05d_%05d.png", lastRef->shell->id, refFrameID);
			IOWrap::writeImage(buf,&mf);
		}

	}
}

void CoarseTracker::debugPlotIDepthMapFloat(std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    if(w[1] == 0) return;
    int lvl = 0;
    MinimalImageF mim(w[lvl], h[lvl], idepth[lvl]);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImageFloat(&mim, lastRef);
}

CoarseDistanceMap::CoarseDistanceMap(int ww, int hh)
{
	fwdWarpedIDDistFinal = new float[ww*hh/4];

	bfsList1 = new Eigen::Vector2i[ww*hh/4];
	bfsList2 = new Eigen::Vector2i[ww*hh/4];

	int fac = 1 << (pyrLevelsUsed-1);

	coarseProjectionGrid = new PointFrameResidual*[2048*(ww*hh/(fac*fac))];
	coarseProjectionGridNum = new int[ww*hh/(fac*fac)];

	w[0]=h[0]=0;
}
CoarseDistanceMap::~CoarseDistanceMap()
{
	delete[] fwdWarpedIDDistFinal;
	delete[] bfsList1;
	delete[] bfsList2;
	delete[] coarseProjectionGrid;
	delete[] coarseProjectionGridNum;
}

void CoarseDistanceMap::makeDistanceMap(
		std::vector<FrameHessian*> frameHessians,
		FrameHessian* frame)
{
	int w1 = w[1];
	int h1 = h[1];
	int wh1 = w1*h1;
	for(int i=0;i<wh1;i++)
		fwdWarpedIDDistFinal[i] = 1000;

	// make coarse tracking templates for latstRef.
	int numItems = 0;

	for(FrameHessian* fh : frameHessians) {
		if(frame == fh) continue;

		SE3 fhToNew = frame->PRE_worldToCam * fh->PRE_camToWorld;
		Mat33f KRKi = (K[1] * fhToNew.rotationMatrix().cast<float>() * Ki[0]);
		Vec3f Kt = (K[1] * fhToNew.translation().cast<float>());

		for(PointHessian* ph : fh->pointHessians) {
			assert(ph->status == PointHessian::ACTIVE);
			Vec3f ptp = KRKi * Vec3f(ph->u, ph->v, 1) + Kt*ph->idepth_scaled;
			int u = ptp[0] / ptp[2] + 0.5f;
			int v = ptp[1] / ptp[2] + 0.5f;
			if(!(u > 0 && v > 0 && u < w[1] && v < h[1])) continue;
			fwdWarpedIDDistFinal[u+w1*v]=0;
			bfsList1[numItems] = Eigen::Vector2i(u,v);
			numItems++;
		}
	}

	growDistBFS(numItems);
}




void CoarseDistanceMap::makeInlierVotes(std::vector<FrameHessian*> frameHessians)
{

}



void CoarseDistanceMap::growDistBFS(int bfsNum)
{
	assert(w[0] != 0);
	int w1 = w[1], h1 = h[1];
	for(int k=1;k<40;k++)
	{
		int bfsNum2 = bfsNum;
		std::swap<Eigen::Vector2i*>(bfsList1,bfsList2);
		bfsNum=0;

		if(k%2==0)
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}
			}
		}
		else
		{
			for(int i=0;i<bfsNum2;i++)
			{
				int x = bfsList2[i][0];
				int y = bfsList2[i][1];
				if(x==0 || y== 0 || x==w1-1 || y==h1-1) continue;
				int idx = x + y * w1;

				if(fwdWarpedIDDistFinal[idx+1] > k)
				{
					fwdWarpedIDDistFinal[idx+1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1] > k)
				{
					fwdWarpedIDDistFinal[idx-1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x,y-1); bfsNum++;
				}

				if(fwdWarpedIDDistFinal[idx+1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1+w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1+w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y+1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx-1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx-1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x-1,y-1); bfsNum++;
				}
				if(fwdWarpedIDDistFinal[idx+1-w1] > k)
				{
					fwdWarpedIDDistFinal[idx+1-w1] = k;
					bfsList1[bfsNum] = Eigen::Vector2i(x+1,y-1); bfsNum++;
				}
			}
		}
	}
}


void CoarseDistanceMap::addIntoDistFinal(int u, int v)
{
	if(w[0] == 0) return;
	bfsList1[0] = Eigen::Vector2i(u,v);
	fwdWarpedIDDistFinal[u+w[1]*v] = 0;
	growDistBFS(1);
}



void CoarseDistanceMap::makeK(CalibHessian* HCalib)
{
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}
//    / 打印函数执行时间
    template <typename FuncT>
    void evaluate_and_call(FuncT func, const std::string &func_name = "",
                           int times = 10) {
        double total_time = 0;
        for (int i = 0; i < times; ++i) {
            auto t1 = std::chrono::steady_clock::now();
            func();
            auto t2 = std::chrono::steady_clock::now();
            total_time +=
                    std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() *
                    1000;
        }

        std::cout << "方法 " << func_name
                  << " 平均调用时间/次数: " << total_time / times << "/" << times
                  << " 毫秒." << std::endl;
    }

Vec6 CoarseTracker::calcRes_MT(int lvl, const SE3 &refToNew, AffLight aff_g2l, float cutoffTH)
{
	float E = 0;
	int numTermsInE = 0;
	int numTermsInWarped = 0;
	int numSaturated=0;

	int wl = w[lvl];
	int hl = h[lvl];
	Eigen::Vector3f* dINewl = newFrame -> dIp[lvl];
	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];

	Mat33f RKi = (refToNew.rotationMatrix().cast<float>() * Ki[lvl]);
	Vec3f t = (refToNew.translation()).cast<float>();
	Vec2f affLL = AffLight::fromToVecExposure(lastRef->ab_exposure,
                                              newFrame->ab_exposure,
                                              lastRef_aff_g2l, aff_g2l).cast<float>();
	float sumSquaredShiftT=0;
	float sumSquaredShiftRT=0;
	float sumSquaredShiftNum=0;

	float maxEnergy = 2*setting_huberTH*cutoffTH-setting_huberTH*setting_huberTH;	// energy for r=setting_coarseCutoffTH.

    MinimalImageB3* resImage = 0;
	if(debugPlot) {
		resImage = new MinimalImageB3(wl,hl);
		resImage->setConst(Vec3b(0,128,0));
	}

    // 跟踪器里的金字塔数据
	int nl = pc_n[lvl];
	float* lpc_u = pc_u[lvl];
	float* lpc_v = pc_v[lvl];
	float* lpc_idepth = pc_idepth[lvl];
	float* lpc_color = pc_color[lvl];

    // 这里可以并行化
    std::mutex M;
    std::mutex m;
//    boost::mutex m;
//    boost::mutex M;
    std::vector <int > indices;

    auto t1 = std::chrono::steady_clock::now();
    int k = 0;
    while (k < nl) {
        if (!indices.empty())
            indices.clear();
        for (int i = 0; i < 10000 && k < nl; i++)
            indices.push_back(k++);

        std::for_each(std::execution::seq, indices.begin(), indices.end(),
                      [&](auto &i) {
//	for(int i = 0;i<nl;i++) {
//    for(int i = min;i < max;i++) {
                          float id = lpc_idepth[i];
                          float x = lpc_u[i];
                          float y = lpc_v[i];

                          // 这里就是warping: (x, y, id) -> (Ku, Kv, new_id)
                          Vec3f pt = RKi * Vec3f(x, y, 1) + t * id;
                          float u = pt[0] / pt[2];
                          float v = pt[1] / pt[2];
                          float Ku = fxl * u + cxl;
                          float Kv = fyl * v + cyl;
                          float new_idepth = id / pt[2];

                          // 第0层，每32个点
                          if (lvl == 0 && i % 32 == 0) {
                              // translation only (positive)
                              Vec3f ptT = Ki[lvl] * Vec3f(x, y, 1) + t * id;
                              float uT = ptT[0] / ptT[2];
                              float vT = ptT[1] / ptT[2];
                              float KuT = fxl * uT + cxl;
                              float KvT = fyl * vT + cyl;

                              // translation only (negative)
                              Vec3f ptT2 = Ki[lvl] * Vec3f(x, y, 1) - t * id;
                              float uT2 = ptT2[0] / ptT2[2];
                              float vT2 = ptT2[1] / ptT2[2];
                              float KuT2 = fxl * uT2 + cxl;
                              float KvT2 = fyl * vT2 + cyl;

                              //translation and rotation (negative)
                              Vec3f pt3 = RKi * Vec3f(x, y, 1) - t * id;
                              float u3 = pt3[0] / pt3[2];
                              float v3 = pt3[1] / pt3[2];
                              float Ku3 = fxl * u3 + cxl;
                              float Kv3 = fyl * v3 + cyl;

                              //translation and rotation (positive)
                              //already have it.
                              {
//                                  boost::unique_lock<boost::mutex> crlock(M);
                                  std::lock_guard<std::mutex> guard(M);

                                  sumSquaredShiftT += (KuT - x) * (KuT - x) + (KvT - y) * (KvT - y);
                                  sumSquaredShiftT += (KuT2 - x) * (KuT2 - x) + (KvT2 - y) * (KvT2 - y);
                                  sumSquaredShiftRT += (Ku - x) * (Ku - x) + (Kv - y) * (Kv - y);
                                  sumSquaredShiftRT += (Ku3 - x) * (Ku3 - x) + (Kv3 - y) * (Kv3 - y);
                                  sumSquaredShiftNum += 2;
                              }
                          }

                          // Ku, Kv在图像边沿，或逆深度<=0，则抛弃该点
                          if (!(Ku > 2 && Kv > 2 && Ku < wl - 3 && Kv < hl - 3 && new_idepth > 0))
                              return;


                          float refColor = lpc_color[i];
                          Vec3f hitColor = getInterpolatedElement33(dINewl, Ku, Kv, wl);
                          if (!std::isfinite((float) hitColor[0]))
                              return;
                          float residual = hitColor[0] - (float) (affLL[0] * refColor + affLL[1]);
                          // hw是 max{阈值/残差, 1}
                          float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);

                          // 如果outlier，即残差太大了，则用maxEnergy表示
                          if (fabs(residual) > cutoffTH) {
//			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(0,0,255));

                              {
//                                  boost::unique_lock<boost::mutex> crlock(m);
                                  std::lock_guard<std::mutex> guard(m);
                                  numTermsInE++;
                                  E += maxEnergy;
                                  numSaturated++;
                              }
                          } else {
//			if(debugPlot) resImage->setPixel4(lpc_u[i], lpc_v[i], Vec3b(residual+128,residual+128,residual+128));
                              // 其实就是残差的平方，但是做了一些处理
                              float tempE = hw * residual * residual * (2 - hw);
                              // 需要改用boost的互斥鎖
                              {
//                                  boost::unique_lock<boost::mutex> crlock(m);
                                  std::lock_guard<std::mutex> guard(m);
                                  numTermsInE++;
                                  E += tempE;
                                  numTermsInWarped++;
                              }

                              buf_warped_idepth[numTermsInWarped - 1] = new_idepth;
                              buf_warped_u[numTermsInWarped - 1] = u;
                              buf_warped_v[numTermsInWarped - 1] = v;
                              buf_warped_dx[numTermsInWarped - 1] = hitColor[1];
                              buf_warped_dy[numTermsInWarped - 1] = hitColor[2];
                              buf_warped_residual[numTermsInWarped - 1] = residual;
                              buf_warped_weight[numTermsInWarped - 1] = hw;
                              buf_warped_refColor[numTermsInWarped - 1] = lpc_color[i];
                          }
                      });
    }

    auto t2 = std::chrono::steady_clock::now();
    calcRes_MT_total_time +=
                    std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count() *
                    1000;
    // 填充0, 对齐
	while(numTermsInWarped%4!=0) {
		buf_warped_idepth[numTermsInWarped] = 0;
		buf_warped_u[numTermsInWarped] = 0;
		buf_warped_v[numTermsInWarped] = 0;
		buf_warped_dx[numTermsInWarped] = 0;
		buf_warped_dy[numTermsInWarped] = 0;
		buf_warped_residual[numTermsInWarped] = 0;
		buf_warped_weight[numTermsInWarped] = 0;
		buf_warped_refColor[numTermsInWarped] = 0;
		numTermsInWarped++;
	}
	buf_warped_n = numTermsInWarped;

//	if(debugPlot) {
//		IOWrap::displayImage("RES", resImage, false);
//		IOWrap::waitKey(0); // ruan kefeng
//		delete resImage;
//	}

    // 关心的一般是平均残差rs[0]/rs[1]。 rs[5]用来表示outlier占比，如果过高则放大阈值重新计算
	Vec6 rs;
	rs[0] = E;
	rs[1] = numTermsInE;
	rs[2] = sumSquaredShiftT/(sumSquaredShiftNum+0.1);
	rs[3] = 0;
	rs[4] = sumSquaredShiftRT/(sumSquaredShiftNum+0.1);
	rs[5] = numSaturated / (float)numTermsInE;

	return rs;
}
// 进行数据预处理。制作投影面。
// 投影面是一个金字塔，底层照抄ph->(u, v, idepth)，但当点重叠时，会取平均
// 即各点的平均逆深度，金字塔上层也相当于低层的四个点重叠，也是取平均逆深度，并且做了一些鲁棒化处理
// 即各个点叠加，会被叠加成一个（u, v, 各点平均逆深度，color）的点
// pointHessian里的是成熟点
    void CoarseTracker::normalize_pyramid()
    {
        for(int lvl=1; lvl<pyrLevelsUsed; lvl++) {
            // 直接遍历整个长和宽，似乎把没有选中的像素点的逆深度预先设置为0了
            for(int y=0;y<h[lvl];y++)
                for(int x=0;x<w[lvl];x++) {
                    int bidx = 2*x  + 2*y*w[lvl - 1];
                    // 逆深度，直接取低层的对应的四个点的逆深度，取和
                    // 其实我们关心的是幅照在像素点上的累加
                    // 低层的不同点汇聚为当前层的同一个点，并且逆深度直接相加，
                    // 因为计数也直接累加了，所以不影响最后求平均值
                    // bidx是四个点中左上角的点
                    idepth[lvl][x + y*w[lvl]] = 		idepth[lvl - 1][bidx] +
                                                       idepth[lvl - 1][bidx+1] +
                                                       idepth[lvl - 1][bidx+w[lvl - 1]] +
                                                       idepth[lvl - 1][bidx+w[lvl - 1]+1];

                    weightSums[lvl][x + y*w[lvl]] = 	weightSums[lvl - 1][bidx] +
                                                       weightSums[lvl - 1][bidx+1] +
                                                       weightSums[lvl - 1][bidx+w[lvl - 1]] +
                                                       weightSums[lvl - 1][bidx+w[lvl - 1]+1];
                }
        }

        // dilate idepth by 1.
        // 将投影计数为零的像素点四周取了四个点，看看它们是否有，如果有，则取其平均赋给该点
        // 应该是为了增加鲁棒性
        // 这个操作，相当于“扩招”了，每个点的邻居也被邀请加入，取平均逆深度作为新点逆深度,
        // 取邻居自有的color（来自lastRef[lvl]->dIp）
        // 所以最后点的数量变多了，详见k的计数如下，5519 + 18506 = 24025

//    INITIALIZE FROM INITIALIZER (5519 pts)!
//    ...........allFrameHistory.size() == 2...............
//    ......setCTrefForFirstFrame...size[1]...lastRef[0]....
//    ..........makeCoarseDepthForFirstFrame..lastRef[0]........
//    .......lastRef -> pointHessians.size(){5519}....
//    ........k = [18506].........
//    ........k = [12588].........
//    pc_n[0] is 24025
//    pc_n[1] is 17885
//    pc_n[2] is 8441
//    pc_n[3] is 3421

        for(int lvl=0; lvl<2; lvl++) {
            int wh = w[lvl]*h[lvl]-w[lvl]; //  (h - 1) * w
            int wl = w[lvl];                // 1 * w
            memcpy(weightSums_bak[lvl], weightSums[lvl], w[lvl]*h[lvl]*sizeof(float));
            // dont need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            // 上下各减一行
            int k = 0;
            for(int i=wl;i<wh;i++) {
                // 就是为0, 浮点数不合适直接用==
                if(weightSums_bak[lvl][i] <= 0) {
                    float sum=0, num=0, numn=0;
                    // 右对下的点
                    if(weightSums_bak[lvl][i+1+wl] > 0) {
                        sum += idepth[lvl][i+1+wl];
                        num += weightSums_bak[lvl][i+1+wl];
                        numn++;
                    }
                    // 左对上的点
                    if(weightSums_bak[lvl][i-1-wl] > 0) {
                        sum += idepth[lvl][i-1-wl];
                        num+=weightSums_bak[lvl][i-1-wl];
                        numn++;
                    }
                    // 左对下的点
                    if(weightSums_bak[lvl][i+wl-1] > 0) {
                        sum += idepth[lvl][i+wl-1];
                        num+=weightSums_bak[lvl][i+wl-1];
                        numn++;
                    }
                    // 右对上的点
                    if(weightSums_bak[lvl][i-wl+1] > 0) {
                        sum += idepth[lvl][i-wl+1];
                        num+=weightSums_bak[lvl][i-wl+1];
                        numn++;
                    }
                    if(numn>0) {idepth[lvl][i] = sum/numn; weightSums[lvl][i] = num/numn;k++;}
                }
            }
//            printf("........k = [%d].........\n", k);
        }

        // dilate idepth by 1 (2 on lower levels).
        // 同上为了增加鲁棒性，只是这次换了正上正下正左正右的四个点，为什么有这个区别，估计是作者实验的结果
        for(int lvl=2; lvl<pyrLevelsUsed; lvl++) {
            int wh = w[lvl]*h[lvl]-w[lvl];
            int wl = w[lvl];
            float* weightSumsl = weightSums[lvl];
            float* weightSumsl_bak = weightSums_bak[lvl];
            memcpy(weightSumsl_bak, weightSumsl, w[lvl]*h[lvl]*sizeof(float));
            float* idepthl = idepth[lvl];	// dotnt need to make a temp copy of depth, since I only
            // read values with weightSumsl>0, and write ones with weightSumsl<=0.
            for(int i=w[lvl];i<wh;i++) {
                if(weightSumsl_bak[i] <= 0) {
                    float sum=0, num=0, numn=0;
                    if(weightSumsl_bak[i+1] > 0) { sum += idepthl[i+1]; num+=weightSumsl_bak[i+1]; numn++;}
                    if(weightSumsl_bak[i-1] > 0) { sum += idepthl[i-1]; num+=weightSumsl_bak[i-1]; numn++;}
                    if(weightSumsl_bak[i+wl] > 0) { sum += idepthl[i+wl]; num+=weightSumsl_bak[i+wl]; numn++;}
                    if(weightSumsl_bak[i-wl] > 0) { sum += idepthl[i-wl]; num+=weightSumsl_bak[i-wl]; numn++;}
                    if(numn>0) {idepthl[i] = sum/numn; weightSumsl[i] = num/numn;}
                }
            }
        }

        // normalize idepths and weights.
        // 对逆深度的平滑，幅照直接取lastRef->dIp，dIp存储金字塔对应像素的color，来自于ph->makeImage()
        // 即各个点叠加，则会被叠加成一个（u, v, 各点平均逆深度，color）的点
        for(int lvl=0; lvl< pyrLevelsUsed; lvl++) {
            Eigen::Vector3f* dIRefl = lastRef->dIp[lvl];
            int lpc_n=0;
            for(int y=2;y<h[lvl]-2;y++)
                for(int x=2;x<w[lvl]-2;x++) {
                    int i = x+y*w[lvl];
                    if(weightSums[lvl][i] > 0) {
                        // weightSums其实是计数，所以其实是求平均
                        // 平均逆深度
                        idepth[lvl][i] /= weightSums[lvl][i];

                        pc_u[lvl][lpc_n] = x;
                        pc_v[lvl][lpc_n] = y;
                        pc_idepth[lvl][lpc_n] = idepth[lvl][i];
                        pc_color[lvl][lpc_n] = dIRefl[i][0];

                        if(!std::isfinite(pc_color[lvl][lpc_n]) || !(idepth[lvl][i]>0)) {
                            idepth[lvl][i] = -1;
                            continue;	// just skip if something is wrong.
                        }
                        lpc_n++;
                    } else
                        idepth[lvl][i] = -1;

                    // 至此，weightSums的在每个像素点都变成了1，即完成了平均化
                    // 而没有逆深度的像素点的逆深度被设置成了-1
                    // 这些有正逆深度的都是活跃的跟踪点
                    weightSums[lvl][i] = 1;
                }
            // 每一层的点的(u, v, idepth, color)留在跟踪器的数据结构里，同时也收集到了pc_xxx数据结构里
            pc_n[lvl] = lpc_n;
//            printf("pc_n[%d] is %d \n", lvl, lpc_n);
        }
    }
}
