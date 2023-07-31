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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "FullSystem/ImmaturePoint.h"
#include "util/nanoflann.h"



#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso {

    CoarseInitializer::CoarseInitializer(int ww, int hh) : thisToNext_aff(0, 0), thisToNext(SE3()) {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            points[lvl] = 0;
            numPoints[lvl] = 0;
        }

        JbBuffer = new Vec10f[ww * hh];
        JbBuffer_new = new Vec10f[ww * hh];

        frameID = -1;
//	fixAffine=true;
        printDebug = false;

        wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
        wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
        wM.diagonal()[6] = SCALE_A;
        wM.diagonal()[7] = SCALE_B;
    }

    CoarseInitializer::~CoarseInitializer() {
        for (int lvl = 0; lvl < pyrLevelsUsed; lvl++) {
            if (points[lvl] != 0) delete[] points[lvl];
        }

        delete[] JbBuffer;
        delete[] JbBuffer_new;
    }

// 不做金字塔，只做原始大小的，双目
    void CoarseInitializer::setFirstStereo(CalibHessian *HCalib,
                                           FrameHessian *newFrameHessian)
    {
        makeK(HCalib);

        this -> firstFrame = newFrameHessian;

        PixelSelector sel(w[0], h[0]);
        float *statusMap = new float[w[0] * h[0]];
        bool *statusMapB = new bool[w[0] * h[0]];
        float densities[] = {0.03, 0.05, 0.15, 0.5, 1};
        idepth[0] = new float[w[0] * h[0]]{0};
        sel.currentPotential = 3;
        int npts;
        // 通过一些玄学选点
        npts = sel.makeMaps(firstFrame, statusMap, densities[0] * w[0] * h[0],
                            1, false, 2);

        if (points[0] != NULL) delete[] points[0];
        points[0] = new Pnt[npts];

        // set idepth map to initially 1 everywhere.
        Pnt *pl = points[0];
        int nl = 0;
        // patternPadding == 2，用来将边框的点排除掉
        for (int y = patternPadding + 1; y < h[0] - patternPadding - 2; y++) {
            for (int x = patternPadding + 1; x < w[0] - patternPadding - 2; x++) {
                // 对选取的每个点建模为未成熟点，然后traceStereo
                // 最后结果放到初始器的points[0]里
                if (statusMap[x + y * w[0]] != 0) {
                    ImmaturePoint *pt = new ImmaturePoint(x, y, firstFrame,
                                                          statusMap[x + y * w[0]], HCalib);
                    pt->u_stereo = pt->u;
                    pt->v_stereo = pt->v;
                    pt->idepth_min_stereo = 0;
                    pt->idepth_max_stereo = NAN;
//                    printf("..........pt->traceStereo....\n");
                    ImmaturePointStatus stat = pt->traceStereo(firstFrame->frame_right,
                                                               HCalib, true);
                    if (stat == ImmaturePointStatus::IPS_GOOD) {
                        pl[nl].u = x;
                        pl[nl].v = y;
                        pl[nl].idepth = pt->idepth_stereo;
                        pl[nl].iR = pt->idepth_stereo;
                        pl[nl].energy.setZero();
                        pl[nl].my_type = statusMap[x + y * w[0]];
                        idepth[0][x + w[0] * y] = pt->idepth_stereo;
                        nl++;
                        assert(nl <= npts);
                    }
                    delete pt;
                }
            }
        }
        numPoints[0]=nl;
// 		LOG(INFO)<<"lvl: "<<lvl<<" nl:"<<nl;
        std::cout<<"lvl: "<<0<<" nl:"<<nl << std::endl;
        delete[] statusMap;
        delete[] statusMapB;

        thisToNext=SE3();
        frameID = 0;
    }

    void CoarseInitializer::makeK(CalibHessian* HCalib)
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

}

