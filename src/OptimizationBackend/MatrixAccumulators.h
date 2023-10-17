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

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

template<int i, int j>
class AccumulatorXX {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  //! 设计成A, A1k, A1m三级纯粹是为了并行。好像也不一定，也许是为了数值稳定性，保持频繁的加法出现在数值较小的情况下，
  //! 不懂数值稳定性这块，只是猜测。
  Eigen::Matrix<float,i,j> A;
  Eigen::Matrix<float,i,j> A1k;
  Eigen::Matrix<float,i,j> A1m;
  size_t num;

  inline void initialize()
  {
    A.setZero();
    A1k.setZero();
    A1m.setZero();
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish()
  {
	shiftUp(true);
    num = numIn1 + numIn1k + numIn1m;
  }

  // L * w * R^T
  //! shape为<i, j>
  inline void update(const Eigen::Matrix<float,i,1> &L, const Eigen::Matrix<float,j,1> &R, float w)
  {
	  A += w*L*R.transpose();
	  numIn1++;
//      num++;
	  shiftUp(false);
  }

private:
  float numIn1, numIn1k, numIn1m;

  //! 最后一次使用force，强制汇总所有的数据到A1m
  void shiftUp(bool force)
  {
	  if(numIn1 > 1000 || force) {
		  A1k += A;
		  A.setZero();
		  numIn1k+=numIn1;
		  numIn1=0;
	  }
	  if(numIn1k > 1000 || force) {
		  A1m += A1k;
		  A1k.setZero();
		  numIn1m+=numIn1k;
		  numIn1k=0;
	  }
  }
};

template<int i>
class AccumulatorX {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Eigen::Matrix<float,i,1> A;
  Eigen::Matrix<float,i,1> A1k;
  Eigen::Matrix<float,i,1> A1m;
  size_t num;

  inline void initialize()
  {
    A.setZero();
    A1k.setZero();
    A1m.setZero();
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish()
  {
	shiftUp(true);
	num = numIn1+numIn1k+numIn1m;
  }

  inline void update(const Eigen::Matrix<float,i,1> &L, float w)
  {
	  A += w*L;
	  numIn1++;
//      num++;
      //! 去掉shiftUp试试
      //! 确实去掉了也无所谓
	  shiftUp(false);
  }

private:
  float numIn1, numIn1k, numIn1m;

  void shiftUp(bool force)
  {
      // 每1000个SSEData搜集完成，就累加到SSEData1k （一千个数据）
	  if(numIn1 > 1000 || force) {
		  A1k += A;
		  A.setZero();
		  numIn1k+=numIn1;
		  numIn1=0;
	  }
      // 每1000个SSEData1k搜集完成，就累加到SSEData1m （一百万个数据）
	  if(numIn1k > 1000 || force) {
		  A1m += A1k;
		  A1k.setZero();
		  numIn1m+=numIn1k;
		  numIn1k=0;
	  }
  }
};

/*
 * computes the outer sum of 10x2 matrices, weighted with a 2x2 matrix:
 * 			H = [x y] * [a b; b c] * [x y]^T
 * (assuming x,y are column-vectors).
 * numerically robust to large sums.
 */
class AccumulatorApprox {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Mat1313f H;
    size_t num;

  inline void initialize()
  {
      //! 这可能是为了数值稳定性，可以看TAOCP第2卷
	memset(Data,0, sizeof(float)*60);
	memset(Data1k,0, sizeof(float)*60);
	memset(Data1m,0, sizeof(float)*60);

	memset(TopRight_Data,0, sizeof(float)*32);
	memset(TopRight_Data1k,0, sizeof(float)*32);
	memset(TopRight_Data1m,0, sizeof(float)*32);

	memset(BotRight_Data,0, sizeof(float)*8);
	memset(BotRight_Data1k,0, sizeof(float)*8);
	memset(BotRight_Data1m,0, sizeof(float)*8);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

inline void finish()
{
	H.setZero();
	shiftUp(true);
	assert(numIn1==0);
	assert(numIn1k==0);

	int idx=0;
	for(int r=0;r<10;r++)
		for(int c=r;c<10;c++) {
			H(r,c) = H(c,r) = Data1m[idx];
			idx++;
		}

	idx=0;
	for(int r=0;r<10;r++)
		for(int c=0; c<3;c++) {
			H(r,c+10) = H(c+10,r) = TopRight_Data1m[idx];
			idx++;
		}

	H(10,10) 			= BotRight_Data1m[0];
	H(10,11) = H(11,10) = BotRight_Data1m[1];
	H(10,12) = H(12,10) = BotRight_Data1m[2];
	H(11,11) 			= BotRight_Data1m[3];
	H(11,12) = H(12,11) = BotRight_Data1m[4];
	H(12,12) 			= BotRight_Data1m[5];

	num = numIn1 + numIn1k + numIn1m;
}


/*
 * same as other method, just that x/y are composed of two parts,
 * the first 4 elements are in x4/y4, the last 6 in x6/y6.
 */
  inline void update(
		  const float* const x4,
		  const float* const x6,
		  const float* const y4,
		  const float* const y6,
		  const float a,
		  const float b,
		  const float c)
  {
	  Data[0] += a*x4[0]*x4[0] + c*y4[0]*y4[0] +  b*(x4[0]*y4[0] + y4[0]*x4[0]);
	  Data[1] += a*x4[1]*x4[0] + c*y4[1]*y4[0] +  b*(x4[1]*y4[0] + y4[1]*x4[0]);
	  Data[2] += a*x4[2]*x4[0] + c*y4[2]*y4[0] +  b*(x4[2]*y4[0] + y4[2]*x4[0]);
	  Data[3] += a*x4[3]*x4[0] + c*y4[3]*y4[0] +  b*(x4[3]*y4[0] + y4[3]*x4[0]);
	  Data[4] += a*x6[0]*x4[0] + c*y6[0]*y4[0] +  b*(x6[0]*y4[0] + y6[0]*x4[0]);
	  Data[5] += a*x6[1]*x4[0] + c*y6[1]*y4[0] +  b*(x6[1]*y4[0] + y6[1]*x4[0]);
	  Data[6] += a*x6[2]*x4[0] + c*y6[2]*y4[0] +  b*(x6[2]*y4[0] + y6[2]*x4[0]);
	  Data[7] += a*x6[3]*x4[0] + c*y6[3]*y4[0] +  b*(x6[3]*y4[0] + y6[3]*x4[0]);
	  Data[8] += a*x6[4]*x4[0] + c*y6[4]*y4[0] +  b*(x6[4]*y4[0] + y6[4]*x4[0]);
	  Data[9] += a*x6[5]*x4[0] + c*y6[5]*y4[0] +  b*(x6[5]*y4[0] + y6[5]*x4[0]);

	  Data[10] += a*x4[1]*x4[1] + c*y4[1]*y4[1] +  b*(x4[1]*y4[1] + y4[1]*x4[1]);
	  Data[11] += a*x4[2]*x4[1] + c*y4[2]*y4[1] +  b*(x4[2]*y4[1] + y4[2]*x4[1]);
	  Data[12] += a*x4[3]*x4[1] + c*y4[3]*y4[1] +  b*(x4[3]*y4[1] + y4[3]*x4[1]);
	  Data[13] += a*x6[0]*x4[1] + c*y6[0]*y4[1] +  b*(x6[0]*y4[1] + y6[0]*x4[1]);
	  Data[14] += a*x6[1]*x4[1] + c*y6[1]*y4[1] +  b*(x6[1]*y4[1] + y6[1]*x4[1]);
	  Data[15] += a*x6[2]*x4[1] + c*y6[2]*y4[1] +  b*(x6[2]*y4[1] + y6[2]*x4[1]);
	  Data[16] += a*x6[3]*x4[1] + c*y6[3]*y4[1] +  b*(x6[3]*y4[1] + y6[3]*x4[1]);
	  Data[17] += a*x6[4]*x4[1] + c*y6[4]*y4[1] +  b*(x6[4]*y4[1] + y6[4]*x4[1]);
	  Data[18] += a*x6[5]*x4[1] + c*y6[5]*y4[1] +  b*(x6[5]*y4[1] + y6[5]*x4[1]);

	  Data[19] += a*x4[2]*x4[2] + c*y4[2]*y4[2] +  b*(x4[2]*y4[2] + y4[2]*x4[2]);
	  Data[20] += a*x4[3]*x4[2] + c*y4[3]*y4[2] +  b*(x4[3]*y4[2] + y4[3]*x4[2]);
	  Data[21] += a*x6[0]*x4[2] + c*y6[0]*y4[2] +  b*(x6[0]*y4[2] + y6[0]*x4[2]);
	  Data[22] += a*x6[1]*x4[2] + c*y6[1]*y4[2] +  b*(x6[1]*y4[2] + y6[1]*x4[2]);
	  Data[23] += a*x6[2]*x4[2] + c*y6[2]*y4[2] +  b*(x6[2]*y4[2] + y6[2]*x4[2]);
	  Data[24] += a*x6[3]*x4[2] + c*y6[3]*y4[2] +  b*(x6[3]*y4[2] + y6[3]*x4[2]);
	  Data[25] += a*x6[4]*x4[2] + c*y6[4]*y4[2] +  b*(x6[4]*y4[2] + y6[4]*x4[2]);
	  Data[26] += a*x6[5]*x4[2] + c*y6[5]*y4[2] +  b*(x6[5]*y4[2] + y6[5]*x4[2]);

	  Data[27] += a*x4[3]*x4[3] + c*y4[3]*y4[3] +  b*(x4[3]*y4[3] + y4[3]*x4[3]);
	  Data[28] += a*x6[0]*x4[3] + c*y6[0]*y4[3] +  b*(x6[0]*y4[3] + y6[0]*x4[3]);
	  Data[29] += a*x6[1]*x4[3] + c*y6[1]*y4[3] +  b*(x6[1]*y4[3] + y6[1]*x4[3]);
	  Data[30] += a*x6[2]*x4[3] + c*y6[2]*y4[3] +  b*(x6[2]*y4[3] + y6[2]*x4[3]);
	  Data[31] += a*x6[3]*x4[3] + c*y6[3]*y4[3] +  b*(x6[3]*y4[3] + y6[3]*x4[3]);
	  Data[32] += a*x6[4]*x4[3] + c*y6[4]*y4[3] +  b*(x6[4]*y4[3] + y6[4]*x4[3]);
	  Data[33] += a*x6[5]*x4[3] + c*y6[5]*y4[3] +  b*(x6[5]*y4[3] + y6[5]*x4[3]);

	  Data[34] += a*x6[0]*x6[0] + c*y6[0]*y6[0] +  b*(x6[0]*y6[0] + y6[0]*x6[0]);
	  Data[35] += a*x6[1]*x6[0] + c*y6[1]*y6[0] +  b*(x6[1]*y6[0] + y6[1]*x6[0]);
	  Data[36] += a*x6[2]*x6[0] + c*y6[2]*y6[0] +  b*(x6[2]*y6[0] + y6[2]*x6[0]);
	  Data[37] += a*x6[3]*x6[0] + c*y6[3]*y6[0] +  b*(x6[3]*y6[0] + y6[3]*x6[0]);
	  Data[38] += a*x6[4]*x6[0] + c*y6[4]*y6[0] +  b*(x6[4]*y6[0] + y6[4]*x6[0]);
	  Data[39] += a*x6[5]*x6[0] + c*y6[5]*y6[0] +  b*(x6[5]*y6[0] + y6[5]*x6[0]);

	  Data[40] += a*x6[1]*x6[1] + c*y6[1]*y6[1] +  b*(x6[1]*y6[1] + y6[1]*x6[1]);
	  Data[41] += a*x6[2]*x6[1] + c*y6[2]*y6[1] +  b*(x6[2]*y6[1] + y6[2]*x6[1]);
	  Data[42] += a*x6[3]*x6[1] + c*y6[3]*y6[1] +  b*(x6[3]*y6[1] + y6[3]*x6[1]);
	  Data[43] += a*x6[4]*x6[1] + c*y6[4]*y6[1] +  b*(x6[4]*y6[1] + y6[4]*x6[1]);
	  Data[44] += a*x6[5]*x6[1] + c*y6[5]*y6[1] +  b*(x6[5]*y6[1] + y6[5]*x6[1]);

	  Data[45] += a*x6[2]*x6[2] + c*y6[2]*y6[2] +  b*(x6[2]*y6[2] + y6[2]*x6[2]);
	  Data[46] += a*x6[3]*x6[2] + c*y6[3]*y6[2] +  b*(x6[3]*y6[2] + y6[3]*x6[2]);
	  Data[47] += a*x6[4]*x6[2] + c*y6[4]*y6[2] +  b*(x6[4]*y6[2] + y6[4]*x6[2]);
	  Data[48] += a*x6[5]*x6[2] + c*y6[5]*y6[2] +  b*(x6[5]*y6[2] + y6[5]*x6[2]);

	  Data[49] += a*x6[3]*x6[3] + c*y6[3]*y6[3] +  b*(x6[3]*y6[3] + y6[3]*x6[3]);
	  Data[50] += a*x6[4]*x6[3] + c*y6[4]*y6[3] +  b*(x6[4]*y6[3] + y6[4]*x6[3]);
	  Data[51] += a*x6[5]*x6[3] + c*y6[5]*y6[3] +  b*(x6[5]*y6[3] + y6[5]*x6[3]);

	  Data[52] += a*x6[4]*x6[4] + c*y6[4]*y6[4] +  b*(x6[4]*y6[4] + y6[4]*x6[4]);
	  Data[53] += a*x6[5]*x6[4] + c*y6[5]*y6[4] +  b*(x6[5]*y6[4] + y6[5]*x6[4]);

	  Data[54] += a*x6[5]*x6[5] + c*y6[5]*y6[5] +  b*(x6[5]*y6[5] + y6[5]*x6[5]);

	  num++;
	  numIn1++;
	  shiftUp(false);
  }

  inline void updateTopRight(
		  const float* const x4,
		  const float* const x6,
		  const float* const y4,
		  const float* const y6,
  	  	  const float TR00, const float TR10,
  	  	  const float TR01, const float TR11,
  	  	  const float TR02, const float TR12 )
  {
	  TopRight_Data[0] += x4[0]*TR00 + y4[0]*TR10;
	  TopRight_Data[1] += x4[0]*TR01 + y4[0]*TR11;
	  TopRight_Data[2] += x4[0]*TR02 + y4[0]*TR12;

	  TopRight_Data[3] += x4[1]*TR00 + y4[1]*TR10;
	  TopRight_Data[4] += x4[1]*TR01 + y4[1]*TR11;
	  TopRight_Data[5] += x4[1]*TR02 + y4[1]*TR12;

	  TopRight_Data[6] += x4[2]*TR00 + y4[2]*TR10;
	  TopRight_Data[7] += x4[2]*TR01 + y4[2]*TR11;
	  TopRight_Data[8] += x4[2]*TR02 + y4[2]*TR12;

	  TopRight_Data[9] += x4[3]*TR00 + y4[3]*TR10;
	  TopRight_Data[10] += x4[3]*TR01 + y4[3]*TR11;
	  TopRight_Data[11] += x4[3]*TR02 + y4[3]*TR12;

	  TopRight_Data[12] += x6[0]*TR00 + y6[0]*TR10;
	  TopRight_Data[13] += x6[0]*TR01 + y6[0]*TR11;
	  TopRight_Data[14] += x6[0]*TR02 + y6[0]*TR12;

	  TopRight_Data[15] += x6[1]*TR00 + y6[1]*TR10;
	  TopRight_Data[16] += x6[1]*TR01 + y6[1]*TR11;
	  TopRight_Data[17] += x6[1]*TR02 + y6[1]*TR12;

	  TopRight_Data[18] += x6[2]*TR00 + y6[2]*TR10;
	  TopRight_Data[19] += x6[2]*TR01 + y6[2]*TR11;
	  TopRight_Data[20] += x6[2]*TR02 + y6[2]*TR12;

	  TopRight_Data[21] += x6[3]*TR00 + y6[3]*TR10;
	  TopRight_Data[22] += x6[3]*TR01 + y6[3]*TR11;
	  TopRight_Data[23] += x6[3]*TR02 + y6[3]*TR12;

	  TopRight_Data[24] += x6[4]*TR00 + y6[4]*TR10;
	  TopRight_Data[25] += x6[4]*TR01 + y6[4]*TR11;
	  TopRight_Data[26] += x6[4]*TR02 + y6[4]*TR12;

	  TopRight_Data[27] += x6[5]*TR00 + y6[5]*TR10;
	  TopRight_Data[28] += x6[5]*TR01 + y6[5]*TR11;
	  TopRight_Data[29] += x6[5]*TR02 + y6[5]*TR12;
  }

  inline void updateBotRight(
		  const float a00,
		  const float a01,
		  const float a02,
		  const float a11,
		  const float a12,
		  const float a22)
  {
	  BotRight_Data[0] += a00;
	  BotRight_Data[1] += a01;
	  BotRight_Data[2] += a02;
	  BotRight_Data[3] += a11;
	  BotRight_Data[4] += a12;
	  BotRight_Data[5] += a22;
  }

//private:
  EIGEN_ALIGN16 float Data[60];
  EIGEN_ALIGN16 float Data1k[60];
  EIGEN_ALIGN16 float Data1m[60];

  EIGEN_ALIGN16 float TopRight_Data[32];
  EIGEN_ALIGN16 float TopRight_Data1k[32];
  EIGEN_ALIGN16 float TopRight_Data1m[32];
//! 估计是只存一个点
  EIGEN_ALIGN16 float BotRight_Data[8];
  EIGEN_ALIGN16 float BotRight_Data1k[8];
  EIGEN_ALIGN16 float BotRight_Data1m[8];

  float numIn1, numIn1k, numIn1m;

  void shiftUp(bool force)
  {
	  if(numIn1 > 1000 || force) {
		  for(int i=0;i<60;i+=4)
			  _mm_store_ps(Data1k+i, _mm_add_ps(_mm_load_ps(Data+i),_mm_load_ps(Data1k+i)));
		  for(int i=0;i<32;i+=4)
			  _mm_store_ps(TopRight_Data1k+i, _mm_add_ps(_mm_load_ps(TopRight_Data+i),_mm_load_ps(TopRight_Data1k+i)));
		  for(int i=0;i<8;i+=4)
			  _mm_store_ps(BotRight_Data1k+i, _mm_add_ps(_mm_load_ps(BotRight_Data+i),_mm_load_ps(BotRight_Data1k+i)));

		  numIn1k+=numIn1;
		  numIn1=0;
		  memset(Data,0, sizeof(float)*60);
		  memset(TopRight_Data,0, sizeof(float)*32);
		  memset(BotRight_Data,0, sizeof(float)*8);
	  }

	  if(numIn1k > 1000 || force) {
		  for(int i=0;i<60;i+=4)
			  _mm_store_ps(Data1m+i, _mm_add_ps(_mm_load_ps(Data1k+i),_mm_load_ps(Data1m+i)));
		  for(int i=0;i<32;i+=4)
			  _mm_store_ps(TopRight_Data1m+i, _mm_add_ps(_mm_load_ps(TopRight_Data1k+i),_mm_load_ps(TopRight_Data1m+i)));
		  for(int i=0;i<8;i+=4)
			  _mm_store_ps(BotRight_Data1m+i, _mm_add_ps(_mm_load_ps(BotRight_Data1k+i),_mm_load_ps(BotRight_Data1m+i)));

		  numIn1m+=numIn1k;
		  numIn1k=0;
		  memset(Data1k,0, sizeof(float)*60);
		  memset(TopRight_Data1k,0, sizeof(float)*32);
		  memset(BotRight_Data1k,0, sizeof(float)*8);
	  }
  }
};

//!  用于前端tracker
class Accumulator9 {
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

  Mat99f H;
  Vec9f b;
  size_t num;

  inline void initialize()
  {
	H.setZero();
	b.setZero();
    //! 三级缓冲，累加
    memset(SSEData,0, sizeof(float)*4*45);
    memset(SSEData1k,0, sizeof(float)*4*45);
    memset(SSEData1m,0, sizeof(float)*4*45);
    num = numIn1 = numIn1k = numIn1m = 0;
  }

  inline void finish()
  {
	H.setZero();
	shiftUp(true);
	assert(numIn1==0);
	assert(numIn1k==0);

	int idx=0;
	for(int r=0;r<9;r++)
		for(int c=r;c<9;c++) {
			float d = SSEData1m[idx+0] + SSEData1m[idx+1] + SSEData1m[idx+2] + SSEData1m[idx+3];
			H(r,c) = H(c,r) = d;
			idx+=4;
		}
	  assert(idx==4*45);
  }

  inline void updateSSE_eighted(
		  const __m128 J0,const __m128 J1,
		  const __m128 J2,const __m128 J3,
		  const __m128 J4,const __m128 J5,
		  const __m128 J6,const __m128 J7,
		  const __m128 J8, const __m128 w)
  {
	  float* pt=SSEData;

	  __m128 J0w = _mm_mul_ps(J0,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J0))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J1))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J2))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J3))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J4))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J5))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J6))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J0w,J8))); pt+=4;

	  __m128 J1w = _mm_mul_ps(J1,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J1))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J2))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J3))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J4))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J5))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J6))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J1w,J8))); pt+=4;

	  __m128 J2w = _mm_mul_ps(J2,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J2w,J2))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J2w,J3))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J2w,J4))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J2w,J5))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J2w,J6))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J2w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J2w,J8))); pt+=4;

	  __m128 J3w = _mm_mul_ps(J3,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J3w,J3))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J3w,J4))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J3w,J5))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J3w,J6))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J3w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J3w,J8))); pt+=4;

	  __m128 J4w = _mm_mul_ps(J4,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J4w,J4))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J4w,J5))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J4w,J6))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J4w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J4w,J8))); pt+=4;

	  __m128 J5w = _mm_mul_ps(J5,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J5w,J5))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J5w,J6))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J5w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J5w,J8))); pt+=4;

	  __m128 J6w = _mm_mul_ps(J6,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J6w,J6))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J6w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J6w,J8))); pt+=4;

	  __m128 J7w = _mm_mul_ps(J7,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J7w,J7))); pt+=4;
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J7w,J8))); pt+=4;

	  __m128 J8w = _mm_mul_ps(J8,w);
	  _mm_store_ps(pt, _mm_add_ps(_mm_load_ps(pt),_mm_mul_ps(J8w,J8))); pt+=4;

	  num+=4;
	  numIn1++;
	  shiftUp(false);
  }

private:
  EIGEN_ALIGN16 float SSEData[4*45];
  EIGEN_ALIGN16 float SSEData1k[4*45];
  EIGEN_ALIGN16 float SSEData1m[4*45];
  float numIn1, numIn1k, numIn1m;


  void shiftUp(bool force)
  {
	  if(numIn1 > 1000 || force)
	  {
		  for(int i=0;i<45;i++)
			  _mm_store_ps(SSEData1k+4*i, _mm_add_ps(_mm_load_ps(SSEData+4*i),_mm_load_ps(SSEData1k+4*i)));
		  numIn1k+=numIn1;
		  numIn1=0;
		  memset(SSEData,0, sizeof(float)*4*45);
	  }

	  if(numIn1k > 1000 || force)
	  {
		  for(int i=0;i<45;i++)
			  _mm_store_ps(SSEData1m+4*i, _mm_add_ps(_mm_load_ps(SSEData1k+4*i),_mm_load_ps(SSEData1m+4*i)));
		  numIn1m+=numIn1k;
		  numIn1k=0;
		  memset(SSEData1k,0, sizeof(float)*4*45);
	  }
  }
};
}
