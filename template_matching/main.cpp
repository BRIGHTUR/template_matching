#include<iostream>
#include<opencv2/opencv.hpp>
#include<math.h>
using namespace std;
using namespace cv;

int calcHOG(cv::Mat src, float * hist, int nAngle, int cellSize)
{
	if (cellSize> src.cols || cellSize> src.rows) {
		return -1;
	}
	//��������
	int nX = src.cols / cellSize;
	int nY = src.rows / cellSize;
	int binAngle = 360 / nAngle;
	//�����ݶȼ��Ƕ�
	Mat gx, gy;
	Mat mag, angle;
	Sobel(src, gx, CV_32F, 1, 0, 1);
	Sobel(src, gy, CV_32F, 0, 1, 1);
	// x�����ݶȣ�y�����ݶȣ��ݶȣ��Ƕȣ������������or�Ƕ�
	// Ĭ���ǻ���radians��ͨ�����һ����������ѡ��Ƕ�degrees.
	cartToPolar(gx, gy, mag, angle, true);
	cv::Rect roi;
	roi.x = 0;
	roi.y = 0;
	roi.width = cellSize;
	roi.height = cellSize;
	for (int i = 0; i < nY; i++) {
		for (int j = 0; j < nX; j++) {
			cv::Mat roiMat;
			cv::Mat roiMag;
			cv::Mat roiAgl;
			roi.x = j*cellSize;
			roi.y = i*cellSize;
			//��ֵͼ��
			roiMat = src(roi);
			roiMag = mag(roi);
			roiAgl = angle(roi);
			//��ǰcell��һ��Ԫ���������е�λ��
			int head = (i*nX + j)*nAngle;
			for (int n = 0; n < roiMat.rows; n++) {
				for (int m = 0; m < roiMat.cols; m++) {
					//����Ƕ����ĸ�bin��ͨ��int�Զ�ȡ��ʵ��
					int pos = (int)(roiAgl.at<float>(n, m) / binAngle);
					hist[head + pos] += roiMag.at<float>(n, m);
				}
			}

		}
	}
	return 0;
}

float normL2(float * Hist1, float * Hist2, int size)
{
	float sum = 0;
	for (int i = 0; i < size; i++) {
		sum += (Hist1[i] - Hist2[i])*(Hist1[i] - Hist2[i]);
	}
	sum = sqrt(sum);
	return sum;
}

int main()
{
	Mat srcMat = imread("img.png");
	Mat refer = imread("template.png",0);
	Mat test1;
	cvtColor(srcMat, test1, CV_BGR2GRAY);
	//refer.convertTo(refer, CV_32F);
	//refer.convertTo(test1, CV_32F);
	float dis1; //ֱ��ͼ����
	float dis_min; //ֱ��ͼ����
	float dis_temp; //ֱ��ͼ�����ݴ�

	/*�����ݴ�*/
	int i_min = 0;
	int j_min = 0;

	int nAngle = 8;
	int blockSize = 16;
	int nx = refer.cols / blockSize;
	int ny = refer.rows / blockSize;
	int bins = nx*ny*nAngle;
	int rows = refer.rows;
	int cols = refer.cols;

	//����ֱ��ͼ����
	float *ref_hist = new float[bins];
	memset(ref_hist, 0, sizeof(float)*bins);

	float *temp_hist = new float[bins];
	memset(temp_hist, 0, sizeof(float)*bins);

	int reCode = 0;
	//����ͼƬ��HOG
	reCode=calcHOG(refer, ref_hist, nAngle, blockSize);
	Mat test2;
	test2 = test1(Rect(0, 0, 108, 48)).clone();
	reCode=calcHOG(test2, temp_hist, nAngle, blockSize);
	dis_min = normL2(ref_hist, temp_hist, bins);
	//test2 = test1(Rect(2, 2, 16, 16));
	//Mat cell1 = Mat(Size(16, 16), CV_32F);
	int nnx = test1.cols / refer.cols;
	int nny= test1.rows / refer.rows;
	for (int i = 0; i < nnx; i++) {
		for (int j = 0; j < nny; j++) {
			float *temp1_hist = new float[bins];
			memset(temp1_hist, 0, sizeof(float)*bins);
			test2 = test1(Rect(108*i, 48*j, 108, 48)).clone();
			/*����ƥ��ֱ��ͼ*/
			reCode=calcHOG(test2, temp1_hist, nAngle, blockSize);
			dis_temp = normL2(ref_hist,temp1_hist, bins);
			if (dis_temp < dis_min) {
				dis_min = dis_temp;
				i_min =  i;
				j_min =  j;
			}
			delete[] temp1_hist;
		}
	}
	/*
	Mat test3;
	test3 = test1(Rect(i_min*108, j_min*48, 108, 48)).clone();
	waitKey(0);*/
	cv::Rect rect;
	rect.x = i_min*108;
	rect.y = j_min*48;
	rect.width = 108;
	rect.height = 48;
	rectangle(srcMat, rect, CV_RGB(255, 0, 0), 1, 8, 0);
	imshow("result", srcMat);
	waitKey(0);
}