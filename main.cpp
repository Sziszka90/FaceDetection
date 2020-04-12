/* FACE DETECTION */

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include <iostream>
#include <stdio.h>
#include <conio.h>

using namespace std;
using namespace cv;

int main()
{

    cout<<"Face and Mouth detection"<<endl;

	CascadeClassifier FaceDetection, SmileDetection;

	FaceDetection.load("haarcascade_frontalface_alt.xml");
	SmileDetection.load("haarcascade_smile.xml");

	VideoCapture camera;

    camera.open(0);

	Mat Frame, grayFrame, faceImg;

	while(true)
	{
		camera>>Frame;
		cvtColor(Frame, grayFrame, cv::COLOR_BGR2GRAY);
		equalizeHist(grayFrame, grayFrame);

		std::vector<Rect> Face;
		std::vector<Rect> Smile;
		Rect croppedRec;

		FaceDetection.detectMultiScale(grayFrame, Face, 1.1, 3, CASCADE_SCALE_IMAGE, Size(30,30));

		for(int i = 0; i < Face.size(); i++)
		{
			Point pt1(Face[i].x + Face[i].width, Face[i].y + Face[i].height);
			Point pt2(Face[i].x, Face[i].y);

			rectangle(Frame, pt1, pt2, CvScalar(0, 255, 0, 0), 1, 8, 0);

			croppedRec=Face[i];
			croppedRec.height=Face[i].height/2;
			faceImg=grayFrame(croppedRec);

			SmileDetection.detectMultiScale(faceImg, Smile, 1.1, 3, CASCADE_SCALE_IMAGE, Size(30,30));

            if(!Smile.empty())
            {
                int z=0;

                Smile[z].x = Smile[z].x+Face[z].x;
                Smile[z].y = Smile[z].y+Face[z].y+Face[z].height/2;

                Point pt3(Smile[z].x, Smile[z].y);
                Point pt4(Smile[z].x + Smile[z].width, Smile[z].y + Smile[z].height);

                rectangle(Frame, pt3, pt4, CvScalar(255, 0, 0, 0), 1, 8, 0);
            }
		 }

         imshow("Frame", Frame);

         waitKey(70);
	}
	return 0;
}
