#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <ctime>
using namespace std;
using namespace cv;

CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
CascadeClassifier smiles_cascade;

int main()
{
	unsigned int start_time = clock();

	if (!face_cascade.load(cv::samples::findFile("D:/Camera/haarcascade_frontalface_alt.xml"))) {
		printf("Error loading face cascade model \n");
		return -1;
	}

	if (!eyes_cascade.load(cv::samples::findFile("D:/Camera/haarcascade_eye_tree_eyeglasses.xml"))) {
		printf("Error loading face cascade model \n");
		return -1;
	}

	if (!smiles_cascade.load(cv::samples::findFile("D:/Camera/haarcascade_smile.xml"))) {
		printf("Error loading face cascade model \n");
		return -1;
	}

	VideoCapture cap("video_face.mp4");
	if (!cap.isOpened()) {
		std::cout << "Error opening video" << std::endl;
		return -1;
	}

	int frame_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
	int frame_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

	VideoWriter video("./outface.mp4", cv::VideoWriter::fourcc('A', 'V', 'C', '1'), cap.get(cv::CAP_PROP_FPS), cv::Size(frame_width * 0.6, frame_height * 0.6));

	Mat Img, origImg, image1, grayImg;
	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> eyes;
	std::vector<cv::Rect> smiles;

	while (true) {
		cap >> Img;
		if (Img.empty())
		{
			break;
		}

		origImg = Img.clone();
		cv::resize(origImg, Img, cv::Size(), 0.6, 0.6);
		GaussianBlur(Img, image1, Size(3, 3), 0);
		cvtColor(image1, grayImg, COLOR_BGR2GRAY);

		face_cascade.detectMultiScale(grayImg, faces, 1.1, 5);
		for (const auto& face : faces) {
			rectangle(Img, face, Scalar(255, 0, 0), 2);
		}

#pragma omp parallel sections num_threads(2)
		{
#pragma omp section
			{
				for (const auto& face : faces) 
				{
					Mat f_mat = Img(face);
					eyes_cascade.detectMultiScale(f_mat, eyes, 1.1, 5);
					for (const auto& eye : eyes) {
						rectangle(Img, Point(face.x + eye.x, face.y + eye.y), Point(face.x + eye.x + eye.width, face.y + eye.y + eye.height), Scalar(0, 255, 0), 2);
					}
				}
				
			}
#pragma omp section
			{
				for (const auto& face : faces)
				{
					Mat f_mat = Img(face);
					smiles_cascade.detectMultiScale(grayImg, smiles, 1.165, 35, 0, cv::Size(25, 25));
					for (const auto& smile : smiles) {
						rectangle(Img, Point(face.x + smile.x, face.y + smile.y), Point(face.x + smile.x + smile.width, face.y + smile.y + smile.height), Scalar(0, 0, 255), 2);
					}
				}
				
			}
		}

		imshow("Output Video", Img);
		video.write(Img);
		char key = (char)waitKey(30);
		if (key == 'q' || key == 27)
		{
			break;
		}
	}

	cap.release();
	video.release();
	destroyAllWindows();

	unsigned int end_time = clock();
	std::cout << "All time: " << (end_time - start_time) / 1000 << "s." << std::endl;

	return 0;

}