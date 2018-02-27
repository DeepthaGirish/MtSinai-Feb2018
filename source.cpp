#include <iostream>  
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <string>
#include <opencv/cv.h>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <fstream>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include<vector>
#include "opencv2/opencv.hpp"
#include <tiffio.h>

using namespace cv;
using namespace std;



ofstream myfile;
string imname;
vector<Point> HessPoints;

void writeImage(string imname, string name, Mat img)
{
	string filename=format("%s_%s.tif",imname.c_str(),name.c_str());
	imwrite(filename,img);
}


float vari(int a, int b, int c, int d)
{
	float mean = (a + b + c + d) / 4;
	float var = (((a - mean)*(a - mean)) + ((b - mean)*(b - mean)) + ((c - mean)*(c - mean)) + ((d - mean)*(d - mean))) / 4;
	return var;
}
float meaan(int a, int b, int c, int d)
{
	float mean = (a + b + c + d) / 4;
	return mean;
}


class dendrite
{
	public:
	 Mat dend_image;

	 vector<Point> filterHessian( Mat image);
	 void calcDend(vector<Point> points,int totArea, ofstream & myfile);
}dend;

class synapse
{
 public:
  Mat syn1;
  Mat syn12;  
  
  void thresholdSynapses ( Mat image1, Mat image2, ofstream & myfile);
  vector<vector<Point2i> > NoofSynapses(string Tname, string lowInt, string medInt, string highInt, ofstream & myfile);
  vector<Point2i> doublePositiveSynapses( vector<vector<Point2i> > v1, vector<vector<Point2i> >  v2, ofstream & myfile);
  void neighboursyn(string Tname, string Lname, string Mname, string Hname, vector<Point> Coordinates, ofstream & myfile, int w, int h);
  void synapsNearDendrite();
  
}syn;



//finds dendrite points in the image
vector<Point>  dendrite::filterHessian(Mat image)
{
	vector<Point> edges;
	int co = 0;
	Mat org = image.clone();
	Mat outputDend(image.size(),CV_8UC3);
	//Mat temp;
  //cv::GaussianBlur( image,temp, cv::Size(0, 0), 3);
	//cv::addWeighted(image, 1.5, temp, -0.5, 0, image);
	cvtColor(image, image, CV_BGR2GRAY);
	cv::Mat dXX, dYY, dXY;
	std::vector<float> eigenvalues(2);
	cv::Mat hessian_matrix(2, 2, CV_32F);
	Mat eigenvec = Mat::ones(2, 2, CV_32F);
	vector<Point> HessianPoints;
	//std::vector<float> eigenvec(2,2); 

	//calculte derivatives
	cv::Sobel(image, dXX, CV_32F, 2, 0);
	cv::Sobel(image, dYY, CV_32F, 0, 2);
	cv::Sobel(image, dXY, CV_32F, 1, 1);

	//apply gaussian filtering to the image
	cv::Mat gau = cv::getGaussianKernel(3, -1, CV_32F);
	cv::sepFilter2D(dXX, dXX, CV_32F, gau.t(), gau);
	cv::sepFilter2D(dYY, dYY, CV_32F, gau.t(), gau);
	cv::sepFilter2D(dXY, dXY, CV_32F, gau.t(), gau);

	HessianPoints.clear();
	//----------Inside image
	int countofdendrites = 0;int edgescount=0;
	int developed = 0; float lambda1=0, lambda2=0; float signedval;
	int lessdeveloped = 0;
	for (int i = 1; i < image.rows; i++) {
		for (int j = 1; j < image.cols; j++) {
			hessian_matrix.at<float>(Point(0, 0)) = dXX.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 1)) = dYY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(0, 1)) = dXY.at<float>(Point(j, i));
			hessian_matrix.at<float>(Point(1, 0)) = dXY.at<float>(Point(j, i));

			// find eigen values of hessian matrix 
			/* Larger  eigenvalue  show the  direction  of  intensity change, while  smallest  eigenvalue  show  the  direction  of  vein.*/
			eigen(hessian_matrix, eigenvalues, eigenvec);
			//assigning higher eigenvalue to lambda1 and lower eigen value to lambda2
			if (abs(eigenvalues[0]) < abs(eigenvalues[1]))
			{lambda1=abs(eigenvalues[0]); lambda2=abs(eigenvalues[1]); signedval=eigenvalues[1]; }
			else
			{lambda2=abs(eigenvalues[0]); lambda1=abs(eigenvalues[1]);signedval=eigenvalues[0];}

		
			/*Main Condition for dendrite detection*/if (lambda1<1 && lambda2>40 )
			{
			
					HessianPoints.push_back(Point(j,i));
					circle(org, cv::Point(j, i), 3, cv::Scalar(128, 255,255), 3, 8, 0);//orange
					//line(org, cv::Point(j, i), cv::Point(j+5, i+5), cv::Scalar(128, 255,255), 3, 8, 0);
			}
		}
	}
	//ConnectComponents(outputDend);
	imwrite("outpDend.png",org);
	return HessianPoints;

}

//calculates total area of dendrites within the image
void dendrite::calcDend(vector<Point> dendpoints,int totArea, ofstream & myfile)
{
float area= dendpoints.size()*3;
float percentArea=(area/totArea)*100;
//myfile<<"Area occupied by dendrites"<<","<<"Percentage area occupied by dendrites<<",";
myfile<<area<<","<<percentArea<<",";
}

//change the proptotype
void synapse:: thresholdSynapses(Mat src_syn1, Mat src_syn2, ofstream & myfile)
{
	 //Mat highInt1_thresholded,medInt1_thresholded,lowInt1_thresholded;
	 Mat highInt1_thresholded(src_syn1.size(), CV_8UC1);
	 Mat medInt1_thresholded(src_syn1.size(), CV_8UC1);
	 Mat lowInt1_thresholded(src_syn1.size(), CV_8UC1);
	 Mat tot1_thresholded(src_syn1.size(), CV_8UC1);
	 src_syn1.convertTo(src_syn1, CV_8UC1);
	 
	 threshold( src_syn1, tot1_thresholded, 50, 255,3 );

	 threshold( src_syn1, lowInt1_thresholded, 50, 100,3 );
	 threshold( src_syn1, medInt1_thresholded, 100, 175,3 );
	 threshold( src_syn1, highInt1_thresholded, 175, 255,3 );
	 
	 lowInt1_thresholded=lowInt1_thresholded-medInt1_thresholded;
	 medInt1_thresholded=medInt1_thresholded-highInt1_thresholded;
	 
	 imwrite("tempSyn1.png",  tot1_thresholded);
	 imwrite("tempLow1.png",  lowInt1_thresholded);
	 imwrite("tempMed1.png",  medInt1_thresholded);
	 imwrite("tempHigh1.png", highInt1_thresholded);
	 
	 //Mat highInt12_thresholded,medInt12_thresholded,lowInt12_thresholded;
	 Mat highInt12_thresholded(src_syn2.size(), CV_8UC1);
	 Mat medInt12_thresholded(src_syn2.size(), CV_8UC1);
	 Mat lowInt12_thresholded(src_syn2.size(), CV_8UC1);
	 Mat tot12_thresholded(src_syn2.size(), CV_8UC1);
	
	 threshold( src_syn2, tot12_thresholded, 50, 255,3 );

	 threshold( src_syn2, lowInt12_thresholded, 50, 100,3 );
	 threshold( src_syn2, medInt12_thresholded, 100, 175,3 );
	 threshold( src_syn2, highInt12_thresholded, 175, 255,3 );
	
	 lowInt12_thresholded=lowInt12_thresholded-medInt12_thresholded;
	 medInt12_thresholded=medInt12_thresholded-highInt12_thresholded;
	 
	 imwrite("tempSyn12.png",  tot12_thresholded);
	 imwrite("tempLow12.png",  lowInt12_thresholded);
	 imwrite("tempMed12.png",  medInt12_thresholded);
	 imwrite("tempHigh12.png", highInt12_thresholded);
	
	 vector<vector<Point2i> > NumSynapses1 = NoofSynapses("tempSyn1.png","tempLow1.png","tempMed1.png","tempHigh1.png",myfile);
	 vector<vector<Point2i> > NumSynapses12= NoofSynapses("tempSyn12.png","tempLow12.png","tempMed12.png","tempHigh12.png", myfile);
   
	vector<Point2i> doublePosCoord= doublePositiveSynapses(NumSynapses1,NumSynapses12, myfile);
}


vector<vector<Point> > synapse:: NoofSynapses(string Tname,string Lname, string Mname, string Hname, ofstream & myfile)
{


vector<vector<Point2i> > nonZeroCoordinates;
int lowcount=0,medcount=0, highcount=0, totcount=0;
vector<Point> temp;


Mat imgT=imread(Tname, CV_8U);
totcount=countNonZero(imgT);
findNonZero(imgT,temp);
nonZeroCoordinates.push_back(temp);


Mat imgL=imread(Lname, CV_8U);
lowcount=countNonZero(imgL);
findNonZero(imgL,temp);
nonZeroCoordinates.push_back(temp);

Mat imgM=imread(Mname, CV_8U);
medcount=countNonZero(imgM);
findNonZero(imgM,temp);
nonZeroCoordinates.push_back(temp);

Mat imgH=imread(Hname, CV_8U);
highcount=countNonZero(imgH);
findNonZero(imgH, temp);
nonZeroCoordinates.push_back(temp);

//myfile<<"Total no. of synpases"<<","<<"No of low int. synpases"<<","<<"No of med int. synpases"<<","<<"No of high int. synpases"<<",";
myfile<<totcount<<","<<lowcount<<","<<medcount<<","<<highcount<<",";
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[0], myfile,20,20);//avg low, med, high syn around all synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[0], myfile,40,40);//avg low, med, high syn around all synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[0], myfile,80,80);//avg low, med, high syn around all synapses

neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[1], myfile,20,20);//avg low, med, high syn around LOW INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[1], myfile,40,40);//avg low, med, high syn around LOW INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[1], myfile,80,80);//avg low, med, high syn around LOW INT synapses

neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[2], myfile,20,20);//avg low, med, high syn around MED INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[2], myfile,40,40);//avg low, med, high syn around MED INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[2], myfile,80,80);//avg low, med, high syn around MED INT synapses

neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[3], myfile,20,20);//avg low, med, high syn around HIGH INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[3], myfile,40,40);//avg low, med, high syn around HIGH INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, nonZeroCoordinates[3], myfile,80,80);//avg low, med, high syn around HIGH INT synapses

neighboursyn(Tname, Lname,  Mname, Hname, HessPoints, myfile,20,20);//avg low, med, high syn around HIGH INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, HessPoints, myfile,40,40);//avg low, med, high syn around HIGH INT synapses
neighboursyn(Tname, Lname,  Mname, Hname, HessPoints, myfile,80,80);//avg low, med, high syn around HIGH INT synapses

return nonZeroCoordinates;
}

vector<Point2i> synapse:: doublePositiveSynapses( vector<vector<Point2i> > v1, vector<vector<Point2i> >  v2, ofstream & myfile)
{
int count=0, Tcount=0, Lcount=0, Mcount=0, Hcount=0;
int p=0;
vector<Point2i> doublePosCoord;
while(p<4)//no. of channels (tot, low, med, high)
 {
	for(int i=0; i< v1[p].size(); i++)
	{
		for (int j=0; j<v2[p].size(); j++)
		{
			if(v1[p][i]==v2[p][j])
				{
					count++;
					if (p==0)// We do not want L, M, H double positive, we wnt all double positive coord
						doublePosCoord.push_back(v1[p][i]);
				}
			
		}
	}
	switch(p) {
	case 0  :
      	Tcount=count;
      	break; 
   	case 1  :
      	Lcount=count;
      	break; 
   	case 2  :
      	Mcount=count;
      	break; 
   	case 3  :
      	Hcount=count;
      	break; 
      	}
      	count=0;
      	++p;
 }
//myfile<<"Total No. of double positive synapses"<<","<<"No. of low int double positive synapses"<<","<<"No. of med int double positive synapses"<<","<<"No. of high int double positive synapses"<<",";
myfile<<Tcount<<","<<Lcount<<","<<Mcount<<","<<Hcount<<",";
return doublePosCoord;
}

//----------------------Finds count of L, M, H intensity synapses around // LMH syn or dendrites --------------------------------
void synapse::neighboursyn(string Tname, string Lname, string Mname, string Hname, vector<Point> Coordinates, ofstream & myfile, int w, int h)
{
  Mat redlow=imread(Lname, CV_8U); Mat redmed=imread(Mname, CV_8U); Mat redhigh=imread(Hname, CV_8U); 
  
	unsigned int totl = 0, totm = 0, toth = 0;
	double totlvar = 0, totmvar = 0, tothvar = 0;
	int lcount1; int mcount1; int hcount1; int lcount2; 
	int mcount2; int hcount2; int lcount3; int mcount3; 
	int hcount3; int lcount4; int mcount4; int hcount4;

	for (int i = 0; i < Coordinates.size(); i++)
	{

		Point a = Point(Coordinates[i].x, Coordinates[i].y);
		lcount1 = 0; mcount1 = 0; hcount1 = 0; 
		lcount2 = 0; mcount2 = 0; hcount2 = 0; 
		lcount3 = 0; mcount3 = 0; hcount3 = 0; 
		lcount4 = 0; mcount4 = 0; hcount4 = 0;


		if (((a.x) + w  < redlow.rows) && ((a.x) - w > 0) && ((a.y) + h < redlow.cols) && ((a.y) - h> 0))

		{
			CvRect myrect = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrect.x >= 0 && myrect.y >= 0 && myrect.width + myrect.x < redlow.cols && myrect.height + myrect.y < redlow.rows)
			{
				int px = (a.x + (0.5*w)); int py = (a.y + (h / 2));
				//if (px >= 0 && py >= 0 && myrect.width + px < redlow.cols && myrect.height + py < redlow.rows)
				Mat lroi = redlow(myrect);// creating a new image from roi of redlow
				cv::Rect top_left(cv::Point(0, 0), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, lroi.size().width / 2), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(lroi.size().height / 2, 0), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(lroi.size().height / 2, lroi.size().width / 2), cv::Size(lroi.size().height / 2, lroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = lroi(top_right);
				Image2 = lroi(top_left);
				Image3 = lroi(bottom_right);
				Image4 = lroi(bottom_left);

				lcount1 = countNonZero(Image1);
				lcount2 = countNonZero(Image2);
				lcount3 = countNonZero(Image3);
				lcount4 = countNonZero(Image4);
				float lvar = meaan(lcount1, lcount2, lcount3, lcount4);
				totlvar = totlvar + lvar;
			}

		}
		if (((a.x) + w + 20 < redmed.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redmed.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectM = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectM.x >= 0 && myrectM.y >= 0 && myrectM.width + myrectM.x < redmed.cols && myrectM.height + myrectM.y < redmed.rows)
			{
				Mat mroi = redmed(myrectM);// creating a new image from roi of redmed

				cv::Rect top_left(cv::Point(0, 0), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, mroi.size().width / 2), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(mroi.size().height / 2, 0), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(mroi.size().height / 2, mroi.size().width / 2), cv::Size(mroi.size().height / 2, mroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = mroi(top_right);
				Image2 = mroi(top_left);
				Image3 = mroi(bottom_right);
				Image4 = mroi(bottom_left);
				mcount1 = countNonZero(Image1);
				mcount2 = countNonZero(Image2);
				mcount3 = countNonZero(Image3);
				mcount4 = countNonZero(Image4);
				float mvar = meaan(mcount1, mcount2, mcount3, mcount4);
				totmvar = totmvar + mvar;
			}
		}


		if (((a.x) + w + 20 < redhigh.rows) && ((a.x) - w - 20 > 0) && ((a.y) + h + 20 < redhigh.cols) && ((a.y) - h - 20> 0))

		{
			CvRect myrectH = cvRect(a.x - w / 2, a.y - h / 2, w, h); /// create a 4*4 rectangle around the synapse
			if (myrectH.x >= 0 && myrectH.y >= 0 && myrectH.width + myrectH.x < redhigh.cols && myrectH.height + myrectH.y < redhigh.rows)
			{
				Mat hroi = redhigh(myrectH);// creating a new image from roi of redmed


				cv::Rect top_left(cv::Point(0, 0), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect top_right(cv::Point(0, hroi.size().width / 2), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect bottom_left(cv::Point(hroi.size().height / 2, 0), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Rect bottom_right(cv::Point(hroi.size().height / 2, hroi.size().width / 2), cv::Size(hroi.size().height / 2, hroi.size().width / 2));
				cv::Mat Image1, Image2, Image3, Image4;
				Image1 = hroi(top_right);
				Image2 = hroi(top_left);
				Image3 = hroi(bottom_right);
				Image4 = hroi(bottom_left);
				hcount1 = countNonZero(Image1);
				hcount2 = countNonZero(Image2);
				hcount3 = countNonZero(Image3);
				hcount4 = countNonZero(Image4);
				float hvar = meaan(hcount1, hcount2, hcount3, hcount4);
				tothvar = tothvar + hvar;

			}
		}


	}
	//myfile << "neignborsyn" << "," << "Average no of low int synapse arnd  int syn(40)" << "," << "Average no of med int synapse arnd  int syn(40)" << "," << "Average no of high int synapse arnd  int syn(40)" << ",";
	if (Coordinates.size()>0)
		myfile << int(totlvar / Coordinates.size()) << "," << int(totmvar / Coordinates.size()) << "," << int(tothvar / Coordinates.size()) << ",";
	else
		myfile << 0 << "," << 0 << "," << 0 << ",";
}





int main(int argc, char** argv)
{

string imname=argv[1];
cout<<"Processing "<<argv[1]<<endl;
string syn12_path="SYN12,channel split,3 genes//SYN1&2,single channel split";
string syn1_path= "SYP1,channel split,3 genes//SYP1,single channel split";
string dend_path="MAP2,channel split,3 genes//MAP2,single channel split";

string dend_name=format("%s//%s",dend_path.c_str(),argv[1]);
string syn1_name=format("%s//%s",syn1_path.c_str(),argv[1]);
string syn12_name=format("%s//%s",syn12_path.c_str(),argv[1]);

myfile.open("KristenMetrics.csv",std::ofstream::out | std::ofstream::app);
/*myfile<<"Area occupied by dendrites"<<","<<"Percentage area occupied by dendrites"<<",";

myfile<<"Total no. of synpases"<<","<<"No of low int. synapses"<<","<<"No of med int. synapses"<<","<<"No of high int. synapses"<<",";

myfile << "Average no of SYP1 low int synapse arnd synapses(20)" << "," << "Average no of SYP1 med int synapse arnd synapses(20)" << "," << "Average no of SYP1 high int synapse arnd synapses(20)" << ",";
myfile << "Average no of SYP1 low int synapse arnd synapses(40)" << "," << "Average no of SYP1 med int synapse arnd synapses(40)" << "," << "Average no of SYP1 high int synapse arnd synapses(40)" << ",";
myfile << "Average no of SYP1 low int synapse arnd synapses(80)" << "," << "Average no of  SYP1 med int synapse arnd synapses(80)" << "," << "Average no of  SYP1 high int synapse arnd synapses(80)" << ",";

myfile << "Average no of SYP1 low int synapse arnd LOW INT synapses(20)" << "," << "Average no of SYP1 med int synapse arnd LOW INT synapses(20)" << "," << "Average no of SYP1 high int synapse arnd LOW INT  synapses(20)" << ",";
myfile << "Average no of SYP1 low int synapse arnd LOW INT synapses(40)" << "," << "Average no of SYP1 med int synapse arnd LOW INT synapses(40)" << "," << "Average no of SYP1 high int synapse arnd LOW INT  synapses(40)" << ",";
myfile << "Average no of SYP1 low int synapse arnd LOW INT synapses(80)" << "," << "Average no of  SYP1 med int synapse arnd LOW INT synapses(80)" << "," << "Average no of  SYP1 high int synapse arnd LOW INT  synapses(80)" << ",";

myfile << "Average no of SYP1 low int synapse arnd MED INT synapses(20)" << "," << "Average no of SYP1 med int synapse arnd MED INT synapses(20)" << "," << "Average no of SYP1 high int synapse arnd MED INT synapses(20)" << ",";
myfile << "Average no of SYP1 low int synapse arnd MED INT synapses(40)" << "," << "Average no of SYP1 med int synapse arnd MED INT synapses(40)" << "," << "Average no of SYP1 high int synapse arnd MED INT  synapses(40)" << ",";
myfile << "Average no of SYP1 low int synapse arnd MED INT synapses(80)" << "," << "Average no of  SYP1 med int synapse arnd MED INT synapses(80)" << "," << "Average no of  SYP1 high int synapse arnd MED INT synapses(80)" << ",";

myfile << "Average no of SYP1 low int synapse arnd HIGH INT synapses(20)" << "," << "Average no of SYP1 med int synapse arnd HIGH INT synapses(20)" << "," << "Average no of SYP1 high int synapse arnd HIGH INT synapses(20)" << ",";
myfile << "Average no of SYP1 low int synapse arnd HIGH INT synapses(40)" << "," << "Average no of SYP1 med int synapse arnd HIGH INT synapses(40)" << "," << "Average no of SYP1 high int synapse arnd HIGH INT synapses(40)" << ",";
myfile << "Average no of SYP1 low int synapse arnd HIGH INT synapses(80)" << "," << "Average no of  SYP1 med int synapse arnd HIGH INT synapses(80)" << "," << "Average no of  SYP1 high int synapse arnd HIGH INT synapses(80)" << ",";

myfile << "Average no of SYP1 low int synapse around dendrites(20)" << "," << "Average no of SYP1 med int synapse arnd dendrites(20)" << "," << "Average no of SYP1 high int synapse arnd dendrites(20)" << ",";
myfile << "Average no of SYP1 low int synapse arnd dendrites(40)" << "," << "Average no of SYP1 med int synapse arnd dendrites(40)" << "," << "Average no of SYP1 high int synapse arnd dendrites(40)" << ",";
myfile << "Average no of SYP1 low int synapse arnd dendrites(80)" << "," << "Average no of  SYP1 med int synapse arnd dendrites(80)" << "," << "Average no of  SYP1 high int synapse arnd dendrites(80)" << ",";

myfile<<"Total no. of synpases"<<","<<"No of low int. synapses"<<","<<"No of med int. synapses"<<","<<"No of high int. synapses"<<",";

myfile << "Average no of SYN12 low int synapse arnd synapses(20)" << "," << "Average no of SYN12 med int synapse arnd synapses(20)" << "," << "Average no of SYN12 high int synapse arnd synapses(20)" << ",";
myfile << "Average no of SYN12 low int synapse arnd synapses(40)" << "," << "Average no of SYN12 med int synapse arnd synapses(40)" << "," << "Average no of SYN12 high int synapse arnd synapses(40)" << ",";
myfile << "Average no of SYN12 low int synapse arnd synapses(80)" << "," << "Average no of  SYN12 med int synapse arnd synapses(80)" << "," << "Average no of  SYN12 high int synapse arnd synapses(80)" << ",";

myfile << "Average no of SYN12 low int synapse arnd LOW INT synapses(20)" << "," << "Average no of SYN12 med int synapse arnd LOW INT synapses(20)" << "," << "Average no of SYN12 high int synapse arnd LOW INT  synapses(20)" << ",";
myfile << "Average no of SYN12 low int synapse arnd LOW INT synapses(40)" << "," << "Average no of SYN12 med int synapse arnd LOW INT synapses(40)" << "," << "Average no of SYN12 high int synapse arnd LOW INT  synapses(40)" << ",";
myfile << "Average no of SYN12 low int synapse arnd LOW INT synapses(80)" << "," << "Average no of  SYN12 med int synapse arnd LOW INT synapses(80)" << "," << "Average no of  SYN12 high int synapse arnd LOW INT  synapses(80)" << ",";

myfile << "Average no of SYN12 low int synapse arnd MED INT synapses(20)" << "," << "Average no of SYN12 med int synapse arnd MED INT synapses(20)" << "," << "Average no of SYN12 high int synapse arnd MED INT synapses(20)" << ",";
myfile << "Average no of SYN12 low int synapse arnd MED INT synapses(40)" << "," << "Average no of SYN12 med int synapse arnd MED INT synapses(40)" << "," << "Average no of SYN12 high int synapse arnd MED INT  synapses(40)" << ",";
myfile << "Average no of SYN12 low int synapse arnd MED INT synapses(80)" << "," << "Average no of  SYN12 med int synapse arnd MED INT synapses(80)" << "," << "Average no of  SYN12 high int synapse arnd MED INT synapses(80)" << ",";

myfile << "Average no of SYN12 low int synapse arnd HIGH INT synapses(20)" << "," << "Average no of SYN12 med int synapse arnd HIGH INT synapses(20)" << "," << "Average no of SYN12 high int synapse arnd HIGH INT synapses(20)" << ",";
myfile << "Average no of SYN12 low int synapse arnd HIGH INT synapses(40)" << "," << "Average no of SYN12 med int synapse arnd HIGH INT synapses(40)" << "," << "Average no of SYN12 high int synapse arnd HIGH INT synapses(40)" << ",";
myfile << "Average no of SYN12 low int synapse arnd HIGH INT synapses(80)" << "," << "Average no of  SYN12 med int synapse arnd HIGH INT synapses(80)" << "," << "Average no of  SYN12 high int synapse arnd HIGH INT synapses(80)" << ",";

myfile << "Average no of SYN12 low int synapse around dendrites(20)" << "," << "Average no of SYN12 med int synapse arnd dendrites(20)" << "," << "Average no of SYN12 high int synapse arnd dendrites(20)" << ",";
myfile << "Average no of SYN12 low int synapse arnd dendrites(40)" << "," << "Average no of SYN12 med int synapse arnd dendrites(40)" << "," << "Average no of SYN12 high int synapse arnd dendrites(40)" << ",";
myfile << "Average no of SYN12 low int synapse arnd dendrites(80)" << "," << "Average no of SYN12 med int synapse arnd dendrites(80)" << "," << "Average no of  SYN12 high int synapse arnd dendrites(80)" << ",";

myfile<<"Total No. of double positive synapses"<<","<<"No. of low int double positive synapses"<<","<<"No. of med int double positive synapses"<<","<<"No. of high int double positive synapses"<<",";
myfile<<endl;*/

myfile<<argv[1]<<",";
dend.dend_image=imread(dend_name);
syn.syn1=imread(syn1_name);
syn.syn12=imread(syn12_name);

int totArea=dend.dend_image.rows*dend.dend_image.cols;
HessPoints=dend.filterHessian(dend.dend_image);
dend.calcDend(HessPoints,totArea, myfile);

syn.thresholdSynapses(syn.syn1, syn.syn12, myfile);
myfile<<endl;
return 0;

}

