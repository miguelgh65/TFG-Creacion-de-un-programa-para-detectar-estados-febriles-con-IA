#include <iostream>
#include <opencv2/opencv.hpp>
#include <unistd.h>
#define GetCurrentDir getcwd

using namespace std;
cv::Mat num_comp[10];
cv::Mat all_images[6];
std::string texts[6];

std::string get_current_dir() {
   char buff[FILENAME_MAX]; //create string buffer to hold path
   GetCurrentDir( buff, FILENAME_MAX );
   string current_working_dir(buff);
   return current_working_dir;
}

void load_numbers(){
    cvtColor(cv::imread("cero.png"), num_comp[0], CV_BGR2GRAY);
    cvtColor(cv::imread("uno.png"), num_comp[1], CV_BGR2GRAY);
    cvtColor(cv::imread("dos.png"), num_comp[2], CV_BGR2GRAY);
    cvtColor(cv::imread("tres.png"), num_comp[3], CV_BGR2GRAY);
    cvtColor(cv::imread("cuatro.png"), num_comp[4], CV_BGR2GRAY);
    cvtColor(cv::imread("cinco.png"), num_comp[5], CV_BGR2GRAY);
    cvtColor(cv::imread("seis.png"), num_comp[6], CV_BGR2GRAY);
    cvtColor(cv::imread("siete.png"), num_comp[7], CV_BGR2GRAY);
    cvtColor(cv::imread("ocho.png"), num_comp[8], CV_BGR2GRAY);
    cvtColor(cv::imread("nueve.png"), num_comp[9], CV_BGR2GRAY);

//    for (int i=0;i<10;i++){
//        cout<<i<<endl;
//        for (int y=0;y<12;y++){
//            for (int x=0;x<8;x++){
//                cout<<(int)(num_comp[i].at<uchar>(y,x))/255;
//            }
//            cout<<endl;
//        }
//    }
//    num_comp[1]=cv::imread("uno.png");
//    num_comp[2]=cv::imread("dos.png");
//    num_comp[3]=cv::imread("tres.png");
//    num_comp[4]=cv::imread("cuatro.png");
//    num_comp[5]=cv::imread("cinco.png");
//    num_comp[6]=cv::imread("seis.png");
//    num_comp[7]=cv::imread("siete.png");
//    num_comp[8]=cv::imread("ocho.png");
//    num_comp[9]=cv::imread("nueve.png");
//    //return num_comp;
}

int render_num (cv::Mat image){
    cv::Mat grey,dif;
    cvtColor(image, grey, CV_BGR2GRAY);
//    cv::imshow("comaprado",image);
//    cv::waitKey(0);
    for (int i=0;i<10;i++){
        cv::absdiff(num_comp[i],grey,dif);
//        cv::imshow("dif",dif);
//            cv::waitKey(0);
        cv::threshold(dif,dif,128,1,cv::THRESH_BINARY);
//        cv::imshow("thres",dif);
//            cv::waitKey(0);
        double s = cv::sum( dif )[0];
        if (s<2){
            return i;
        }

    }
    return 0;
}

float render_image(cv::Mat image){
    cv::Rect Roi1= cv::Rect(3,5,8,12);
    cv::Rect Roi2= cv::Rect(12,5,8,12);
    cv::Rect Roi3= cv::Rect(26,5,8,12);
//    cv::imshow("recorte",image);
//    cv::waitKey(0);
    cv::Mat max1=image(Roi1);
//    max2=image(rectMax2);
//    max3=image(rectMax3);
    int value=render_num(image(Roi1));
    int value2=render_num(image(Roi2));
    int value3=render_num(image(Roi3));
    return value*10+value2+(float)value3/10;

}

cv::Mat temp_min(cv::Mat image, float max, float min, float temp_thres){
    int min_pix=0;
    int max_pix=252;
    int pix_thres=(temp_thres-min)*(max_pix-min_pix)/(max-min);
    //int pix_thres=128;
    cv::Mat result;
    cv::threshold(image,result,pix_thres,255,cv::THRESH_BINARY);
    return result;

}
cv::Mat relative_min(cv::Mat image, float relative_thres){
    int min_pix=0;
    int max_pix=252;
    int pix_thres=(min_pix+max_pix)*relative_thres;
    cv::Mat result;
    cv::threshold(image,result,pix_thres,255,cv::THRESH_BINARY);
    return result;

}
cv::Mat composition(cv::Mat images[],std::string* texts, int vertical, int horizontal){
    int height=images[0].rows;
    int width=images[0].cols;
    cv::Mat im(height*vertical, width*horizontal, CV_8UC3, cv::Scalar(255,255,255));
    for (int i=0;i<vertical;i++){
        for (int j=0; j<horizontal;j++){
            //(images+sizeof(cv::Mat)*(0*horizontal+0))->copyTo(im(cv::Rect(j*width,i*height,width,height)));
            images[i*horizontal+j].copyTo(im(cv::Rect(j*width,i*height,width,height)));
            //cv::imshow("aver",images[i*horizontal+j]);
            //cv::waitKey(0);
        }
    }
    return im;
}

void render_text(cv::Mat* img, std::string text, cv::Point center, cv::Scalar color, int font_type){
    //int font_type=cv::FONT_HERSHEY_SIMPLEX;
    int font_size=1;
    int baseline=0;
    int thickness=1;
    cv::Size text_size=cv::getTextSize(text,font_type,font_size,thickness,&baseline);
    baseline += thickness;
    cv::Point orig(center.x-text_size.width/2,center.y-baseline);
    cv::putText(*img,text,orig,font_type,font_size,color,thickness);
}

int main(int argc, char *argv[])
{
    cout << "Hello World!" << endl;
    cout << get_current_dir() << endl;
    load_numbers();
    cv::VideoCapture cap(1);
    cv::VideoWriter video("output.avi",CV_FOURCC('M','J','P','G'),10, cv::Size(960,480));
    cv::Mat image,min,max,column;
    cv::Rect rectMax= cv::Rect(278,4,38,21);
    cv::Rect rectMin= cv::Rect(278,215,38,21);
    cv::Rect rectColumn= cv::Rect(304,28,12,185);
    int recorte_h=0;
    int recorte_v=25;
    int recorte_h2=15;
    int recorte_v2=20;
//    cv::Rect rectMax1= cv::Rect(281,9,8,12);
//    cv::Rect rectMax2= cv::Rect(290,9,8,12);
//    cv::Rect rectMax3= cv::Rect(304,9,8,12);
//    int n=0;
    while (1){
        cap>> image;
        max= image(rectMax);
        min= image(rectMin);
        column= image(rectColumn);
        float result_max= render_image(max);
        float result_min=render_image(min);
        int rows = image.rows;
        int cols = image.cols;
        cv::Mat recortada (image, cv::Rect(recorte_h, recorte_v, cols-recorte_h-recorte_h2, rows-recorte_v-recorte_v2));
//        cv::Mat umbral=temp_min(recortada,result_max, result_min,30);
//        cv::Mat umbral2=temp_min(recortada,result_max, result_min,40);
//        cv::Mat umbral3=relative_min(recortada,0.5);
//        cv::Mat umbral4=relative_min(recortada,0.75);
//        cv::Mat umbral5=relative_min(recortada,0.9);
        all_images[0]=image;
        all_images[1]=temp_min(recortada,result_max, result_min,20);
        all_images[2]=temp_min(recortada,result_max, result_min,25);
        relative_min(recortada,0.5).copyTo(all_images[3]);
        relative_min(recortada,0.75).copyTo(all_images[4]);
        relative_min(recortada,0.9).copyTo(all_images[5]);
        //uno.copyTo(all_images[3]);
        //uno=relative_min(recortada,0.75);
        //uno.copyTo(all_images[4]);
        //all_images[5]=relative_min(recortada,0.9);

        for (int i=1;i<6;i++){
            cv::copyMakeBorder(all_images[i],all_images[i],0,recorte_v+recorte_v2,recorte_h,recorte_h2,cv::BORDER_CONSTANT,cv::Scalar(255,255,255));
        }
        texts[0]="original";
        texts[1]=">20C";
        texts[2]=">25C";
        texts[3]=">50%";
        texts[4]=">75%";
        texts[5]=">90%";
        cv::Mat composicion=composition(all_images,texts,2,3);
        cv::Point center;
        cv::Scalar color(0,0,255);
        //render_text(&composicion,texts[0],center, cv::Scalar(0,0,255));
        for (int i=0;i<2;i++){
            for (int j=0;j<3;j++){
            if ((j+i)!=0) color=cv::Scalar(0,0,0);
            center.x=cols*(j+0.5);
            center.y=rows*(1+i);
            render_text(&composicion,texts[3*i+j],center, color,4);
            }
        }


//        cout<<"max: "<<result_max<<" min: "<<result_min<<endl;
//        cv::imshow("temp",image);
    video.write(composicion);
cv::imshow("mosaico",composicion);
//        cv::imshow("recortada",recortada);
////        cv::imshow("min",min);
////        cv::imshow("max",max);
//        //        cv::imshow("column",column);
////        cv::imshow("umbral30", umbral);
////        cv::imshow("umbral40", umbral2);
////        cv::imshow("umbral0.5", umbral3);
////        cv::imshow("umbral0.75", umbral4);
////        cv::imshow("umbral0.9", umbral5);
//        cv::imshow("umbral30", all_images[1]);
//        cv::imshow("umbral40", all_images[2]);
//        cv::imshow("umbral0.5", all_images[3]);
//        cv::imshow("umbral0.75", all_images[4]);
//        cv::imshow("umbral0.9", all_images[5]);
char c = (char)cv::waitKey(1);
if( c == 27 )
  break;
//        all_images.clear();

    }
    cap.release();
    video.release();
    return 0;
}
