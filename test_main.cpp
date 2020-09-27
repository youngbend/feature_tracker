#include "tracker.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    Mat img = imread("../tracking_dark.jpg");
    if (img.empty()) {
        cout << "Could not find image!" << endl;
        return -1;
    }
    
    resize(img, img, Size(640,480));
    Mat gray;
    cvtColor(img, gray, COLOR_RGB2GRAY);

    Tracker tracker(4, 4, 5, 25, 15, 15);

    tracker.scan(gray);
    tracker.update_targets(gray);

    vector<point> centers = tracker.get_target_centers();

    cout << "Centers at: " << endl;
    for (auto c : centers) {
        cout << "(" << c.row << "," << c.col << ")" << endl;
        circle(img, Point(c.col, c.row), 3, Scalar( 0, 0, 255 ), FILLED, LINE_8 );
    }
    cout << endl;

    imshow("opencv", img); // Show our image inside the created window.
    waitKey(0);
    destroyAllWindows();
}
