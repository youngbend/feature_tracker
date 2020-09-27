#include "tracker.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {
    if (argc != 2) return  -1;
    
    Mat img = imread(argv[1]);
    if (img.empty()) {
        cout << "Could not find image!" << endl;
        return -1;
    }
    
    resize(img, img, Size(640,480));
    Mat gray;
    cvtColor(img, gray, COLOR_RGB2GRAY);

    Tracker tracker(2, 2, 6, 20, 15, 15);

    using namespace std::chrono;
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    tracker.scan(gray, 3);
    /* tracker.update_targets(gray); */
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2-t1);
    cout << "Time taken to run: " << time_span.count() << " seconds" << endl;

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
