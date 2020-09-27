#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <list>
#include <math.h>
#include <numeric>
#include <mutex>
#include <opencv2/opencv.hpp>

class Tracker;

struct point {
    int row;
    int col;
};

class Target {
public:
    Target(point center_in, int radius_in) : 
        center(center_in), prev_center({-1,-1}), radius(radius_in), prev_radius(-1), loss_count(0), dead(false) {}
    
    void update(point center_in, int radius_in) {
        prev_center = center;
        prev_radius = radius;
        loss_count = 0;
        dead = false;
        center = center_in;
        radius = radius_in;
    }

    void lost() { loss_count++; }

    void kill() { dead = true; }

    point get_center() { return center; }

    friend class Tracker;

protected:
    point center;
    point prev_center;
    int radius;
    int prev_radius;
    int loss_count;
    bool dead;
};

class Tracker {
public:
    Tracker(int scan_offset_row, int scan_offset_col, int target_offset_in, int threshold_in, int tracking_offset_in, int tracking_timeout_in);

    std::vector<point> get_target_centers();

    void scan(cv::Mat &image, int n_threads);

    void update_targets(cv::Mat &image);

private:
    int row_scan_offset;
    int col_scan_offset;
    int target_offset;
    int threshold;
    int tracking_offset;
    int tracking_timeout;

    std::mutex target_lock;

    std::list<Target> targets;

    void scan_thread(cv::Mat &image, int lbound, int rbound);
    void update_targets_thread(cv::Mat &image, Target &target);
    bool pinpoint_target(cv::Mat &image, point start_loc, point &center, int &radius);
};

class TrackerException {
public:
    TrackerException(std::string err_msg) : err(err_msg) {}
    std::string get_msg() { return err; }
private:
    std::string err;
};

inline int gradient(uint8_t left, uint8_t right) { return static_cast<int>(left) - static_cast<int>(right); }

inline double dot(const std::vector<double> &first, const std::vector<double> &second) {
    return std::inner_product(first.begin(), first.end(), second.begin(), 0.0);
}

inline double L2Norm(const std::vector<double> &vec) { return sqrt(dot(vec, vec)); }

inline std::vector<double> unitVec(const std::vector<double> &vec) {
    double norm = L2Norm(vec);
    std::vector<double> ret;
    for (auto v : vec) {
        ret.push_back(v / norm);
    }
    return ret;
}

#endif
