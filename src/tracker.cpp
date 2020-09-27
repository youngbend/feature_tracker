#include "tracker.h"
#include <iostream>
#include <algorithm>

using namespace std;

void print_vec(const vector<double> &vec) {
    cout << "(";
    for (auto v:vec) {
        cout << v << ',';
    }
    cout << ')' << endl;
}

Tracker::Tracker(int scan_offset_row, int scan_offset_col, int target_offset_in, int threshold_in, int tracking_offset_in, int tracking_timeout_in) :
    row_scan_offset(scan_offset_row), col_scan_offset(scan_offset_col), target_offset(target_offset_in), threshold(threshold_in),
    tracking_offset(tracking_offset_in), tracking_timeout(tracking_timeout_in) { }

vector<point> Tracker::get_target_centers() {
    vector<point> centers;
    for (auto target:targets) {
        centers.push_back(target.get_center());
    }
    return centers;
}

void Tracker::scan(cv::Mat &image) {
    // Update targets
    if (image.channels() > 1) throw TrackerException("Error: Image must be grayscale");

    update_targets(image);

    for (auto iter = targets.begin(); iter != targets.end();) {
        if (iter->loss_count) {
            iter = targets.erase(iter);
        }
        else iter++;
    }

    uint8_t *data = image.data;
    int stride = image.step;

    cout << "Rows = " << image.rows << " Columns = " << image.cols << endl;
    for (int r = 0; r < image.rows; r += row_scan_offset) {
        for (int c = col_scan_offset; c < image.cols; c += col_scan_offset) {
            if (gradient(data[r * stride + c - col_scan_offset], data[r * stride + c]) > threshold) {
                bool inside_target = false;
                for (auto t : targets) {
                    if ((t.center.row - t.radius) <= r && r <= (t.center.row + t.radius) && 
                        (t.center.col - t.radius) <= c && c <= (t.center.col + t.radius)) {
                            inside_target = true;
                            break;
                        }
                }
                
                if (inside_target) {
                    c += col_scan_offset;
                    continue;
                }

                point center;
                int radius;
                
                if (pinpoint_target(image, {r,c}, center, radius)) {
                    targets.push_back(Target(center, radius));
                    data[center.row * stride + center.col - col_scan_offset] = 255;
                }
                

            }
        }
    }

    return;
}

void Tracker::update_targets(cv::Mat &image) {
    uint8_t *data = image.data;
    int stride = image.step;
    
    for (auto t = targets.begin(); t != targets.end();) {
        double row_offset = 0;
        double col_offset = 0;
        double depth_offset = 0;
        double track_offset = tracking_offset;
        if (t->prev_center.row != -1 && t->prev_center.col != -1) {
            row_offset = t->center.row - t->prev_center.row;
            col_offset = t->center.col - t->prev_center.col;
            depth_offset = t->radius - t->prev_radius;
            if (depth_offset < 0) depth_offset = 0;
            if (t->loss_count) {
                double interpolation_factor = 3.0 - 3.25 * exp(-t->loss_count / 2.0);
                row_offset *= interpolation_factor;
                col_offset *= interpolation_factor;
                depth_offset *= interpolation_factor;
                track_offset *= interpolation_factor;
            }
        }
        else if (t->loss_count) tracking_offset *= 1.5;

        int top = int(t->center.row - t->radius + row_offset - depth_offset - track_offset);
        int bottom = int(t->center.row + t->radius + row_offset + depth_offset + track_offset);
        int left = int(t->center.col - t->radius + col_offset - depth_offset - track_offset);
        int right = int(t->center.col + t->radius + col_offset + depth_offset + track_offset);

        if (top < 0 || left < 0 || bottom > image.rows || right > image.cols) {
            t = targets.erase(t);
            continue;
        }

        bool is_found = false;
        point center;
        int radius = 0;

        for (int r = top; r < bottom; r += row_scan_offset) {
            for (int c = left; c < right; c += col_scan_offset) {
                if (gradient(data[r*stride+c-col_scan_offset], data[r*stride+c]) > threshold) {
                    if (pinpoint_target(image, {r,c}, center, radius)) {
                        is_found = true;
                        break;
                    }
                }
            }
            if (is_found) break;
        }

        if (is_found) t->update(center, radius);
        else {
            t->lost();
            if (t->loss_count >= tracking_timeout) {
                t = targets.erase(t);
                continue;
            }
        }

        t++;
    }
}

bool Tracker::pinpoint_target(cv::Mat &image, point start_loc, point &center, int &radius) {
    int new_scan_offset_row = ceil(row_scan_offset / 2.0);
    int new_scan_offset_col = ceil(col_scan_offset / 2.0);

    uint8_t *data = image.data;
    int stride = image.step;

    int initial_left = -1;
    for (int c = start_loc.col; c > 0; c -= new_scan_offset_col) {
        if (gradient(data[start_loc.row * stride + c - new_scan_offset_col], data[start_loc.row * stride + c]) > threshold) {
            initial_left = c;
            break;
        }
    }
    if (initial_left == -1) return false;

    int initial_right = -1;
    for (int c = start_loc.col; c < image.cols; c += new_scan_offset_col) {
        if (gradient(data[start_loc.row * stride + c - new_scan_offset_col], data[start_loc.row * stride + c]) < -threshold) {
            initial_right = c;
            break;
        }
    }
    if (initial_right == -1) return false;

    // Column number of the top left, top right, bottom left, bottom right edges of the bar, in that order
    vector<int> vbar_bounds;

    // Row number of the top and bottom
    int top = -1;
    int bottom = -1;

    int min_intensity = 255;
    vector<int> left_right = {initial_left, initial_right};
    int gradient_counter = 0;

    for (int r = start_loc.row - new_scan_offset_row; r < image.rows; r += new_scan_offset_row) {
        if (r < 0) return false;

        bool edge_trigger = false;
        bool cross_encounter = false;
        int min_row_intensity = 255;

        for (int c = left_right[0] - target_offset; c < left_right[1] + target_offset; c += new_scan_offset_col) {
            if (c < new_scan_offset_col || c >= image.cols) return false;

            if (gradient(data[r * stride + c - new_scan_offset_col], data[r * stride + c]) > threshold) {
                // data[r*stride + c - new_scan_offset_col] = 255;
                cross_encounter = true;
                if (!edge_trigger) {
                    edge_trigger = true;
                    left_right[0] = c;

                    if (top == -1) {
                        top = r;
                        vbar_bounds.push_back(c);
                    }
                }
            }

            else if (gradient(data[r * stride + c - new_scan_offset_col], data[r * stride + c]) < -threshold) {
                // data[r*stride + c - new_scan_offset_col] = 0;
                cross_encounter = true;
                if (edge_trigger) {
                    left_right[1] = c;

                    if (top && vbar_bounds.size() == 1) {
                        vbar_bounds.push_back(c);
                    }

                    continue;
                }
            }

            if (top != -1 && int(data[r*stride+c]) < min_row_intensity) {
                min_row_intensity = int(data[r*stride+c]);
                if (min_row_intensity < min_intensity) {
                    min_intensity = min_row_intensity;
                }
            }
        }

        if (cross_encounter) gradient_counter = 0;

        else if (top != -1 && (min_row_intensity - min_intensity) >= threshold) {
            bottom = r - new_scan_offset_row;
            vbar_bounds.push_back(left_right[0]);
            vbar_bounds.push_back(left_right[1]);
            break;
        }

        else gradient_counter++;

        if (gradient_counter > 1.5 * (initial_right - initial_left)) return false;
    }
    
    if (top == -1 | bottom == -1 || vbar_bounds.size() != 4) return false;

    if (abs((vbar_bounds[1] - vbar_bounds[0]) - (vbar_bounds[3] - vbar_bounds[2])) > target_offset) return false;

    double center_row = (top + bottom) / 2.0;
    double center_column = accumulate(vbar_bounds.begin(), vbar_bounds.end(), 0) / 4.0 + new_scan_offset_col;
    int column_radius = (vbar_bounds[1] - vbar_bounds[0]) / 2;

    if ((vbar_bounds[1] - vbar_bounds[0]) > (bottom - top) / 2) return false;
    
    int left = -1;
    int right = -1;

    vector<int> hbar_bounds;
    
    vector<int> up_down = {int(center_row) - column_radius, int(center_row) + column_radius};
    for (int c = int(center_column) - column_radius; c >= new_scan_offset_col; c -= new_scan_offset_col) {
        bool edge_trigger = false;
        bool cross_encounter = false;
        int min_col_intensity = 255;

        for (int r = up_down[0] - target_offset; r < up_down[1] + target_offset; r += new_scan_offset_row) {
            if (r < 0 || r >= image.rows) return false;
            
            if (gradient(data[(r-new_scan_offset_row)*stride + c], data[r*stride+c]) > threshold) {
                cross_encounter = true;

                if (!edge_trigger) {
                    edge_trigger = true;
                    up_down[0] = r;
                }
            }

            else if (gradient(data[(r-new_scan_offset_row)*stride + c], data[r*stride+c]) < -threshold) {
                cross_encounter = true;
                if (edge_trigger) {
                    up_down[1] = r;
                    continue;
                }
            }

            if (int(data[r*stride+c]) < min_col_intensity) {
                min_col_intensity = int(data[r*stride+c]);
            }
        }

        if (!cross_encounter && abs(min_col_intensity - min_intensity) > threshold) {
            left = c + new_scan_offset_col;
            hbar_bounds.push_back(up_down[0]);
            hbar_bounds.push_back(up_down[1]);
            break;
        }
    }

    up_down = {int(center_row) - column_radius, int(center_row) + column_radius};
    for (int c = int(center_column) + column_radius; c < image.cols; c += new_scan_offset_col) {
        bool edge_trigger = false;
        bool cross_encounter = false;
        int min_col_intensity = 255;

        for (int r = up_down[0] - target_offset; r < up_down[1] + target_offset; r += new_scan_offset_row) {
            if (r < 0 || r >= image.rows) return false;
            
            if (gradient(data[(r-new_scan_offset_row)*stride + c], data[r*stride+c]) > threshold) {
                cross_encounter = true;

                if (!edge_trigger) {
                    edge_trigger = true;
                    up_down[0] = r;
                }
            }

            else if (gradient(data[(r-new_scan_offset_row)*stride + c], data[r*stride+c]) < -threshold) {
                cross_encounter = true;
                if (edge_trigger) {
                    up_down[1] = r;
                    continue;
                }
            }

            if (int(data[r*stride+c]) < min_col_intensity) {
                min_col_intensity = int(data[r*stride+c]);
            }
        }

        if (!cross_encounter && abs(min_col_intensity - min_intensity) > threshold) {
            right = c - new_scan_offset_col;
            hbar_bounds.push_back(up_down[0]);
            hbar_bounds.push_back(up_down[1]);
            break;
        }
    }

    if (left == -1 || right == -1 || hbar_bounds.size() != 4) return false;

    if (abs((hbar_bounds[1] - hbar_bounds[0]) - (hbar_bounds[3] - hbar_bounds[2])) > target_offset) return false;

    if (hbar_bounds[1] - hbar_bounds[0] > (right - left) / 2) return false;

    center_column = (right + left) / 2.0;

    vector<double> up_vec = {(vbar_bounds[1] + vbar_bounds[0]) / 2.0 - center_column, double(center_row - top)};
    if (all_of(up_vec.begin(), up_vec.end(), [](double i) { return i == 0; })) return false;
    vector<double> up_unit = unitVec(up_vec);

    vector<double> down_vec = {(vbar_bounds[3] + vbar_bounds[2]) / 2.0 - center_column, double(center_row - bottom)};
    if (all_of(down_vec.begin(), down_vec.end(), [](double i) { return i == 0; })) return false;
    vector<double> down_unit = unitVec(down_vec);

    vector<double> right_vec = {double(right - center_column), center_row - (hbar_bounds[3] + hbar_bounds[2]) / 2.0};
    if (all_of(right_vec.begin(), right_vec.end(), [](double i) { return i == 0; })) return false;
    vector<double> right_unit = unitVec(right_vec);

    vector<double> left_vec = {double(left - center_column), center_row - (hbar_bounds[1] + hbar_bounds[0]) / 2.0};
    if (all_of(left_vec.begin(), left_vec.end(), [](double i) { return i == 0; })) return false;
    vector<double> left_unit = unitVec(left_vec);

    if (abs(L2Norm(up_vec) - L2Norm(right_vec)) > 2 * target_offset) return false;

    if (L2Norm(down_vec) < target_offset || L2Norm(left_vec) < target_offset) return false;

    // DEBUG
    // data[top*stride + (vbar_bounds[1] + vbar_bounds[0])/2] = 255;
    // data[bottom*stride + (vbar_bounds[3] + vbar_bounds[2])/2] = 255;
    // data[(hbar_bounds[1]+hbar_bounds[0])/2*stride + left] = 255;
    // data[(hbar_bounds[3]+hbar_bounds[2])/2*stride + right] = 255;

    // cout << "Center at: (" << center_row << "," << center_column << ")" << endl;

    // print_vec(up_unit);
    // print_vec(down_unit);
    // print_vec(right_unit);
    // print_vec(left_unit);

    // cout << "Orthogonals: " << dot(up_unit, right_unit) << "," << dot(up_unit, left_unit) << "," << dot(left_unit, down_unit) << "," << dot(right_unit, down_unit) << endl;
    // cout << "Parallels: " << dot(up_unit, down_unit) << "," << dot(right_unit, left_unit) << endl;
    // END DEBUG

    if (abs(dot(up_unit, right_unit)) > 0.35) return false;
    if (abs(dot(up_unit, left_unit)) > 0.35) return false;
    if (abs(dot(left_unit, down_unit)) > 0.35) return false;
    if (abs(dot(right_unit, down_unit)) > 0.35) return false;
    if (abs(abs(dot(up_unit, down_unit)) - 1) > 0.15) return false;
    if (abs(abs(dot(right_unit, left_unit)) - 1) > 0.15) return false;

    center = {int(center_row), int(center_column)};
    radius = max(bottom - top, right - left) / 2;

    cout << "Center at: (" << center_row << "," << center_column << ")" << " with radius = " << radius << endl;

    return true;
}
