#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/Image.h>
#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "/home/kim/catkin_ws/src/camera_test/include/camera_test/RoadLaneDetector.h" 

//color filtering
Mat RoadLaneDetector::filter_colors(Mat img_frame) {
    Mat output;
    UMat img_hsv;
    UMat white_mask, white_image;
    UMat yellow_mask, yellow_image;
    img_frame.copyTo(output);

    //range of roadlane color
    Scalar lower_white = Scalar(200, 200, 200); //white lane(color)
    Scalar upper_white = Scalar(255, 255, 255);
    Scalar lower_yellow = Scalar(10, 50, 50); //yellow lane(color)
    Scalar upper_yellow = Scalar(40, 255, 255);

    //white filter
    inRange(output, lower_white, upper_white, white_mask);
    bitwise_and(output, output, white_image, white_mask);

    cvtColor(output, img_hsv, COLOR_BGR2HSV);

    //yellow filter
    inRange(img_hsv, lower_yellow, upper_yellow, yellow_mask);
    bitwise_and(output, output, yellow_image, yellow_mask);

    addWeighted(white_image, 1.0, yellow_image, 1.0, 0.0, output);
    return output;
}

Mat RoadLaneDetector::limit_region(Mat img_edges) {
    //masking for detect the edges of the region of interest
    //return a binary image
    int width = img_edges.cols;
    int height = img_edges.rows;

    Mat output;
    Mat mask = Mat::zeros(height, width, CV_8UC1);

    Point points[4]{
        Point((width * (1 - poly_bottom_width)) / 2, height),
        Point((width * (1 - poly_top_width)) / 2, height - height * poly_height),
        Point(width - (width * (1 - poly_top_width)) / 2, height - height * poly_height),
        Point(width - (width * (1 - poly_bottom_width)) / 2, height)
    };

    fillConvexPoly(mask, points, 4, Scalar(255, 0, 0));
    
    bitwise_and(img_edges, mask, output);
    return output;
}
//Probability-applied huff transform straight line detection function
vector<Vec4i> RoadLaneDetector::houghLines(Mat img_mask) {
   
    vector<Vec4i> line;

    HoughLinesP(img_mask, line, 1,  CV_PI / 180, 20, 10, 20);
    return line;
}
//All detected Huff transform straight lines are aligned by slope
vector<vector<Vec4i>> RoadLaneDetector::separateLine(Mat img_edges, vector<Vec4i> lines) {

    vector<vector<Vec4i>> output(2);
    Point p1, p2;
    vector<double> slopes;
    vector<Vec4i> final_lines, left_lines, right_lines;
    double slope_thresh = 0.3;

    for (int i = 0; i < lines.size(); i++) {
        Vec4i line = lines[i];
        p1 = Point(line[0], line[1]);
        p2 = Point(line[2], line[3]);

        double slope;
        if (p2.x - p1.x == 0) //case of the corner
            slope = 999.0;
        else
            slope = (p2.y - p1.y) / (double)(p2.x - p1.x);

        if (abs(slope) > slope_thresh) {
            slopes.push_back(slope);
            final_lines.push_back(line);
        }
    }

    img_center = (double)((img_edges.cols / 2));

    for (int i = 0; i < final_lines.size(); i++) {
        p1 = Point(final_lines[i][0], final_lines[i][1]);
        p2 = Point(final_lines[i][2], final_lines[i][3]);

        if (slopes[i] > 0 && p1.x > img_center && p2.x > img_center) {
            right_detect = true;
            right_lines.push_back(final_lines[i]);
        }
        else if (slopes[i] < 0 && p1.x < img_center && p2.x < img_center ) {
            left_detect = true;
            left_lines.push_back(final_lines[i]);
        }
    }

    output[0] = right_lines;
    output[1] = left_lines;
    return output;
}

//finding the most suitable line both lanes through linear regressinon
vector<Point> RoadLaneDetector::regression(vector<vector<Vec4i>> separatedLines, Mat img_input) {
    vector<Point> output(4);
    Point p1, p2, p3, p4;
    Vec4d left_line, right_line;
    vector<Point> left_points, right_points;
    //right detection
    if (right_detect) {
        for (auto i : separatedLines[0]) {
            p1 = Point(i[0], i[1]);
            p2 = Point(i[2], i[3]);

            right_points.push_back(p1);
            right_points.push_back(p2);
        }

        if (right_points.size() > 0) {
            fitLine(right_points, right_line, DIST_L2, 0, 0.01, 0.01);

            right_m = right_line[1] / right_line[0]; 
            right_b = Point(right_line[2], right_line[3]);
        }
    }
    //left detection
    if (left_detect) {
        for (auto j : separatedLines[1]) {
            p3 = Point(j[0], j[1]);
            p4 = Point(j[2], j[3]);

            left_points.push_back(p3);
            left_points.push_back(p4);
        }

        if (left_points.size() > 0) {
            fitLine(left_points, left_line, DIST_L2, 0, 0.01, 0.01);

            left_m = left_line[1] / left_line[0];  
            left_b = Point(left_line[2], left_line[3]); 
        }
    }

    //y = m*x + b  --> x = (y-b) / m
    int y1 = img_input.rows;
    int y2 = 470;

    double right_x1 = ((y1 - right_b.y) / right_m) + right_b.x;
    double right_x2 = ((y2 - right_b.y) / right_m) + right_b.x;

    double left_x1 = ((y1 - left_b.y) / left_m) + left_b.x;
    double left_x2 = ((y2 - left_b.y) / left_m) + left_b.x;

    output[0] = Point(right_x1, y1);
    output[1] = Point(right_x2, y2);
    output[2] = Point(left_x1, y1);
    output[3] = Point(left_x2, y2);

    return output;
}
// Predict the direction of progress by whether it is on the left or right.
string RoadLaneDetector::predictDir() {

    string output;
    double x, threshold = 10;

    x = (double)(((right_m * right_b.x) - (left_m * left_b.x) - right_b.y + left_b.y) / (right_m - left_m));

    if (x >= (img_center - threshold) && x <= (img_center + threshold))
        output = "Straight";
    else if (x > img_center + threshold)
        output = "Right Turn";
    else if (x < img_center - threshold)
        output = "Left Turn";

    return output;
}
//Visualization of Lane Output
Mat RoadLaneDetector::drawLine(Mat img_input, vector<Point> lane, string dir) {
 
    vector<Point> poly_points;
    Mat output;
    img_input.copyTo(output);
    Point left_endpoint = lane[0];
    Point right_endpoint = lane[1];
    Point left_strpoint = lane[2];
    Point right_strpoint = lane[3];
    left_endpoint.y += 5;
    right_endpoint.y -= 5;
    left_strpoint.y += 5;
    right_strpoint.y -= 5;
    poly_points.push_back(lane[2]);
    poly_points.push_back(lane[0]);
    poly_points.push_back(lane[1]);
    poly_points.push_back(lane[3]);

    fillConvexPoly(output, poly_points, Scalar(0,230, 30), LINE_AA, 0);  
    addWeighted(output, 0.3, img_input, 0.7, 0, img_input);  
    putText(img_input, dir, Point(520, 100), FONT_HERSHEY_PLAIN, 3, Scalar(255, 255, 255), 3, LINE_AA);
    line(img_input, left_endpoint, right_endpoint, Scalar(0, 255, 255), 5, LINE_AA);
    line(img_input, left_strpoint, right_strpoint, Scalar(0, 255, 255), 5, LINE_AA);

    return img_input;
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "image_publisher");
    ros::NodeHandle nh;
    ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("camera_image", 1);

    RoadLaneDetector roadLaneDetector;
    Mat img_frame, img_filter, img_edges, img_mask, img_lines, img_result;
    vector<Vec4i> lines;
    vector<vector<Vec4i> > separated_lines;
    vector<Point> lane;
    string dir;
    std::string videoPath ="/home/kim/catkin_ws/src/camera_test/input.mp4";
    cv::VideoCapture video(videoPath);

    if (!video.isOpened()) {
        ROS_ERROR("Could not open video file.");
        return -1;
    }
    video.read(img_frame);
    if (img_frame.empty()) return -1;

    VideoWriter writer;
    int codec = VideoWriter::fourcc('X', 'V', 'I', 'D'); 
    double fps = 25.0;  
    string filename = "/home/kim/catkin_ws/src/camera_test/result.mp4"; 

    writer.open(filename, codec, fps, img_frame.size(), CV_8UC3);
    if (!writer.isOpened()) {
        cout << "we can't open your video. \n";
        return -1;
    }

    video.read(img_frame);
    int cnt = 0;

    while (ros::ok()) {
        //read the video
        if (!video.read(img_frame)) break;

        //call the filter_colors
        img_filter = roadLaneDetector.filter_colors(img_frame);

        //change the video to grayscale
        cvtColor(img_filter, img_filter, COLOR_BGR2GRAY);

        //extract the edges to Canny Edge Detection
        Canny(img_filter, img_edges, 50, 150);

        //Specify an area of interest to detect only lanes existing on the floor of the vehicle in the direction of travel
        img_mask = roadLaneDetector.limit_region(img_edges);

        //Extracting linear components from edges with Hough transformations
        lines = roadLaneDetector.houghLines(img_mask);

        if (lines.size() > 0) {
            //To find the best line by performing a linear regression
            separated_lines = roadLaneDetector.separateLine(img_mask, lines);
            lane = roadLaneDetector.regression(separated_lines, img_frame);

            //Detect of direction of progress
            dir = roadLaneDetector.predictDir();

            //Outputs predictive directional text to the image 
            img_result = roadLaneDetector.drawLine(img_frame, lane, dir);
        }

        writer << img_result;
        if (cnt++ == 15) 
            imwrite("img_result.jpg", img_result);  
        //saving data and output
        imshow("/home/kim/catkin_ws/src/camera_test/result.mp4", img_result);

        if (waitKey(1) == 27) 
            break;

    }

    return 0;
}

